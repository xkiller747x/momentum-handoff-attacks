"""Explicit targeted experiment commands; existing untargeted commands unchanged."""
import contextlib
import math
from pathlib import Path
import time
import numpy as np
import torch

from baseline.targeted import (FEATURE_METHODS,PROFILES,FeatureTap,ForwardCounter,
    choose_reference,feature_distance,optimize,target_metrics)
from . import runner
from .io import append_jsonl,image_tensor,read_json,read_jsonl,read_pixels,save_adversarial,sha256,write_json
from .metrics import image_seed
from .models import load_model
from .targeted_data import target_labels,reference_gallery


def checkpoints(value,steps):
    values=([s for s in (10,30,100,200,299) if s<=steps] if value is None else [int(s) for s in value.split(',')])
    if any(s<0 or s>steps or (s==0 and steps>0) for s in values):raise ValueError('Checkpoint outside actual attack steps')
    return sorted(set([*values,steps]))


def verify_saved(row):
    for entry in row['checkpoints'].values():
        if sha256(entry['png_path'])!=entry['png_sha256']:raise ValueError('Saved attack checkpoint PNG changed')
        if entry.get('float_path') and sha256(entry['float_path'])!=entry['float_sha256']:raise ValueError('Saved float checkpoint changed')


def attack_command(args):
    if args.steps<0 or not all(math.isfinite(v) and v>=0 for v in (args.epsilon_bytes,args.alpha_bytes)):raise ValueError('Invalid attack steps/budget/step size')
    feature=args.method in FEATURE_METHODS
    if feature and (not args.layer or not args.reference_manifest):raise ValueError('Feature attacks require --layer and --reference-manifest')
    if not feature and (args.layer or args.reference_manifest):raise ValueError('Logit attacks do not use feature layer/gallery')
    if args.reference_rule=='index' and args.reference_index is None:raise ValueError('Index rule requires --reference-index')
    if args.reference_rule!='index' and args.reference_index is not None:raise ValueError('--reference-index requires index rule')
    points=checkpoints(args.checkpoints,args.steps)
    records=runner._records(args);labels=target_labels(args.labels_csv,records)
    output=runner._output(args.output);device=runner._device(args.device)
    gallery=reference_gallery(args.reference_manifest) if feature else None
    model,identity=load_model(args.source,device)
    config=runner._config(args,records,kind='targeted-attack',source_identity=identity,
        labels_csv_sha256=sha256(args.labels_csv),target_labels=labels,profile=PROFILES[args.method],
        reference_manifest_sha256=sha256(args.reference_manifest) if feature else None,checkpoint_steps=points,
        budget_units='epsilon_bytes and alpha_bytes divided by255 independently; no implicit epsilon/T override',
        source_endpoint_policy='One clean forward;10 reference forwards for feature methods; two source forwards (float,PNG) per checkpoint including final. Final/checkpoint PNG deduplicated.',
        elapsed_seconds_profile='Per-image end-to-end: reference image IO/forwards/selection, optimization, checkpoint PNG/float IO and source endpoint forwards; excludes model loading and clean input PNG loading. Not pure attack-forward time.')
    completed=runner._open_run(output,config)
    record_map={r['id']:r for r in records}
    for row in completed:
        if row['id'] not in record_map or row['target_label']!=labels[row['id']] or row['canonical_sha256']!=record_map[row['id']]['canonical_sha256']:raise ValueError('Completed image identity differs')
        runner._check_clean(record_map[row['id']]);verify_saved(row)
    done={r['id'] for r in completed}
    for record in records:
        if record['id'] in done:continue
        runner._check_clean(record)
        clean=image_tensor(record['canonical_path'],device);target=labels[record['id']]
        if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();saved={};candidates=[];reference=None;selected=None
        tap_context=FeatureTap(model,args.layer) if feature else contextlib.nullcontext(None)
        with ForwardCounter(model) as count,tap_context as tap:
            count.phase='clean'
            with torch.no_grad():
                logits=model(clean);source_clean=target_metrics(logits,target,record['label'])
                clean_feature=tap.value.detach().clone() if feature else None
            if feature:
                features={}
                for ref in gallery[target]:
                    if sha256(ref['canonical_path'])!=ref['canonical_sha256']:raise ValueError('Reference PNG SHA changed')
                    count.phase='reference'
                    with torch.no_grad():
                        ref_logits=model(image_tensor(ref['canonical_path'],device));ref_feature=tap.value.detach()
                        distance=float(feature_distance(clean_feature,ref_feature,args.method).item())
                        candidates.append({'id':ref['id'],'label':ref['label'],'canonical_path':ref['canonical_path'],
                            'canonical_sha256':ref['canonical_sha256'],'distance':distance,
                            'source':target_metrics(ref_logits,target,ref['label']),
                            'source_logits':ref_logits[0].detach().cpu().tolist()})
                        features[ref['id']]=ref_feature.cpu().clone()
                selected=choose_reference(candidates,args.reference_rule,args.reference_index)
                reference=features[selected['id']].to(device);del features
                d0=selected['distance']
            else:d0=None

            def endpoint(step,image):
                final=step==args.steps
                path=output/'images'/f'{record["id"]}.png' if final else output/'checkpoints'/str(step)/'images'/f'{record["id"]}.png'
                count.phase='checkpoint_float'
                with torch.no_grad():
                    float_logits=model(image);float_metrics=target_metrics(float_logits,target,record['label'])
                    distance=float(feature_distance(tap.value,reference,args.method).item()) if feature else None
                    if distance is not None and not math.isfinite(distance):raise ValueError('Nonfinite checkpoint feature distance')
                png_metrics=save_adversarial(path,image,read_pixels(record['canonical_path']),args.epsilon_bytes)
                count.phase='checkpoint_png'
                with torch.no_grad():
                    png_logits=model(image_tensor(path,device));png_endpoint=target_metrics(png_logits,target,record['label'])
                    png_distance=float(feature_distance(tap.value,reference,args.method).item()) if feature else None
                    if png_distance is not None and not math.isfinite(png_distance):raise ValueError('Nonfinite PNG feature distance')
                item={'step':step,'png_path':str(path),'float_source':float_metrics,'png_source':png_endpoint,
                      'feature_loss':distance,'png_feature_loss':png_distance,**png_metrics}
                if args.save_float:
                    float_path=path.with_suffix('.npz');np.savez_compressed(float_path,image=image.detach().cpu().numpy())
                    item.update(float_path=str(float_path),float_sha256=sha256(float_path))
                saved[str(step)]=item

            _image,stats=optimize(model,clean,target,args.method,args.steps,args.epsilon_bytes,args.alpha_bytes,
                image_seed(args.seed,record['id']),tap,reference,points,endpoint,count)
            final=saved[str(args.steps)]
            expected=args.steps+1+(10 if feature else 0)+2*len(points)
            if count.total!=expected or count.counts.get('optimization',0)!=args.steps:raise ValueError('Actual/declaration source forward mismatch')
            actual={'forward_images':count.total,'by_phase':dict(count.counts),'resolution_counts':dict(count.resolutions)}
            forward_pixels=sum(int(shape.split('x')[0])*int(shape.split('x')[1])*n for shape,n in count.resolutions.items())
        if device.type=='cuda':torch.cuda.synchronize(device)
        row={'id':record['id'],'split':record['split'],'label':record['label'],'target_label':target,
            'canonical_sha256':record['canonical_sha256'],'image_seed':image_seed(args.seed,record['id']),
            'method':args.method,'profile':PROFILES[args.method],'layer':args.layer,
            'reference_rule':args.reference_rule if feature else None,'selected_reference_id':selected['id'] if feature else None,
            'reference_candidates':candidates,'d0':d0,'dT':final['feature_loss'],
            'relative_reduction':(d0-final['feature_loss'])/d0 if feature and d0!=0 else None,
            'source_clean':source_clean,'source_float':final['float_source'],'source_png':final['png_source'],
            'checkpoints':saved,'png_path':final['png_path'],'png_sha256':final['png_sha256'],
            'forward_images':actual['forward_images'],'reference_forward_images':actual['by_phase'].get('reference',0),
            'forward_pixels':forward_pixels,'input_gradient_pixels':args.steps*clean.shape[-2]*clean.shape[-1],
            'clean_forward_images':actual['by_phase'].get('clean',0),'endpoint_forward_images':actual['by_phase'].get('checkpoint_float',0)+actual['by_phase'].get('checkpoint_png',0),
            'actual_forward_hook':actual,'elapsed_seconds':time.perf_counter()-start,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            **stats}
        append_jsonl(output/'progress.jsonl',row);completed.append(row)
        print(f'targeted-attack {args.method} {record["id"]} target={target}',flush=True)
    summary={'kind':'targeted-attack','count':len(completed),'expected_images':len(records),'complete':len(completed)==len(records),
        'forward_images':sum(r['forward_images'] for r in completed),'input_gradient_images':sum(r['input_gradient_images'] for r in completed),
        'forward_pixels':sum(r['forward_pixels'] for r in completed),'input_gradient_pixels':sum(r['input_gradient_pixels'] for r in completed),
        'reference_forward_images':sum(r['reference_forward_images'] for r in completed),'target_forward_images':0}
    write_json(output/'summary.json',summary);return summary


def denominator_summary(rows):
    result={}
    for name,eligible in [('all',rows),('target-clean',[r for r in rows if r['clean']['true_correct']]),
                          ('source-target-clean',[r for r in rows if r['clean']['true_correct'] and r['source_clean_correct']])]:
        successes=sum(r['adversarial']['targeted_success'] for r in eligible)
        result[name]={'denominator':len(eligible),'denominator_ids':[r['id'] for r in eligible],
            'targeted_successes':successes,'targeted_success_rate':successes/len(eligible) if eligible else None,
            'adversarial_true_accuracy':sum(r['adversarial']['true_correct'] for r in eligible)/len(eligible) if eligible else None}
    return result


def evaluate_command(args):
    records=runner._records(args);labels=target_labels(args.labels_csv,records)
    output=runner._output(args.output);device=runner._device(args.device);attack_root=Path(args.attack_run).resolve()
    attack_config=read_json(attack_root/'config.json')
    if attack_config.get('kind')!='targeted-attack' or attack_config['manifest_sha256']!=sha256(args.manifest) or attack_config['labels_csv_sha256']!=sha256(args.labels_csv):raise ValueError('Targeted attack manifest/target CSV differs')
    attacked={r['id']:r for r in read_jsonl(attack_root/'progress.jsonl')}
    selected={r['id'] for r in records}
    if not selected.issubset(attacked):raise ValueError('Selected images are not complete in attack run')
    for record in records:
        row=attacked[record['id']]
        if row['target_label']!=labels[record['id']] or row['label']!=record['label'] or row['canonical_sha256']!=record['canonical_sha256']:raise ValueError('Attack target/image identity differs')
        verify_saved(row)
    names=args.targets.split(',')
    if len(set(names))!=len(names):raise ValueError('Duplicate target models')
    identities={}
    for name in names:
        model,identities[name]=load_model(name,'cpu');del model
    config=runner._config(args,records,kind='targeted-evaluate',labels_csv_sha256=sha256(args.labels_csv),
        target_labels=labels,target_identities=identities,attack_config_sha256=sha256(attack_root/'config.json'),
        attack_progress_sha256=sha256(attack_root/'progress.jsonl'),source_clean_reference='Explicit source_clean endpoint in joined attack progress; no additional source forwards')
    completed=runner._open_run(output,config);done={(r['id'],r['model']) for r in completed}
    for name in names:
        model,loaded=load_model(name,device)
        if loaded!=identities[name]:raise ValueError('Target model identity changed')
        if name==attack_config['arguments']['source'] and loaded!=attack_config['source_identity']:raise ValueError('Evaluation source identity differs from attack source')
        for record in records:
            if (record['id'],name) in done:continue
            runner._check_clean(record);row=attacked[record['id']]
            key=str(args.checkpoint) if args.checkpoint is not None else str(attack_config['arguments']['steps'])
            if key not in row['checkpoints']:raise ValueError('Requested checkpoint absent from attack')
            endpoint=row['checkpoints'][key];target=labels[record['id']]
            if device.type=='cuda':torch.cuda.synchronize(device);torch.cuda.reset_peak_memory_stats(device)
            start=time.perf_counter()
            with ForwardCounter(model) as count,torch.no_grad():
                count.phase='clean';clean=target_metrics(model(image_tensor(record['canonical_path'],device)),target,record['label'])
                count.phase='adversarial_png';adv=target_metrics(model(image_tensor(endpoint['png_path'],device)),target,record['label'])
                if count.total!=2:raise ValueError('Target evaluation actual cost differs')
            if device.type=='cuda':torch.cuda.synchronize(device)
            if name==attack_config['arguments']['source'] and clean['prediction']!=row['source_clean']['prediction']:raise ValueError('Source clean argmax differs')
            value={'id':record['id'],'split':record['split'],'label':record['label'],'target_label':target,'model':name,
                **{key:row[key] for key in ('batch_id','cluster_id','batch_size','batch_image_ids','batch_rng_seed','odd_singleton') if key in row},
                'canonical_sha256':record['canonical_sha256'],'image_seed':row['image_seed'],
                'attack_method':row['method'],'attack_profile':row['profile'],'layer':row['layer'],'selected_reference_id':row['selected_reference_id'],
                'd0':row['d0'],'checkpoint_float_feature_loss':endpoint['feature_loss'],'checkpoint_png_feature_loss':endpoint['png_feature_loss'],
                'checkpoint_step':int(key),'png_sha256':endpoint['png_sha256'],'clean':clean,'adversarial':adv,
                'source_clean_correct':row['source_clean']['true_correct'],'actual_forward_images':count.total,
                'forward_images':count.total,'forward_pixels':sum(int(shape.split('x')[0])*int(shape.split('x')[1])*n for shape,n in count.resolutions.items()),
                'resolution_counts':dict(count.resolutions),'input_gradient_images':0,'input_gradient_pixels':0,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                'elapsed_seconds':time.perf_counter()-start}
            append_jsonl(output/'progress.jsonl',value);completed.append(value)
            print(f'targeted-evaluate {name} {record["id"]}',flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    summary={'kind':'targeted-evaluate','selected_images':len(records),'expected_rows':len(records)*len(names),'count':len(completed),
        'complete':len(completed)==len(records)*len(names),'forward_images':sum(r['forward_images'] for r in completed),
        'forward_pixels':sum(r['forward_pixels'] for r in completed),'input_gradient_images':0,'input_gradient_pixels':0,
        'models':{name:denominator_summary([r for r in completed if r['model']==name]) for name in names}}
    write_json(output/'summary.json',summary);return summary


def report_command(args):
    from .matrix_report import _csv
    output=runner._output(args.output);runs=[];images=[];table=[]
    for value in args.runs:
        root=Path(value).resolve();config=read_json(root/'config.json')
        if config.get('kind')!='targeted-evaluate':raise ValueError('targeted-report requires targeted evaluation directories')
        rows=read_jsonl(root/'progress.jsonl');expected={(i,m) for i in config['selected_ids'] for m in config['target_identities']}
        keys=[(r['id'],r['model']) for r in rows]
        if len(set(keys))!=len(keys) or not set(keys).issubset(expected):raise ValueError('Unexpected/duplicate evaluation rows')
        complete=set(keys)==expected
        item={'run':str(root),'complete':complete,'status':'complete' if complete else 'partial','observed_rows':len(rows),'expected_rows':len(expected),
            'checkpoint_step':config['arguments'].get('checkpoint'),'models':{name:denominator_summary([r for r in rows if r['model']==name]) for name in config['target_identities']}}
        runs.append(item);images.extend({'run':str(root),**r} for r in rows)
        for name,populations in item['models'].items():
            for population,stats in populations.items():table.append({'run':str(root),'model':name,'population':population,'status':item['status'],'observed_rows':len([r for r in rows if r['model']==name]),'expected_images':len(config['selected_ids']),**stats})
    write_json(output/'report.json',{'runs':runs,'interpretation':'Targeted top1 means argmax equals the specified target; arbitrary wrong classes are failures. Partial denominators are observed rows only, never claimed as full split.'})
    _csv(output/'summary.csv',table);_csv(output/'images.csv',images)
    lines=['# Targeted transfer summary','','All-image targeted top1 is primary. Conditional rates retain explicit denominator IDs.','',
        '| Run | Model | Population | Status | Success/denominator | Rate |','|---|---|---|---|---:|---:|']
    for row in table:lines.append(f'| {row["run"]} | {row["model"]} | {row["population"]} | {row["status"]} | {row["targeted_successes"]}/{row["denominator"]} | {row["targeted_success_rate"]} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'output':str(output),'runs':len(runs),'complete_runs':sum(r['complete'] for r in runs)}
