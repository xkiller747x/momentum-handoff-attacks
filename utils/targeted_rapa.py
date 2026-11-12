"""Local single-image RaPA author-hook runner; no implicit target evaluation."""
import math
from pathlib import Path
import time
import numpy as np
import torch
from baseline.rapa import AuthorMask,digest,generator_sha,module_table,optimize_rapa
from baseline.targeted import ForwardCounter,target_metrics
from . import runner
from .io import append_jsonl,image_tensor,read_jsonl,read_pixels,save_adversarial,sha256,write_json
from .metrics import image_seed
from .models import load_model
from .targeted_data import target_labels
from .targeted_runner import checkpoints,verify_saved


def stream_seeds(seed,identity):
    return {'rdi_cpu':image_seed(seed,'rapa-rdi:'+identity),'mask_device':image_seed(seed,'rapa-mask:'+identity)}


def pixel_count(count):return sum(int(k.split('x')[0])*int(k.split('x')[1])*n for k,n in count.resolutions.items())
def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def attack_command(args):
    if args.steps<0 or args.samples<1 or not all(math.isfinite(x) and x>=0 for x in (args.epsilon_bytes,args.alpha_bytes)) or not 0<=args.drop_probability<=1:raise ValueError('Invalid RaPA configuration')
    records=runner._records(args);labels=target_labels(args.labels_csv,records);points=checkpoints(args.checkpoints,args.steps)
    output=runner._output(args.output);device=runner._device(args.device);model,identity=load_model('rn50',device)
    table=module_table(model);table_sha=digest(table)
    profile=f'rapa-author-hook-T{args.steps}-S{args.samples}-local224-V2'
    config=runner._config(args,records,kind='targeted-attack',source_identity=identity,labels_csv_sha256=sha256(args.labels_csv),
        target_labels=labels,profile=profile,checkpoint_steps=points,mask_modules=table,module_table_sha256=table_sha,
        dual_stream_seeds={r['id']:stream_seeds(args.seed,r['id']) for r in records},
        adaptation='Author pretrained=True RN50V1/NIPS original299 ToTensor versus local RN50V2/shared canonical224. Single-image, local T/S; not original T1000/S5/batch32 numerical reproduction.',
        mask_semantics='Author .data pre-hook Bernoulli weight/(rand>p) bias, no rescaling; immediate post-hook restoration before backward. No functional masked-model substitution. p0 still draws and preserves rand==0 bias boundary.',
        random_streams='RDI dedicated CPU generator; parameter masks dedicated parameter-device generator. Same-device distribution and repeatability; CPU/GPU seeds do not imply equal masks.',
        cost_profile='Per-image end-to-end mask snapshot/verification, optimization, checkpoint float/PNG endpoint IO/forwards; excludes model load and initial clean PNG load. All source forwards count; S*T input gradients, no hidden target forwards.')
    rows=runner._open_run(output,config);record_map={r['id']:r for r in records}
    for row in rows:
        if row['id'] not in record_map or row['target_label']!=labels[row['id']] or row['module_table_sha256']!=table_sha:raise ValueError('Completed RaPA identity differs')
        runner._check_clean(record_map[row['id']]);verify_saved(row)
    done={r['id'] for r in rows}
    for record in records:
        if record['id'] in done:continue
        runner._check_clean(record);clean=image_tensor(record['canonical_path'],device);target=labels[record['id']]
        seeds=stream_seeds(args.seed,record['id']);rdi_gen=torch.Generator(device='cpu').manual_seed(seeds['rdi_cpu'])
        mask_gen=torch.Generator(device=device).manual_seed(seeds['mask_device']);initial_states={'rdi_cpu':generator_sha(rdi_gen),'mask_device':generator_sha(mask_gen)}
        sync(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();saved={};counter=ForwardCounter(model);mask=None
        try:
            with counter,AuthorMask(model,mask_gen,args.drop_probability) as mask:
                counter.phase='clean'
                with mask.disabled(),torch.no_grad():source_clean=target_metrics(model(clean),target,record['label'])
                def endpoint(step,image):
                    path=output/'images'/f'{record["id"]}.png' if step==args.steps else output/'checkpoints'/str(step)/'images'/f'{record["id"]}.png'
                    with mask.disabled(),torch.no_grad():
                        counter.phase='checkpoint_float';floating=target_metrics(model(image),target,record['label'])
                        stats=save_adversarial(path,image,read_pixels(record['canonical_path']),args.epsilon_bytes)
                        counter.phase='checkpoint_png';encoded=target_metrics(model(image_tensor(path,device)),target,record['label'])
                    entry={'step':step,'png_path':str(path),'float_source':floating,'png_source':encoded,'feature_loss':None,'png_feature_loss':None,**stats}
                    if args.save_float:
                        float_path=path.with_suffix('.npz');np.savez_compressed(float_path,image=image.detach().cpu().numpy())
                        entry.update(float_path=str(float_path),float_sha256=sha256(float_path))
                    saved[str(step)]=entry
                _image,stats=optimize_rapa(mask,clean,target,args.steps,args.samples,args.epsilon_bytes,args.alpha_bytes,rdi_gen,points,endpoint,counter)
                restored=mask.verify();bn=mask.bn_record()
                expected=args.steps*args.samples+1+2*len(points)
                if counter.total!=expected or counter.counts.get('optimization',0)!=args.steps*args.samples:raise ValueError('RaPA actual F/B count differs')
                if mask.counts['module_mask_calls']!=args.steps*args.samples*len(table) or mask.counts['module_restore_calls']!=mask.counts['module_mask_calls']:raise ValueError('RaPA mask/restore count differs')
            sync(device);final=saved[str(args.steps)]
            row={'id':record['id'],'split':record['split'],'label':record['label'],'target_label':target,'canonical_sha256':record['canonical_sha256'],
                'image_seed':image_seed(args.seed,record['id']),'dual_stream_seeds':seeds,'initial_generator_state_sha256':initial_states,
                'final_generator_state_sha256':{'rdi_cpu':generator_sha(rdi_gen),'mask_device':generator_sha(mask_gen)},
                'mask_generator_device':str(device),'module_table_sha256':table_sha,'mask_restore_counts':dict(mask.counts),
                'parameters_restored':restored,'restoration_checks':'Every masked forward: counts/storage pointers; per-image end and exit: bitwise values/BN, BN SHA before/after.',
                'bn_statistics':bn,'method':'rapa','profile':profile,'layer':None,
                'selected_reference_id':None,'reference_rule':None,'reference_candidates':[],'d0':None,'dT':None,'relative_reduction':None,
                'source_clean':source_clean,'source_float':final['float_source'],'source_png':final['png_source'],'checkpoints':saved,
                'png_path':final['png_path'],'png_sha256':final['png_sha256'],'forward_images':counter.total,'forward_pixels':pixel_count(counter),
                'input_gradient_pixels':args.steps*args.samples*clean.shape[-2]*clean.shape[-1],
                'reference_forward_images':0,'clean_forward_images':1,'endpoint_forward_images':2*len(points),
                'actual_forward_hook':{'forward_images':counter.total,'by_phase':dict(counter.counts),'resolution_counts':dict(counter.resolutions)},
                'elapsed_seconds':time.perf_counter()-start,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,**stats}
            append_jsonl(output/'progress.jsonl',row);rows.append(row)
            print(f'targeted-rapa {record["id"]} T={args.steps} S={args.samples}',flush=True)
        except BaseException as error:
            sync(device)
            append_jsonl(output/'failed_attempts.jsonl',{'id':record['id'],'error':f'{type(error).__name__}:{error}',
                'forward_images_observed':counter.total,'forward_pixels_observed':pixel_count(counter),'elapsed_seconds':time.perf_counter()-start,
                'mask_restore_counts':dict(mask.counts) if mask else None,'input_gradient_cost':'in-flight cost not inferred'})
            raise
    summary={'kind':'targeted-attack','count':len(rows),'expected_images':len(records),'complete':len(rows)==len(records),'profile':profile,
        'forward_images':sum(r['forward_images'] for r in rows),'input_gradient_images':sum(r['input_gradient_images'] for r in rows),
        'forward_pixels':sum(r['forward_pixels'] for r in rows),'input_gradient_pixels':sum(r['input_gradient_pixels'] for r in rows),
        'reference_forward_images':0,'target_forward_images':0}
    write_json(output/'summary.json',summary);return summary
