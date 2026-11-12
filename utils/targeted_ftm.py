"""Serial FTM batches with fixed pairs, explicit preparation cost and atomic commits."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
import torch

from baseline.ftm import FeatureTuning,layer_digest,optimize_ftm,qualification
from baseline.ftm_sphere import STATE_CONTROL_SPEC,UnitStateControl,StateTransformError,validate_state_control,state_profile
from baseline.ftm_initialization import FirstInitializationControl,validate_initialization,initialization_profile
from baseline.targeted import ForwardCounter,target_metrics
from . import runner
from .io import append_jsonl,image_tensor,read_json,read_jsonl,read_pixels,save_adversarial,sha256,write_json
from .metrics import image_seed
from .models import load_model
from .targeted_data import target_labels
from .targeted_runner import checkpoints,verify_saved
from .ftm_update import UpdateTrace
from .original299 import input_spec,model_identity
from . import ftm_cpu_audit
from .ftm_efficacy import ConvEfficacyProbe,validate_mode as efficacy_mode,terminal_u_digest


def pair_table(records,batch_size,seed,cpu_audit=False):
    if batch_size not in (1,2,20):raise ValueError('FTM batch size must be1,2 or explicit CPU audit20')
    if (batch_size==20)!=bool(cpu_audit):raise ValueError('Batch20 and CPU audit flag must be used together')
    ids=sorted(r['id'] for r in records)
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate cohort IDs')
    if cpu_audit and len(ids)!=20:raise ValueError('CPU audit requires one complete20-member group')
    result=[]
    for start in range(0,len(ids),batch_size):
        members=ids[start:start+batch_size]
        digest=hashlib.sha256(json.dumps(members,separators=(',',':')).encode()).hexdigest()
        batch_id=('batch20-' if cpu_audit else 'pair-')+digest
        result.append({'batch_id':batch_id,'cluster_id':batch_id,'image_ids':members,'batch_size':len(members),
            'odd_singleton':len(members)<batch_size,'rng_seed':image_seed(seed,batch_id)})
    return result


def atomic_progress(path,rows):
    temporary=path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    temporary.replace(path)


def pixels(resolutions):return sum(int(shape.split('x')[0])*int(shape.split('x')[1])*n for shape,n in resolutions.items())
def utc():return datetime.now(timezone.utc).isoformat()
def sync(device):
    if device.type=='cuda':torch.cuda.synchronize(device)


def ftm_beta(value):
    beta=float(value)
    if not math.isfinite(beta) or beta<0:raise ValueError('FTM beta must be finite and nonnegative')
    return 0.0 if beta==0 else beta


def beta_profile(profile,beta,mode):
    if beta==.01:return profile
    suffix='-update-'+mode
    if not profile.endswith(suffix):raise ValueError('Unexpected FTM profile suffix')
    label='beta0' if beta==0 else 'beta'+repr(beta)
    return profile[:-len(suffix)]+'-'+label+suffix


def final_u_nonzero(context):
    """Read-only final parameter-state count, summed across learned layers."""
    counts=None
    for value in context.outputs_tuning.values():
        current=torch.count_nonzero(value.detach().reshape(value.shape[0],-1),dim=1)
        counts=current if counts is None else counts+current
    if counts is None:raise ValueError('Final FTM state has no learned parameters')
    return [int(value) for value in counts.tolist()]


def attack_command(args):
    if args.steps<0 or not all(math.isfinite(v) and v>=0 for v in (args.epsilon_bytes,args.alpha_bytes)):raise ValueError('Invalid FTM step/budget')
    beta=ftm_beta(getattr(args,'ftm_beta',.01))
    state_control=validate_state_control(getattr(args,'ftm_state_control','raw'),args.ftm_update_mode)
    if hasattr(args,'ftm_beta'):args.ftm_beta=beta
    probe_mode=efficacy_mode(args)
    initialization_control,initialization_aux_seed=validate_initialization(args)
    gate_mode=getattr(args,'ftm_gate_diagnostics','off')
    if gate_mode!='off':
        from .ftm_gate_diagnostics import validate_mode as validate_gate_mode
        validate_gate_mode(args)
    records=sorted(runner._records(args),key=lambda r:r['id']);records_by_id={r['id']:r for r in records}
    cpu_audit=ftm_cpu_audit.validate(args,records)
    spec=input_spec(args,records)
    labels=target_labels(args.labels_csv,records);pairs=pair_table(records,args.batch_size,args.seed,cpu_audit=cpu_audit)
    points=checkpoints(args.checkpoints,args.steps);output=runner._output(args.output);device=runner._device(args.device)
    cpu_metadata=ftm_cpu_audit.configure() if cpu_audit else None
    model,identity=load_model(args.source,device);identity=model_identity(identity,spec)
    size=spec['qualification_size']
    sync(device)
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
    prepare_start=time.perf_counter()
    with ForwardCounter(model) as preparation:
        preparation.phase='qualification'
        entries=qualification(model,device,size)
    sync(device);prepare_seconds=time.perf_counter()-prepare_start
    layer_sha=layer_digest(entries);names=[entry['name'] for entry in entries if entry['eligible']]
    if not names:raise ValueError('FTM qualified layer list is empty')
    profile=f'ftm-author-config-{args.ftm_layer_profile}-{spec["profile_component"]}-batch{args.batch_size}-update-{args.ftm_update_mode}'
    profile=beta_profile(profile,beta,args.ftm_update_mode)
    profile=state_profile(profile,state_control,args.ftm_update_mode)
    profile=initialization_profile(profile,initialization_control,args.ftm_update_mode)
    config=runner._config(args,records,kind='targeted-attack',source_identity=identity,labels_csv_sha256=sha256(args.labels_csv),
        target_labels=labels,profile=profile,checkpoint_steps=points,pairs=pairs,pair_table_sha256=layer_digest(pairs),
        update_mode=args.ftm_update_mode,trace_updates=args.trace_updates,
        state_control=state_control,state_control_spec=STATE_CONTROL_SPEC,
        ftm_efficacy_probe=probe_mode,
        initialization_control=initialization_control,initialization_aux_seed=initialization_aux_seed,
        ftm_gate_diagnostics=gate_mode,
        layer_qualification=entries,layer_table_sha256=layer_sha,qualified_layer_names=names,
        settings={'mix_prob':.1,'ftm_beta':beta,'divisor':4,'mixup_layer':'conv_linear_include_last',
            'mixed_image_type_feature':'C','shuffle_image_feature':'SelfShuffle','channelwise':True,
            'blending_mode_feature':'M','mix_lower_bound_feature':0.,'mix_upper_bound_feature':.75},
        random_stream='One dedicated CPU generator per fixed sorted pair; RDI and hooks share author draw order. Identity randperm is allowed.',
        adaptation=spec['adaptation'],
        cost_profile='Per-image row source F includes clean1+cache1+gradientT+two endpoint forwards/checkpoint. Qualification is separate per invocation, never duplicated in image costs. Row elapsed includes cache/clean/optimization/checkpoint IO, divided by actual batch size; excludes input loading and progress commit. attempts.jsonl also times commit. Peak CUDA is shared batch peak, not additive.')
    if probe_mode!='off':config['cost_profile']+=' Efficacy observe additionally includes four no-grad source image-forwards per step; reference adds zero forwards. Terminal-u digest and all observation work are included in wall/peak; no additional B or target F.'
    if cpu_audit:
        config['cpu_audit']=cpu_metadata
        config['adaptation']='RN50 IMAGENET1K_V1/original299 RGB PNG, no resize; exact299 qualification/actual input. One complete20-member SelfShuffle CPU batch, not10 pairs or independent images. Local software/strict flags and dedicated group RNG do not reproduce author global RNG/environment. Source-only audit.'
        config['random_stream']='One dedicated CPU generator for the single sorted20-member batch group; RDI and hooks retain author draw order. Identity randperm is allowed.'
    completed=runner._open_run(output,config);done={r['id'] for r in completed}
    for pair in pairs:
        overlap=done.intersection(pair['image_ids'])
        if overlap and overlap!=set(pair['image_ids']):raise ValueError('Partial committed FTM pair; refusing to change pair dynamics')
    pair_by_id={p['batch_id']:p for p in pairs}
    for row in completed:
        pair=pair_by_id.get(row.get('batch_id'))
        if not pair or row['id'] not in pair['image_ids'] or row['layer_table_sha256']!=layer_sha or row['target_label']!=labels[row['id']] or row['batch_rng_seed']!=pair['rng_seed'] or row['batch_image_ids']!=pair['image_ids'] or row['batch_size']!=pair['batch_size']:raise ValueError('Completed FTM pair/layer identity differs')
        runner._check_clean(records_by_id[row['id']]);verify_saved(row)
        batch_path=Path(row['shared_batch_total_path'])
        if sha256(batch_path)!=row['shared_batch_total_sha256']:raise ValueError('Saved batch metadata changed')
        trace=read_json(batch_path).get('update_trace')
        if trace and sha256(trace['path'])!=trace['sha256']:raise ValueError('Saved completed trace changed')
    append_jsonl(output/'preparation.jsonl',{'utc':utc(),'profile':args.ftm_layer_profile,'qualification_size':size,
        'layer_table_sha256':layer_sha,'forward_images':preparation.total,'forward_pixels':pixels(preparation.resolutions),
        'elapsed_seconds':prepare_seconds,'source_identity':identity,'input_gradient_images':0,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0})
    for pair in pairs:
        if set(pair['image_ids']).issubset(done):continue
        members=[records_by_id[i] for i in pair['image_ids']];n=len(members)
        for record in members:runner._check_clean(record)
        clean=torch.cat([image_tensor(r['canonical_path'],device) for r in members]);targets=torch.tensor([labels[r['id']] for r in members],device=device)
        generator=torch.Generator(device='cpu').manual_seed(pair['rng_seed']);saved={r['id']:{} for r in members}
        sync(device)
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        start=time.perf_counter();attempt={'batch_id':pair['batch_id'],'image_ids':pair['image_ids'],'started_utc':utc(),'status':'running'}
        count=ForwardCounter(model);actual_gradients=0;trace_writer=None;efficacy=None
        efficacy_metadata=None;terminal_digest=None;terminal_digest_seconds=None
        first_controller=None;first_metadata=None;end_step_rng=[];gate=None;gate_metadata=None
        try:
            with count,FeatureTuning(model,generator,input_size=spec['input_size'],
                    eligible_names=names if args.ftm_layer_profile=='mapped299' else None,beta=beta,trace_updates=args.trace_updates) as context:
                count.phase='clean'
                with context.disabled(),torch.no_grad():
                    clean_logits=model(clean)
                    source_clean=[target_metrics(clean_logits[i:i+1],int(targets[i]),r['label']) for i,r in enumerate(members)]
                count.phase='cache';context.cache(clean)
                if set(context.outputs)!=set(names):raise ValueError('Actual cached FTM layer list differs from qualification')
                if args.trace_updates:
                    trace_writer=UpdateTrace(output/'traces'/f'{pair["batch_id"]}.jsonl',pair['image_ids'],
                        {name:type(module).__name__ for name,module in context.layers},args.ftm_update_mode,beta=beta,state_control=state_control)
                if probe_mode=='observe':
                    efficacy=ConvEfficacyProbe(output/'efficacy'/f'{pair["batch_id"]}.jsonl',context,pair['image_ids'],
                        [labels[i] for i in pair['image_ids']],[r['label'] for r in members],args.seed,pair['batch_id'],args.steps,count,
                        {'config_sha256':sha256(output/'config.json'),'manifest_sha256':config['manifest_sha256'],
                            'labels_csv_sha256':config['labels_csv_sha256'],'source_identity':identity,
                            'layer_table_sha256':layer_sha,'pair_table_sha256':config['pair_table_sha256'],
                            'batch_rng_seed':pair['rng_seed'],'state_control':state_control,'update_mode':args.ftm_update_mode,
                            'numeric_settings':config['numeric_settings']})
                if gate_mode!='off':
                    from .ftm_gate_diagnostics import GateDiagnostics
                    gate=GateDiagnostics(output/'gates'/pair['batch_id'],gate_mode,context,pair['image_ids'],
                        {'config_sha256':sha256(output/'config.json'),'source_identity':identity,'numeric_settings':config['numeric_settings'],
                            'batch_rng_seed':pair['rng_seed'],'full_seed':args.seed,'layer_table_sha256':layer_sha,
                            'manifest_sha256':config['manifest_sha256'],'labels_csv_sha256':config['labels_csv_sha256'],
                            'update_mode':args.ftm_update_mode,'alpha_bytes':args.alpha_bytes,'epsilon_bytes':args.epsilon_bytes})
                def endpoint(step,image):
                    with context.disabled(),torch.no_grad():
                        count.phase='checkpoint_float';floating=model(image)
                        paths=[];encoded=[]
                        for i,record in enumerate(members):
                            path=output/'images'/f'{record["id"]}.png' if step==args.steps else output/'checkpoints'/str(step)/'images'/f'{record["id"]}.png'
                            encoded.append(save_adversarial(path,image[i:i+1],read_pixels(record['canonical_path']),args.epsilon_bytes));paths.append(path)
                        count.phase='checkpoint_png';png=model(torch.cat([image_tensor(p,device) for p in paths]))
                        for i,record in enumerate(members):
                            entry={'step':step,'png_path':str(paths[i]),'feature_loss':None,'png_feature_loss':None,
                                'float_source':target_metrics(floating[i:i+1],int(targets[i]),record['label']),
                                'png_source':target_metrics(png[i:i+1],int(targets[i]),record['label']),**encoded[i]}
                            if args.save_float:
                                path=paths[i].with_suffix('.npz');np.savez_compressed(path,image=image[i:i+1].detach().cpu().numpy())
                                entry.update(float_path=str(path),float_sha256=sha256(path))
                            saved[record['id']][str(step)]=entry
                def observed(step,_image,_context):
                    nonlocal actual_gradients
                    actual_gradients=step*n
                    if efficacy:efficacy.observe(step,_image,_context)
                    if first_controller is not None and args.trace_updates:
                        end_step_rng.append({'step':step,'main_generator_sha256':hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()})
                def completed_gradient(step):
                    nonlocal actual_gradients
                    actual_gradients=step*n
                transform=UnitStateControl(state_control) if state_control!='raw' else None
                if initialization_control!='off':
                    first_controller=FirstInitializationControl(initialization_control,initialization_aux_seed,pair['batch_id'],pair['image_ids'],context.layers)
                    transform=first_controller
                _image,stats=optimize_ftm(context,clean,targets,args.steps,args.epsilon_bytes,args.alpha_bytes,points,endpoint,count,observed,
                    update_mode=args.ftm_update_mode,update_observer=efficacy.capture if efficacy else trace_writer,state_transform=transform,after_gradient=completed_gradient,diagnostic_observer=gate)
                if gate:gate_metadata=gate.close('complete')
                expected=n*(args.steps+2+2*len(points)+(4*args.steps if efficacy else 0))
                if count.total!=expected or count.counts.get('optimization',0)!=args.steps*n:raise ValueError('FTM source actual/declaration count differs')
                if efficacy and any(count.counts.get('efficacy_'+branch,0)!=n*args.steps for branch in ('current','previous','zeroConv','randomConv')):raise ValueError('Conv efficacy actual source phase count differs')
                if any(p.grad is not None for p in model.parameters()):raise ValueError('Model parameter gradient unexpectedly accumulated')
                # Every saved endpoint is already complete. Count once before
                # context cleanup; transfer only one scalar per batch member.
                sync(device);audit_start=time.perf_counter()
                u_nonzero=final_u_nonzero(context)
                sync(device);audit_seconds=time.perf_counter()-audit_start
                if probe_mode!='off':
                    sync(device);digest_start=time.perf_counter()
                    terminal_digest=terminal_u_digest(context)
                    sync(device);terminal_digest_seconds=time.perf_counter()-digest_start
            sync(device)
            if efficacy:efficacy_metadata=efficacy.metadata()
            trace_metadata=trace_writer.commit() if trace_writer else None
            final_rng_sha=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
            if first_controller is not None:first_metadata=first_controller.metadata()
            wall=time.perf_counter()-start
            batch_stats={**pair,'forward_images':count.total,'forward_pixels':pixels(count.resolutions),'input_gradient_images':stats['input_gradient_images'],
                'input_gradient_pixels':stats['input_gradient_images']*clean.shape[-2]*clean.shape[-1],
                'forward_by_phase':dict(count.counts),'resolution_counts':dict(count.resolutions),'elapsed_seconds':wall,
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                'layer_table_sha256':layer_sha,'selected_layer_counts':stats['selected_layer_counts'],
                'final_u_nonzero_elements':dict(zip(pair['image_ids'],u_nonzero)),
                'final_u_state_audit_seconds':audit_seconds,
                'final_u_state_audit_scope':'Read-only final learned-u parameter-state check after all endpoints; across-layer element count, not a mechanism effect size. Included in batch wall; no model F/B or RNG draws.',
                'update_mode':args.ftm_update_mode,'state_control':state_control,'update_trace':trace_metadata,'final_generator_state_sha256':final_rng_sha}
            if probe_mode!='off':batch_stats.update(ftm_efficacy_probe=probe_mode,inner_efficacy=efficacy_metadata,
                terminal_u_digest=terminal_digest,terminal_u_digest_seconds=terminal_digest_seconds)
            if first_controller is not None:batch_stats.update(initialization_control=initialization_control,
                initialization_aux_seed=initialization_aux_seed,initialization_audit=first_metadata,
                initialization_end_step_main_rng=end_step_rng if args.trace_updates else None)
            if gate_metadata:batch_stats['gate_diagnostics']=gate_metadata
            batch_path=output/'batches'/f'{pair["batch_id"]}.json';write_json(batch_path,batch_stats)
            rows=[]
            for i,record in enumerate(members):
                final=saved[record['id']][str(args.steps)]
                rows.append({'id':record['id'],'split':record['split'],'label':record['label'],'target_label':labels[record['id']],
                    'canonical_sha256':record['canonical_sha256'],'image_seed':image_seed(args.seed,record['id']),
                    'batch_id':pair['batch_id'],'cluster_id':pair['cluster_id'],'batch_size':n,'batch_image_ids':pair['image_ids'],
                    'batch_rng_seed':pair['rng_seed'],'odd_singleton':pair['odd_singleton'],'shared_batch_total_path':str(batch_path),
                    'shared_batch_total_sha256':sha256(batch_path),'update_mode':args.ftm_update_mode,
                    'state_control':state_control,
                    'ftm_efficacy_probe':probe_mode,
                    'initialization_control':initialization_control,'initialization_aux_seed':initialization_aux_seed,
                    'ftm_gate_diagnostics':gate_mode,
                    'final_u_nonzero_elements':u_nonzero[i],
                    'final_u_state_audit_reference':str(batch_path),
                    'layer_table_sha256':layer_sha,'method':'ftm','profile':profile,'layer':None,'reference_rule':None,
                    'selected_reference_id':None,'reference_candidates':[],'d0':None,'dT':None,'relative_reduction':None,
                    'source_clean':source_clean[i],'source_float':final['float_source'],'source_png':final['png_source'],
                    'checkpoints':saved[record['id']],'png_path':final['png_path'],'png_sha256':final['png_sha256'],
                    'forward_images':count.total//n,'forward_pixels':batch_stats['forward_pixels']//n,
                    'input_gradient_images':args.steps,'input_gradient_pixels':args.steps*clean.shape[-2]*clean.shape[-1],
                    'actual_forward_hook':{'forward_images':count.total//n,'by_phase':{k:v//n for k,v in count.counts.items()},'batch_total_reference':str(batch_path)},
                    'reference_forward_images':0,'clean_forward_images':1,'cache_forward_images':1,'endpoint_forward_images':2*len(points),
                    'elapsed_seconds':wall/n,'elapsed_seconds_is_batch_share':True,'peak_cuda_allocated_bytes':batch_stats['peak_cuda_allocated_bytes'],
                    'peak_cuda_is_shared_batch_peak':True,'preparation_cost_reference':str(output/'preparation.jsonl')})
            atomic_progress(output/'progress.jsonl',[*completed,*rows]);completed.extend(rows);done.update(pair['image_ids'])
            attempt['status']='complete';attempt['committed']=True
            print(f'targeted-ftm {pair["batch_id"]} images={pair["image_ids"]}',flush=True)
        except BaseException as error:
            attempt['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed'
            attempt['error']=f'{type(error).__name__}:{error}';attempt['committed']=False
            if isinstance(error,StateTransformError):attempt['state_transform_failure']=error.details
            raise
        finally:
            if trace_writer:trace_writer.close()
            if efficacy:
                efficacy.close()
                attempt['inner_efficacy']=efficacy_metadata or efficacy.metadata()
            if first_controller is not None:
                attempt.update(initialization_control=initialization_control,initialization_aux_seed=initialization_aux_seed,
                    initialization_audit=first_metadata or first_controller.metadata(),
                    initialization_end_step_main_rng=end_step_rng if args.trace_updates else None)
            if gate:attempt['gate_diagnostics']=gate_metadata or gate.close('partial')
            sync(device)
            attempt.update(ended_utc=utc(),elapsed_seconds=time.perf_counter()-start,actual_forward_images=count.total,
                actual_forward_pixels=pixels(count.resolutions),completed_input_gradient_images=actual_gradients,
                input_gradient_cost_note='Counts autograd-returned input-gradient steps before state transform/trace. A failing in-flight backward remains unknown, not fabricated.')
            if trace_writer:attempt.update(trace_rows_observed=trace_writer.rows,observer_seconds=trace_writer.seconds,
                trace_path=str(trace_writer.temporary if trace_writer.temporary.exists() else trace_writer.path))
            append_jsonl(output/'attempts.jsonl',attempt)
    preparation_rows=read_jsonl(output/'preparation.jsonl');attempts=read_jsonl(output/'attempts.jsonl')
    summary={'kind':'targeted-attack','count':len(completed),'expected_images':len(records),'complete':len(completed)==len(records),
        'committed_image_forward_images':sum(r['forward_images'] for r in completed),
        'qualification_forward_images':sum(r['forward_images'] for r in preparation_rows),
        'qualification_forward_pixels':sum(r['forward_pixels'] for r in preparation_rows),
        'forward_images':sum(r['actual_forward_images'] for r in attempts)+sum(r['forward_images'] for r in preparation_rows),
        'forward_pixels':sum(r['actual_forward_pixels'] for r in attempts)+sum(r['forward_pixels'] for r in preparation_rows),
        'input_gradient_images':sum(r['input_gradient_images'] for r in completed),
        'input_gradient_pixels':sum(r['input_gradient_pixels'] for r in completed),
        'observed_all_attempt_input_gradient_images':sum(r['completed_input_gradient_images'] for r in attempts),
        'failed_attempts':sum(r['status']!='complete' for r in attempts),'target_forward_images':0,'profile':profile,
        'pair_table_sha256':config['pair_table_sha256'],'layer_table_sha256':layer_sha}
    if cpu_audit:summary['cpu_process_memory']=ftm_cpu_audit.process_peak_memory()
    write_json(output/'summary.json',summary);return summary
