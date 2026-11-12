"""Engineering checks and frozen four-block OPS lifecycle comparison."""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import random
import time
import zipfile

import torch

from baseline.attacks import mean_view_gradient
from baseline.ops import OPSContext
from baseline.ops_refresh import OPSRefreshContext
from .data import select_records
from .io import append_jsonl, image_tensor, read_json, read_jsonl, sha256, write_json
from .metrics import image_seed
from .models import ROOT, load_model
from .pool_diagnostic import REQUIRED_KERNELS, _hash_map, validate_completed_results


def _utc():
    return datetime.now(timezone.utc).isoformat()


def rng_snapshot(device=None):
    """CUDA is read only when a real explicitly selected CUDA device is supplied."""
    result = {'python': hashlib.sha256(repr(random.getstate()).encode()).hexdigest(),
              'torch_cpu': hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
              'torch_cuda': None}
    if device is not None and torch.device(device).type == 'cuda':
        result['torch_cuda'] = hashlib.sha256(torch.cuda.get_rng_state(device).cpu().numpy().tobytes()).hexdigest()
    return result


def gradient_comparison(first, second):
    first, second = first.detach().cpu().double().flatten(), second.detach().cpu().double().flatten()
    norm_first, norm_second = first.norm().item(), second.norm().item()
    difference = second - first
    return {'max_abs_error': difference.abs().max().item(),
            'relative_l2_error': difference.norm().item() / norm_first if norm_first else None,
            'cosine': float(torch.dot(first, second).item() / (norm_first * norm_second)) if norm_first and norm_second else None,
            'sign_agreement': (first.sign() == second.sign()).double().mean().item(),
            'both_zero': norm_first == 0 and norm_second == 0}


def gradient_check_refresh(args):
    from . import runner
    if args.split != 'dev' or args.limit != 2 or args.ids or args.source != 'rn50':
        raise ValueError('ops-refresh engineering gradient check is restricted to dev first2 and rn50')
    if (args.ne, args.np, args.beta, args.epsilon_bytes, args.seed) != (5, 5, 2., 8., 20261027):
        raise ValueError('ops-refresh gradient check requires Ne=Np5,beta2,EPS8,seed20261027')
    if args.profile not in (None, 'ops-author-noiserefresh'):
        raise ValueError('ops-refresh profile mismatch')
    records, output, device = runner._records(args), runner._output(args.output), runner._device(args.device)
    model, identity = load_model('rn50', device)
    config = runner._config(args, records, source_identity=identity, comparisons=['refresh_mb1','refresh_mb2','fixed_mb1'],
                            planned_source_forward_images=156, planned_input_gradient_images=156,
                            profile='ops-author-noiserefresh', purpose='first-step engineering, no target ASR')
    rows = runner._open_run(output, config)
    done = {row['id'] for row in rows}
    for record in records:
        if record['id'] in done:
            continue
        runner._check_clean(record)
        clean = image_tensor(record['canonical_path'], device)
        labels = torch.tensor([record['label']], device=device)
        seed = image_seed(args.seed, record['id'])
        before = rng_snapshot(device)
        refreshed = OPSRefreshContext(clean.shape, 8/255, 5, 5, 2., seed)
        fresh_plans = refreshed.step_plans()
        after_refresh = rng_snapshot(device)
        fixed = OPSContext(clean.shape, 8/255, 5, 5, 2., seed)
        fixed_plans = fixed.step_plans()
        after_fixed = rng_snapshot(device)
        gradients, counters = [], []
        for name, plans, view_batch in [('refresh_mb1', fresh_plans, 1), ('refresh_mb2', fresh_plans, 2), ('fixed_mb1', fixed_plans, 1)]:
            counter = {'name': name, 'view_batch': view_batch, 'forward_images': 0, 'input_gradient_images': 0,
                       'forward_pixels': 0, 'input_gradient_pixels': 0, 'resolution_counts': {}}
            actual = {'forward_images': 0}
            def hook(_module, inputs):
                actual['forward_images'] += inputs[0].shape[0]
            handle = model.register_forward_pre_hook(hook)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            try:
                gradient = mean_view_gradient(model, clean, labels, plans, view_batch, counter)
            finally:
                handle.remove()
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            if actual['forward_images'] != 26 or counter['input_gradient_images'] != 26:
                raise RuntimeError('First-step engineering actual26-view count mismatch')
            counter.update({'actual_forward_hook': actual, 'elapsed_seconds': time.perf_counter()-start,
                            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0})
            gradients.append(gradient.detach().cpu())
            counters.append(counter)
        row = {'id': record['id'], 'split': 'dev', 'label': record['label'], 'image_seed': seed,
               'refresh_microbatch_comparison': gradient_comparison(gradients[0], gradients[1]),
               'fixed_vs_refresh_first_step': gradient_comparison(gradients[2], gradients[0]),
               'first_pool_max_abs_difference': max((a-b).abs().max().item() for a,b in zip(refreshed.pool,fixed.pool)),
               'paired_indices_and_complete_operators': all(a.pool_index==b.pool_index and a.operators==b.operators for a,b in zip(fresh_plans,fixed_plans)),
               'global_rng_snapshots': {'before': before, 'after_refresh_pool_plan': after_refresh, 'after_fixed_pool_plan': after_fixed},
               'refresh_pool_plan_global_rng_unchanged': before==after_refresh,
               'fixed_pool_plan_global_rng_unchanged': after_refresh==after_fixed,
               'checks': counters, 'forward_images': sum(c['forward_images'] for c in counters),
               'input_gradient_images': sum(c['input_gradient_images'] for c in counters), 'target_forward_images': 0,
               'lifecycle': refreshed.lifecycle_record()}
        append_jsonl(output/'progress.jsonl',row)
        rows.append(row)
        print(f'ops-refresh gradient-check {record["id"]}:78 sourceF/B,0 targetF',flush=True)
    summary={'count':len(rows),'forward_images':sum(r['forward_images'] for r in rows),
             'input_gradient_images':sum(r['input_gradient_images'] for r in rows),'target_forward_images':0,'checks':rows}
    write_json(output/'summary.json',summary)
    return summary


def profile_metadata(related_run, protocol, freeze_path):
    related, protocol, freeze_path = Path(related_run).resolve(),Path(protocol).resolve(),Path(freeze_path).resolve()
    old, freeze = read_json(related/'plan.json'),read_json(freeze_path)
    if sha256(protocol)!=freeze['protocol_sha256'] or sha256(related/'plan.json')!=freeze['related_plan_sha256']:
        raise ValueError('Frozen lifecycle protocol/related-plan SHA mismatch')
    kernels=_hash_map(freeze['fixed_kernel_sha256'])
    old_hash=_hash_map(old['code_sha256'])
    for relative in REQUIRED_KERNELS:
        if sha256(ROOT/relative)!=kernels[relative] or old_hash.get(relative)!=kernels[relative]:
            raise ValueError(f'Original fixed kernel changed:{relative}')
    manifest=Path(old['manifest'])
    if sha256(manifest)!=old['manifest_sha256']:
        raise ValueError('Manifest SHA mismatch')
    records=select_records(manifest,'dev')
    ids=[r['id'] for r in records]
    if len(ids)!=276 or ids!=old['selected_ids'] or old['blocks']!=[ids[i:i+69] for i in range(0,276,69)]:
        raise ValueError('Frozen dev276/block ordering mismatch')
    reference=old['clean_reference']; clean_path=Path(reference['path'])
    if sha256(clean_path/'config.json')!=reference['config_sha256'] or sha256(clean_path/'progress.jsonl')!=reference['progress_sha256']:
        raise ValueError('Frozen clean reference changed')
    clean_config=read_json(clean_path/'config.json')
    clean={(r['model'],r['id']):r for r in read_jsonl(clean_path/'progress.jsonl')}
    if any((model,identity) not in clean for model in ('rn50','dn121','vit_b_16') for identity in ids):
        raise ValueError('Clean reference must contain all three models on dev276')
    populations={}
    for target,expected,common_expected in [('dn121',248,241),('vit_b_16',262,255)]:
        own=[identity for identity in ids if clean[(target,identity)]['correct']]
        common=[identity for identity in own if clean[('rn50',identity)]['correct']]
        if len(own)!=expected or len(common)!=common_expected:
            raise ValueError('Frozen target-own/source-target clean population differs')
        populations[target]={'target_clean_ids':own,'source_target_clean_ids':common}
    plan={'protocol':str(protocol),'protocol_sha256':sha256(protocol),'freeze':str(freeze_path),'freeze_sha256':sha256(freeze_path),
          'related_run':str(related),'related_plan_sha256':sha256(related/'plan.json'),'old_code_sha256':old['code_sha256'],
          'kernel_sha256':kernels,'manifest':str(manifest),'manifest_sha256':sha256(manifest),'selected_ids':ids,'blocks':old['blocks'],
          'clean_reference':reference,'source_identity':clean_config['model_identities']['rn50'],
          'target_identities':{target:clean_config['model_identities'][target] for target in ('dn121','vit_b_16')},
          'populations':populations,'targets':['dn121','vit_b_16'],'seed':20261027,
          'method':'ops-refresh','profile':'ops-author-noiserefresh','steps':10,'views':1,'ne':5,'np':5,'beta':2.,
          'epsilon_bytes':8.,'alpha_mult':1.,'view_batch':1,'planned_forward_images':71760,'planned_input_gradient_images':71760,
          'planned_target_forward_images':1104,'bootstrap_seed':freeze['bootstrap_seed'],
          'last_gpu_start_utc':freeze['last_gpu_start_utc'],'window_deadline_utc':freeze['window_deadline_utc'],
          'block_boundary_runtime_seconds':freeze['block_boundary_runtime_seconds'],
          'interpretation':'Both fixed and refresh profiles retained; exploratory lifecycle sensitivity, not a new method'}
    return plan,old,records,clean


def _source_archive(output,code_identity):
    path=output/'source_snapshot.zip'
    files={key.replace('\\','/'):ROOT/key for key in code_identity}
    files.update({name:ROOT/name for name in ('main.py','baseline.py')})
    files.update({str(path.relative_to(ROOT)).replace('\\','/'):path for path in (ROOT/'test').glob('*.py')})
    if not path.exists():
        with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
            for relative,source in sorted(files.items()):
                archive.writestr(relative,source.read_bytes())
    with zipfile.ZipFile(path) as archive:
        for relative,source in files.items():
            if hashlib.sha256(archive.read(relative)).hexdigest()!=sha256(source):
                raise ValueError('Lifecycle source archive differs from current source snapshot')
    return {'path':path.name,'sha256':sha256(path)}


def ops_profile_matrix_command(args):
    from . import runner
    command_start=time.perf_counter()
    output=runner._output(args.output)
    plan,old,records,_clean=profile_metadata(args.related_run,args.protocol,args.freeze)
    code_identity=runner._code_identity()
    plan.update({'new_code_sha256':code_identity,'device':args.device,'source_archive':_source_archive(output,code_identity)})
    plan_path=output/'plan.json'
    if plan_path.exists():
        if read_json(plan_path)!=plan:
            raise ValueError('Lifecycle plan differs; use a separate output directory')
    else:
        write_json(plan_path,plan)
    if not args.execute:
        return {'plan':str(plan_path),'blocks':4,'forward_images':71760,'input_gradient_images':71760,'target_forward_images':1104,'executed':False}
    if datetime.now(timezone.utc)>datetime.fromisoformat(plan['last_gpu_start_utc'].replace('Z','+00:00')):
        raise ValueError('Latest permitted lifecycle baseline GPU start has passed')
    original=validate_completed_results(plan,old,records)
    for record in records:
        if sha256(record['canonical_path'])!=record['canonical_sha256'] or sha256(original[record['id']]['png'])!=original[record['id']]['png_sha256']:
            raise ValueError('Existing fixed PNG/canonical content changed')
    if sha256(plan['source_identity']['checkpoint'])!=plan['source_identity']['checkpoint_sha256']:
        raise ValueError('Source checkpoint SHA mismatch')
    for identity in plan['target_identities'].values():
        if sha256(identity['checkpoint'])!=identity['checkpoint_sha256']:
            raise ValueError('Target checkpoint SHA mismatch')
    state_path=output/'state.json'
    state=read_json(state_path) if state_path.exists() else {'plan_sha256':sha256(plan_path),'blocks':{},'attempts':[]}
    if state['plan_sha256']!=sha256(plan_path):
        raise ValueError('Lifecycle state belongs to another plan')
    attempt={'started_utc':_utc(),'status':'running'}; state['attempts'].append(attempt)
    write_json(state_path,state)
    try:
        for block,ids in enumerate(plan['blocks'],1):
            if state['blocks'].get(str(block),{}).get('status')=='complete':
                continue
            elapsed=sum(a.get('wall_seconds',0) for a in state['attempts'])+time.perf_counter()-command_start
            if elapsed>=2400 or state.get('first_block_projected_over_limit') or datetime.now(timezone.utc)>=datetime.fromisoformat(plan['window_deadline_utc'].replace('Z','+00:00')):
                attempt['stop_reason']='frozen block-boundary time/projection limit'; break
            entry=state['blocks'].setdefault(str(block),{'attempts':[]})
            group={'started_utc':_utc(),'status':'running'};entry['attempts'].append(group);entry['status']='running'
            write_json(state_path,state)
            group_start=time.perf_counter()
            selection=dict(manifest=plan['manifest'],split='dev',allow_holdout=False,limit=None,ids=','.join(ids),device=args.device)
            attack_output=output/'attacks'/f'b{block}'
            evaluate_output=output/'evaluations'/f'b{block}'
            attack_args=argparse.Namespace(command='attack',**selection,output=str(attack_output),method='ops-refresh',source='rn50',
                steps=10,views=1,epsilon_bytes=8.,alpha_mult=1.,seed=20261027,view_batch=1,di_probability=.5,resize_rate=1.1,
                save_float=False,ne=5,np=5,beta=2.,profile='ops-author-noiserefresh')
            eval_args=argparse.Namespace(command='evaluate',**selection,output=str(evaluate_output),attack_run=str(attack_output),
                targets='dn121,vit_b_16',population='target-clean',source_clean=plan['clean_reference']['path'])
            try:
                runner.attack_command(attack_args)
                runner.evaluate_command(eval_args)
                entry['status']=group['status']='complete'
            except BaseException as error:
                entry['status']=group['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed'
                group['error']=f'{type(error).__name__}:{error}';raise
            finally:
                group['ended_utc']=_utc();group['wall_seconds']=time.perf_counter()-group_start
                entry['total_attempt_wall_seconds']=sum(g.get('wall_seconds',0) for g in entry['attempts'])
                if block==1 and entry['status']=='complete':
                    state['first_block_projected_seconds']=entry['total_attempt_wall_seconds']*4
                    state['first_block_projected_over_limit']=state['first_block_projected_seconds']>2400
                write_json(state_path,state)
                append_jsonl(output/'events.jsonl',{'block':block,**group})
        attempt['status']='complete' if len(state['blocks'])==4 and all(b.get('status')=='complete' for b in state['blocks'].values()) else 'partial'
    except BaseException as error:
        attempt['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed'
        attempt['error']=f'{type(error).__name__}:{error}';raise
    finally:
        attempt['ended_utc']=_utc();attempt['wall_seconds']=time.perf_counter()-command_start
        write_json(state_path,state)
    return {'output':str(output),'status':attempt['status'],'completed_blocks':sum(b.get('status')=='complete' for b in state['blocks'].values()),'planned_blocks':4}
