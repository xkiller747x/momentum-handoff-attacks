"""Frozen two-new-seed OPS lifecycle replication, reusing existing runners."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import time

import torch

from baseline.ops import OPSContext
from baseline.ops_refresh import OPSRefreshContext
from .io import append_jsonl, read_json, read_jsonl, sha256, write_json
from .metrics import image_seed
from .models import ROOT
from .ops_profile import _source_archive, profile_metadata, rng_snapshot
from .pool_diagnostic import _hash_map, validate_completed_results

NEW_SEEDS=(20261028,20261029)
ALL_SEEDS=(20261027,*NEW_SEEDS)
FROZEN_KERNELS=('baseline/attacks.py','baseline/ops.py','baseline/ops_refresh.py','baseline/bsr.py','baseline/transforms.py',
                'utils/models.py','utils/data.py','utils/io.py','utils/metrics.py')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _time(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)


def seed_units(ids):
    if len(ids)!=276 or len(set(ids))!=276:
        raise ValueError('Seed replication requires the original276 unique global IDs')
    units=[]
    for seed in NEW_SEEDS:
        for block in range(1,5):
            fixed_first=(block%2==1) if seed==NEW_SEEDS[0] else (block%2==0)
            profiles=['fixed','refresh'] if fixed_first else ['refresh','fixed']
            units.append({'id':f's{seed}_b{block}','seed':seed,'block':block,'profiles':profiles,
                          'selected_ids':ids[(block-1)*69:block*69],
                          'source_forward_images':35880,'source_gradient_images':35880,'target_forward_images':552})
    return units


def seed_identities(ids):
    mapping={str(seed):{identity:image_seed(seed,identity) for identity in ids} for seed in ALL_SEEDS}
    values=[value for rows in mapping.values() for value in rows.values()]
    if len(values)!=828 or len(set(values))!=828:
        raise ValueError('Three seeds×276 global image seed derivations must be collision-free')
    return mapping


def seed_tasks(units):
    tasks=[]
    for unit in units:
        for profile in unit['profiles']:
            tasks.append({'id':f'{unit["id"]}_{profile}','unit':unit['id'],'order':len(tasks)+1,
                'seed':unit['seed'],'block':unit['block'],'profile':profile,
                'method':'ops' if profile=='fixed' else 'ops-refresh',
                'profile_name':'ops-author-31ops-93pool' if profile=='fixed' else 'ops-author-noiserefresh',
                'selected_ids':unit['selected_ids'],'source_forward_images':17940,'source_gradient_images':17940,'target_forward_images':276})
    return tasks


def cpu_pair_identity(seed,identity,shape=(1,3,224,224)):
    """CPU-only deterministic reconstruction, not a captured fixed GPU trajectory."""
    start=time.perf_counter();derived=image_seed(seed,identity);before=rng_snapshot()
    fixed=OPSContext(shape,8/255,5,5,2.,derived)
    refresh=OPSRefreshContext(shape,8/255,5,5,2.,derived)
    if any(not torch.equal(a,b) for a,b in zip(fixed.pool,refresh.pool)):
        raise ValueError('CPU pair first pools differ')
    per_step=[]
    for _step in range(10):
        a,b=fixed.step_plans(),refresh.step_plans()
        left=[{'pool_index':p.pool_index,'operators':[asdict(op) for op in p.operators]} for p in a]
        right=[{'pool_index':p.pool_index,'operators':[asdict(op) for op in p.operators]} for p in b]
        if left!=right or fixed.rng.getstate()!=refresh.rng.getstate():
            raise ValueError('CPU pair complete index/BasicPlan stream differs')
        encoded=json.dumps(left,sort_keys=True,separators=(',',':')).encode()
        per_step.append(hashlib.sha256(encoded).hexdigest())
    after=rng_snapshot()
    if before!=after:
        raise ValueError('CPU reconstruction advanced global Python/torch CPU RNG')
    lifecycle=refresh.lifecycle_record()
    return {'base_seed':seed,'id':identity,'image_seed':derived,'shape':list(shape),'dtype':'float32',
            'first_pool_sha256':lifecycle['noise_pool_sha256_by_pool'][0],
            'step_index_basicplan_sha256':per_step,
            'complete_10step_plan_sha256':hashlib.sha256(''.join(per_step).encode()).hexdigest(),
            'refresh_pool_sha256_by_pool':lifecycle['noise_pool_sha256_by_pool'],
            'pool_generation_count':lifecycle['noise_pool_generation_count'],
            'global_cpu_rng_unchanged':before==after,'cuda_rng_accessed':False,
            'reconstruction_seconds':time.perf_counter()-start,'extra_forward_images':0,'extra_gradient_images':0,
            'provenance':'CPU reconstruction of original contexts; not fixed GPU trajectory capture'}


def old_artifact_hashes(related,profile):
    result={}
    related_plan=read_json(related/'plan.json')
    for task in related_plan['tasks']:
        for group in ('attacks','evaluations'):
            for name in ('config.json','progress.jsonl'):
                path=related/group/task['id']/name
                result[str(path.resolve())]=sha256(path)
    for block in range(1,5):
        for group in ('attacks','evaluations'):
            for name in ('config.json','progress.jsonl'):
                path=profile/group/f'b{block}'/name
                result[str(path.resolve())]=sha256(path)
    return result


def replication_metadata(args):
    related,profile=Path(args.related_run).resolve(),Path(args.profile_run).resolve()
    protocol,freeze_path=Path(args.protocol).resolve(),Path(args.freeze).resolve()
    freeze=read_json(freeze_path)
    if sha256(protocol)!=freeze['protocol_sha256']:
        raise ValueError('Frozen seed replication protocol SHA differs')
    old_profile=read_json(profile/'plan.json')
    base,old_related,records,_clean=profile_metadata(related,old_profile['protocol'],old_profile['freeze'])
    if old_profile['related_plan_sha256']!=base['related_plan_sha256'] or old_profile['selected_ids']!=base['selected_ids']:
        raise ValueError('Old lifecycle and related plans differ')
    prior=_hash_map(old_profile['new_code_sha256']);kernels={}
    for relative in FROZEN_KERNELS:
        kernels[relative]=sha256(ROOT/relative)
        if kernels[relative]!=prior.get(relative):
            raise ValueError(f'Frozen attack kernel changed:{relative}')
    ids=base['selected_ids'];units=seed_units(ids)
    plan={'protocol':str(protocol),'protocol_sha256':sha256(protocol),'freeze':str(freeze_path),'freeze_sha256':sha256(freeze_path),
          'related_run':str(related),'profile_run':str(profile),'related_plan_sha256':sha256(related/'plan.json'),
          'profile_plan_sha256':sha256(profile/'plan.json'),'old_artifact_sha256':old_artifact_hashes(related,profile),
          'kernel_sha256':kernels,'old_runner_sha256':prior.get('utils/runner.py'),
          'manifest':base['manifest'],'manifest_sha256':base['manifest_sha256'],'clean_reference':base['clean_reference'],
          'source_identity':base['source_identity'],'target_identities':base['target_identities'],'populations':base['populations'],
          'selected_ids':ids,'blocks':base['blocks'],'units':units,'tasks':seed_tasks(units),'image_seed_mapping':seed_identities(ids),
          'new_seeds':list(NEW_SEEDS),'all_seeds':list(ALL_SEEDS),'targets':['dn121','vit_b_16'],
          'parameters':{'source':'rn50','steps':10,'views':1,'ne':5,'np':5,'beta':2.,'epsilon_bytes':8.,'alpha_mult':1.,'view_batch':1},
          'planned_source_forward_images':287040,'planned_source_gradient_images':287040,'planned_target_forward_images':4416,
          'bootstrap_seed':freeze['bootstrap_seed'],'last_gpu_start_utc':freeze['last_gpu_start_utc'],
          'last_new_unit_start_utc':freeze['last_new_unit_start_utc'],'last_estimated_unit_end_utc':freeze['last_estimated_unit_end_utc'],
          'window_deadline_utc':freeze['window_deadline_utc'],'unit_boundary_runtime_seconds':freeze['unit_boundary_runtime_seconds'],
          'identity_note':'CPU reconstruction0 extraF; old fixed historical rows do not gain an actual hook retrospectively'}
    return plan,base,old_related,records


def may_start_unit(plan,state,now,cumulative_seconds):
    first=state.get('first_complete_unit_wall_seconds')
    if cumulative_seconds>=7200:return False,'cumulative7200-second execute cap'
    if state.get('first_unit_projected_over_limit'):return False,'first complete unit×8 exceeds7200seconds'
    if now>_time(plan['last_new_unit_start_utc']):return False,'latest new-unit start reached'
    if first is None:
        return (now<=_time(plan['last_gpu_start_utc']),'latest first start reached')
    if now+timedelta(seconds=first)>_time(plan['last_estimated_unit_end_utc']):
        return False,'first-unit measured duration predicts end after04:00:11UTC'
    return True,None


def _check_unit_outputs(output,unit):
    metadata={identity:read_json(output/'cpu_identity'/f's{unit["seed"]}'/f'{identity}.json') for identity in unit['selected_ids']}
    for profile in ('fixed','refresh'):
        attack=read_jsonl(output/'attacks'/unit['id']/profile/'progress.jsonl')
        evaluations=read_jsonl(output/'evaluations'/unit['id']/profile/'progress.jsonl')
        if len(attack)!=69 or {r['id'] for r in attack}!=set(unit['selected_ids']):raise ValueError('Unit attack images incomplete')
        for target in ('dn121','vit_b_16'):
            selected=[r for r in evaluations if r['model']==target]
            if len(selected)!=69 or {r['id'] for r in selected}!=set(unit['selected_ids']):raise ValueError('Unit target evaluations incomplete')
        for row in attack:
            reference=metadata[row['id']]
            if row['image_seed']!=reference['image_seed'] or row['actual_forward_hook']['forward_images']!=260 or row['forward_images']!=260 or row['input_gradient_images']!=260:
                raise ValueError('New task seed/actual hook260 mismatch')
            if profile=='refresh' and row['noise_pool_sha256_by_pool']!=reference['refresh_pool_sha256_by_pool']:
                raise ValueError('Actual refresh pools differ from independent CPU reconstruction')


def ops_seed_matrix_command(args):
    from . import runner
    command_start=time.perf_counter();command_started_utc=_utc();output=runner._output(args.output)
    plan,base,old_related,records=replication_metadata(args)
    code=runner._code_identity()
    plan.update({'device':args.device,'new_code_sha256':code,'source_archive':_source_archive(output,code),
                 'new_runner_sha256':sha256(ROOT/'utils/runner.py'),
                 'runner_instrumentation':'Existing passive forward hook condition now includes both new fixed/refresh runs; no attack numerical path change'})
    plan_path=output/'plan.json'
    if plan_path.exists():
        if read_json(plan_path)!=plan:raise ValueError('Seed replication plan differs; use separate output')
    else:write_json(plan_path,plan)
    if not args.execute:return {'plan':str(plan_path),'units':8,'tasks':16,'source_F':287040,'source_B':287040,'target_F':4416,'executed':False}
    state_path=output/'state.json'
    state=read_json(state_path) if state_path.exists() else {'plan_sha256':sha256(plan_path),'units':{},'execute_attempts':[]}
    if state['plan_sha256']!=sha256(plan_path):raise ValueError('Replication state plan mismatch')
    now=datetime.now(timezone.utc)
    if not state['execute_attempts'] and now>_time(plan['last_gpu_start_utc']):raise ValueError('Latest first execute start passed')
    validate_completed_results(base,old_related,records)
    old_profile_state=read_json(Path(args.profile_run)/'state.json')
    if len(old_profile_state['blocks'])!=4 or any(b.get('status')!='complete' for b in old_profile_state['blocks'].values()):
        raise ValueError('Original seed27 refresh baseline must be all4 blocks complete')
    for identity in [plan['source_identity'],*plan['target_identities'].values()]:
        if sha256(identity['checkpoint'])!=identity['checkpoint_sha256']:raise ValueError('Frozen weight SHA mismatch')
    clean_root=Path(plan['clean_reference']['path'])
    clean={(r['model'],r['id']):r for r in read_jsonl(clean_root/'progress.jsonl')}
    attempt={'started_utc':command_started_utc,'status':'running'};state['execute_attempts'].append(attempt);write_json(state_path,state)
    try:
        for unit in plan['units']:
            if state['units'].get(unit['id'],{}).get('status')=='complete':continue
            elapsed=sum(a.get('wall_seconds',0) for a in state['execute_attempts'])+time.perf_counter()-command_start
            allowed,reason=may_start_unit(plan,state,datetime.now(timezone.utc),elapsed)
            if not allowed:attempt['stop_reason']=reason;break
            entry=state['units'].setdefault(unit['id'],{'attempts':[],'tasks':{}})
            group={'started_utc':_utc(),'status':'running'};entry['attempts'].append(group);entry['status']='running';write_json(state_path,state)
            unit_start=time.perf_counter()
            try:
                for identity in unit['selected_ids']:
                    path=output/'cpu_identity'/f's{unit["seed"]}'/f'{identity}.json'
                    if path.exists():
                        value=read_json(path)
                        if value['image_seed']!=plan['image_seed_mapping'][str(unit['seed'])][identity] or value['base_seed']!=unit['seed'] or value['id']!=identity:
                            raise ValueError('CPU reconstruction identity differs')
                    else:
                        value=cpu_pair_identity(unit['seed'],identity);write_json(path,value)
                for profile in unit['profiles']:
                    attack_output=output/'attacks'/unit['id']/profile;eval_output=output/'evaluations'/unit['id']/profile
                    selection=dict(manifest=plan['manifest'],split='dev',allow_holdout=False,limit=None,ids=','.join(unit['selected_ids']),device=args.device)
                    attack_args=argparse.Namespace(command='attack',**selection,output=str(attack_output),method='ops' if profile=='fixed' else 'ops-refresh',
                        source='rn50',steps=10,views=1,epsilon_bytes=8.,alpha_mult=1.,seed=unit['seed'],view_batch=1,di_probability=.5,resize_rate=1.1,
                        save_float=False,ne=5,np=5,beta=2.,profile='ops-author-31ops-93pool' if profile=='fixed' else 'ops-author-noiserefresh')
                    eval_args=argparse.Namespace(command='evaluate',**selection,output=str(eval_output),attack_run=str(attack_output),targets='dn121,vit_b_16',
                        population='target-clean',source_clean=plan['clean_reference']['path'])
                    task=entry['tasks'].setdefault(profile,{'attempts':[]})
                    task_attempt={'started_utc':_utc(),'status':'running'};task['attempts'].append(task_attempt);task['status']='running';write_json(state_path,state)
                    task_start=time.perf_counter()
                    try:
                        runner.attack_command(attack_args);runner.evaluate_command(eval_args)
                        evaluated=read_jsonl(eval_output/'progress.jsonl')
                        if any(r['clean_prediction']!=clean[(r['model'],r['id'])]['prediction'] for r in evaluated):
                            raise ValueError('New clean argmax differs from frozen population reference')
                        task['status']=task_attempt['status']='complete'
                    except BaseException as error:
                        task['status']=task_attempt['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed'
                        task_attempt['error']=f'{type(error).__name__}:{error}';raise
                    finally:
                        task_attempt['ended_utc']=_utc();task_attempt['wall_seconds']=time.perf_counter()-task_start
                        task['total_attempt_wall_seconds']=sum(a.get('wall_seconds',0) for a in task['attempts']);write_json(state_path,state)
                _check_unit_outputs(output,unit)
                entry['status']=group['status']='complete'
            except BaseException as error:
                entry['status']=group['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed'
                group['error']=f'{type(error).__name__}:{error}';raise
            finally:
                group['ended_utc']=_utc();group['wall_seconds']=time.perf_counter()-unit_start
                entry['total_attempt_wall_seconds']=sum(a.get('wall_seconds',0) for a in entry['attempts'])
                if entry['status']=='complete' and 'first_complete_unit_wall_seconds' not in state:
                    state['first_complete_unit_wall_seconds']=entry['total_attempt_wall_seconds']
                    state['first_unit_projected_seconds']=8*entry['total_attempt_wall_seconds']
                    state['first_unit_projected_over_limit']=state['first_unit_projected_seconds']>7200
                write_json(state_path,state);append_jsonl(output/'events.jsonl',{'unit':unit['id'],**group})
        attempt['status']='complete' if sum(u.get('status')=='complete' for u in state['units'].values())==8 else 'partial'
    except BaseException as error:
        attempt['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed';attempt['error']=f'{type(error).__name__}:{error}';raise
    finally:
        attempt['ended_utc']=_utc();attempt['wall_seconds']=time.perf_counter()-command_start;write_json(state_path,state)
    return {'status':attempt['status'],'output':str(output),'complete_units':sum(u.get('status')=='complete' for u in state['units'].values()),'planned_units':8}
