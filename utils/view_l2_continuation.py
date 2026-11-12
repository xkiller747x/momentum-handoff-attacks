"""Frozen three-seed continuation; reuses existing attack/evaluation numerics."""
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
import time

from baseline.norm_control import PROFILE
from .io import append_jsonl, read_json, read_jsonl, sha256, write_json
from .metrics import image_seed, paired_bootstrap
from .models import ROOT
from .ops_profile import _source_archive
from .ops_seed_report import image_seed_mean, exact_mcnemar
from .pool_diagnostic import _hash_map

SEEDS=(20261027,20261028,20261029)

def parsed(value):
    return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)

def utc():
    return datetime.now(timezone.utc).isoformat()

def verify_hashes(mapping):
    for path,digest in mapping.items():
        if sha256(path)!=digest:raise ValueError(f'Frozen artifact SHA mismatch: {path}')

def tasks(ids,profile_run,seed_run):
    if len(ids)!=276 or len(set(ids))!=276:raise ValueError('Expected original276 unique IDs')
    result=[]
    for seed in SEEDS:
        for block in range(1,5):
            name=f's{seed}_b{block}'
            old=Path(profile_run) if seed==SEEDS[0] else Path(seed_run)
            relative=f'b{block}' if seed==SEEDS[0] else f'{name}/refresh'
            result.append({'id':name,'seed':seed,'block':block,'selected_ids':ids[(block-1)*69:block*69],
                'old_attack':str(old/'attacks'/relative),'old_evaluation':str(old/'evaluations'/relative),
                'source_forward_images':17940,'source_gradient_images':17940,'target_forward_images':276})
    return result

def metadata(args):
    protocol=Path(args.protocol).resolve();freeze_path=Path(args.freeze).resolve();freeze=read_json(freeze_path)
    workspace=protocol.parent.parent
    resolve=lambda value:Path(value) if Path(value).is_absolute() else workspace/value
    pins={str(protocol):freeze['protocol_sha256'],str(freeze_path):sha256(freeze_path)}
    for key in ('compatibility','timing'):
        pins[str(resolve(freeze[key]))]=freeze[key+'_sha256']
    verify_hashes(pins)
    compat=read_json(resolve(freeze['compatibility']))
    for key in ('historical_protocol','measurement_freeze','old_seed_plan','old_seed_integrity','old_raw_snapshot'):
        pins[str(resolve(compat[key]))]=compat[key+'_sha256']
    for proof in compat['evidence'].values():
        for name in ('config','summary','progress'):
            filename='progress.jsonl' if name=='progress' else name+'.json'
            pins[str(Path(proof['path'])/filename)]=proof[name+'_sha256']
    verify_hashes(pins)
    kernels={str(ROOT/key):digest for key,digest in compat['numeric_sha256'].items()}
    verify_hashes(kernels)
    seed_root=Path(args.seed_run).resolve();profile_root=Path(args.profile_run).resolve()
    if sha256(seed_root/'plan.json')!=compat['old_seed_plan_sha256']:raise ValueError('Old seed plan differs')
    seed=read_json(seed_root/'plan.json');old=read_json(profile_root/'plan.json')
    if sha256(profile_root/'plan.json')!=seed['profile_plan_sha256']:raise ValueError('Old profile plan differs')
    snapshot=read_json(resolve(compat['old_raw_snapshot']))
    if snapshot['mismatches'] or len(snapshot['artifact_sha256'])!=876:raise ValueError('Old raw snapshot incomplete')
    verify_hashes(snapshot['artifact_sha256'])
    verify_hashes(seed['old_artifact_sha256'])
    if old['selected_ids']!=seed['selected_ids'] or old['manifest_sha256']!=seed['manifest_sha256']:raise ValueError('Old population identities differ')
    pins[str(Path(seed['manifest']))]=seed['manifest_sha256']
    clean=seed['clean_reference']
    for name in ('config','progress'):
        pins[str(Path(clean['path'])/('config.json' if name=='config' else 'progress.jsonl'))]=clean[name+'_sha256']
    weights={identity['checkpoint']:identity['checkpoint_sha256'] for identity in [seed['source_identity'],*seed['target_identities'].values()]}
    verify_hashes(weights)
    verify_hashes(pins)
    result={key:seed[key] for key in ('manifest','manifest_sha256','selected_ids','blocks','source_identity','target_identities','clean_reference','populations')}
    result.update({'protocol':str(protocol),'protocol_sha256':sha256(protocol),'freeze':freeze,'freeze_path':str(freeze_path),
        'freeze_sha256':sha256(freeze_path),'compatibility_sha256':freeze['compatibility_sha256'],'timing_sha256':freeze['timing_sha256'],
        'pins':pins,'weight_sha256':weights,'numeric_sha256':compat['numeric_sha256'],'old_artifact_sha256':snapshot['artifact_sha256'],
        'profile_run':str(profile_root),'seed_run':str(seed_root),'old_profile_code':old['new_code_sha256'],
        'old_seed_code':seed['new_code_sha256'],'tasks':tasks(seed['selected_ids'],profile_root,seed_root),
        'method':'ops-view-l2','profile':PROFILE,'source_forward_images':215280,'source_gradient_images':215280,
        'target_forward_images':3312,'reused_evidence':'Five frozen proofs and prior three-seed CPU identities; no repeated model engineering.'})
    for task in result['tasks']:validate_task(result,task,Path(task['old_attack']),Path(task['old_evaluation']),old=True,require_complete=True)
    return result

def validate_task(plan,task,attack_root,eval_root,old=False,require_complete=False):
    ids=task['selected_ids'];a=read_jsonl(attack_root/'progress.jsonl');e=read_jsonl(eval_root/'progress.jsonl')
    components=[eval_root/target for target in plan['target_identities'] if (eval_root/target/'config.json').exists()]
    if components:e=[r for root in components for r in read_jsonl(root/'progress.jsonl')]
    if len({r['id'] for r in a})!=len(a) or any(r['id'] not in ids for r in a):raise ValueError('Invalid attack IDs')
    if len({(r['model'],r['id']) for r in e})!=len(e) or any(r['id'] not in ids or r['model'] not in plan['target_identities'] for r in e):raise ValueError('Invalid evaluation IDs')
    code=plan['old_profile_code'] if old and task['seed']==SEEDS[0] else plan['old_seed_code'] if old else plan['new_code_sha256']
    for root,is_attack in [(attack_root,True),*[(root,False) for root in (components or [eval_root])]]:
        if not (root/'config.json').exists():continue
        cfg=read_json(root/'config.json')
        if cfg['selected_ids']!=ids or cfg['manifest_sha256']!=plan['manifest_sha256'] or _hash_map(cfg['code_sha256'])!=_hash_map(code):raise ValueError('Task config identity differs')
        if is_attack:
            wanted={'method':'ops-refresh' if old else 'ops-view-l2','profile':'ops-author-noiserefresh' if old else PROFILE,
                'seed':task['seed'],'steps':10,'views':1,'view_batch':1,'ne':5,'np':5,'beta':2.,'epsilon_bytes':8.,'alpha_mult':1.,'source':'rn50'}
            if cfg['source_identity']!=plan['source_identity'] or any(cfg['arguments'].get(k)!=v for k,v in wanted.items()):raise ValueError('Attack parameters differ')
        elif not set(cfg['target_identities']).issubset(plan['target_identities']) or cfg['target_identities']!={key:plan['target_identities'][key] for key in cfg['target_identities']} or (components and (set(cfg['target_identities'])!={root.name} or cfg['arguments']['targets']!=root.name)):raise ValueError('Target identities differ')
    old_rows={r['id']:r for r in read_jsonl(Path(task['old_attack'])/'progress.jsonl')}
    clean={(r['model'],r['id']):r for r in read_jsonl(Path(plan['clean_reference']['path'])/'progress.jsonl')}
    records={r['id']:r for r in read_json(plan['manifest'])['records']}
    for row in a:
        record=records[row['id']]
        if row['image_seed']!=image_seed(task['seed'],row['id']) or row['label']!=record['label'] or row['canonical_sha256']!=record['canonical_sha256']:raise ValueError('Image identity differs')
        if sha256(attack_root/'images'/f'{row["id"]}.png')!=row['png_sha256'] or sha256(record['canonical_path'])!=record['canonical_sha256']:raise ValueError('PNG/canonical SHA differs')
        if row['forward_images']!=260 or row['input_gradient_images']!=260 or row['noise_pool_generation_count']!=10 or len(row['noise_pool_sha256_by_pool'])!=10:raise ValueError('Image cost/pool count differs')
        if row['png_max_byte_difference']>8 or row['png_roundtrip_max_byte_error']!=0:raise ValueError('PNG bound/roundtrip differs')
        if not old and (row['actual_forward_hook']['forward_images']!=260 or len(row['steps_norm_summary'])!=10 or row['noise_pool_sha256_by_pool']!=old_rows[row['id']]['noise_pool_sha256_by_pool']):raise ValueError('Actual hook/step/pool identity differs')
    for row in e:
        if row['clean_prediction']!=clean[(row['model'],row['id'])]['prediction'] or row['label']!=records[row['id']]['label']:raise ValueError('Frozen clean argmax/label differs')
    complete=set(r['id'] for r in a)==set(ids) and all({r['id'] for r in e if r['model']==target}==set(ids) for target in plan['target_identities'])
    if require_complete and not complete:raise ValueError('Task incomplete')
    return a,e,complete

def may_start(plan,state,now,elapsed):
    freeze=plan['freeze']
    if elapsed>=freeze['boundary_cap_seconds']:return False,'cumulative boundary cap'
    if state.get('first_block_projected_over_limit'):return False,'first block×12 abnormal-cost sentinel'
    if now>parsed(freeze['last_new_block_utc']):return False,'last new block UTC'
    longest=max((b.get('total_attempt_wall_seconds',0) for b in state.get('blocks',{}).values() if b.get('status')=='complete'),default=0)
    if now+timedelta(seconds=longest)>parsed(freeze['last_estimated_block_end_utc']):return False,'longest completed block projects beyond deadline'
    if not state.get('attempts') and now+timedelta(seconds=freeze['initial_projection_seconds'])>parsed(freeze['last_estimated_block_end_utc']):return False,'initial full-cost projection exceeds deadline'
    return True,None

def matrix_command(args):
    from . import runner
    start=time.perf_counter();plan=metadata(args);output=runner._output(args.output)
    plan['new_code_sha256']=runner._code_identity();plan['device']=args.device
    plan['source_archive']=_source_archive(output,plan['new_code_sha256'])
    path=output/'plan.json'
    if path.exists() and read_json(path)!=plan:raise ValueError('Continuation plan differs; use identical config to resume')
    write_json(path,plan)
    if not args.execute:return {'output':str(output),'tasks':12,'source_forward_images':215280,'source_gradient_images':215280,'target_forward_images':3312,'executed':False}
    state_path=output/'state.json';state=read_json(state_path) if state_path.exists() else {'plan_sha256':sha256(path),'blocks':{},'attempts':[]}
    if state['plan_sha256']!=sha256(path):raise ValueError('State/plan SHA differs')
    verify_hashes(plan['pins']);verify_hashes(plan['weight_sha256']);allowed,reason=may_start(plan,state,datetime.now(timezone.utc),sum(a.get('wall_seconds',0) for a in state['attempts']))
    if not allowed:raise ValueError(reason)
    attempt={'started_utc':utc(),'status':'running'};state['attempts'].append(attempt);write_json(state_path,state)
    try:
        for task in plan['tasks']:
            if state['blocks'].get(task['id'],{}).get('status')=='complete':continue
            elapsed=sum(a.get('wall_seconds',0) for a in state['attempts'])+time.perf_counter()-start
            allowed,reason=may_start(plan,state,datetime.now(timezone.utc),elapsed)
            if not allowed:attempt['stop_reason']=reason;break
            entry=state['blocks'].setdefault(task['id'],{'attempts':[]});group={'started_utc':utc(),'status':'running'};entry['attempts'].append(group);entry['status']='running';write_json(state_path,state);block_start=time.perf_counter()
            try:
                selection=dict(manifest=plan['manifest'],split='dev',allow_holdout=False,limit=None,ids=','.join(task['selected_ids']),device=args.device)
                attack=output/'attacks'/task['id'];evaluation=output/'evaluations'/task['id']
                runner.attack_command(argparse.Namespace(command='attack',**selection,output=str(attack),method='ops-view-l2',source='rn50',steps=10,views=1,epsilon_bytes=8.,alpha_mult=1.,seed=task['seed'],view_batch=1,di_probability=.5,resize_rate=1.1,save_float=False,ne=5,np=5,beta=2.,profile=PROFILE))
                validate_task(plan,task,attack,evaluation)
                # Separate target calls make the pinned-manifest check precede each target.
                for target in plan['target_identities']:
                    verify_hashes(plan['pins'])
                    runner.evaluate_command(argparse.Namespace(command='evaluate',**selection,output=str(evaluation/target),attack_run=str(attack),targets=target,population='target-clean',source_clean=plan['clean_reference']['path']))
                merge_evaluations(evaluation,plan,task)
                validate_task(plan,task,attack,evaluation,require_complete=True)
                entry['status']=group['status']='complete'
            except BaseException as error:
                entry['status']=group['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed';group['error']=f'{type(error).__name__}:{error}';raise
            finally:
                group['ended_utc']=utc();group['wall_seconds']=time.perf_counter()-block_start
                entry['total_attempt_wall_seconds']=sum(a.get('wall_seconds',0) for a in entry['attempts'])
                if task['id']==plan['tasks'][0]['id'] and entry['status']=='complete':
                    state['first_block_projected_seconds']=entry['total_attempt_wall_seconds']*12;state['first_block_projected_over_limit']=state['first_block_projected_seconds']>6000
                write_json(state_path,state);append_jsonl(output/'events.jsonl',{'task':task['id'],**group})
        attempt['status']='complete' if sum(b.get('status')=='complete' for b in state['blocks'].values())==12 else 'partial'
    except BaseException as error:
        attempt['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed'
        attempt['error']=f'{type(error).__name__}:{error}'
        raise
    finally:
        attempt['ended_utc']=utc();attempt['wall_seconds']=time.perf_counter()-start;write_json(state_path,state)
    return {'output':str(output),'status':attempt['status']}

def merge_evaluations(root,plan,task):
    """Aggregate two independently resumable target runs without new inference."""
    rows=[];configs=[]
    for target in plan['target_identities']:
        directory=root/target;configs.append(read_json(directory/'config.json'));rows.extend(read_jsonl(directory/'progress.jsonl'))
    # Validate every component before writing explicitly synthetic aggregation metadata.
    validate_task(plan,task,root.parent.parent/'attacks'/task['id'],root)
    cfg={'aggregation':'saved rows from two independent native single-target runner evaluations; no extra inference',
         'selected_ids':task['selected_ids'],'manifest_sha256':plan['manifest_sha256'],'code_sha256':plan['new_code_sha256'],
         'target_identities':{k:v for value in configs for k,v in value['target_identities'].items()}}
    cfg['component_config_sha256']={target:sha256(root/target/'config.json') for target in plan['target_identities']}
    write_json(root/'config.json',cfg)
    temporary=root/'progress.jsonl.tmp'
    import json
    temporary.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows),encoding='utf-8');temporary.replace(root/'progress.jsonl')

def contrast(rows,ids,seeds,complete):
    result={'expected_images':len(ids),'observed_seed_image_pairs':sum(r['id'] in ids and r['seed'] in seeds for r in rows),'seeds':list(seeds),'difference':None,'ci':None}
    if not complete:return result
    differences={seed:{r['id']:r['difference'] for r in rows if r['seed']==seed and r['id'] in ids} for seed in seeds}
    values=image_seed_mean(differences,seeds,ids);ci=paired_bootstrap(values,10000,20261110)
    result.update(difference=ci['estimate'],ci=ci);return result

def report_command(args):
    from .runner import _output
    from .view_l2_report import _csv,_plot_setup,PERTURBATION_KEYS
    source=Path(args.run).resolve();plan=read_json(source/'plan.json');output=_output(args.output)
    verify_hashes(plan['pins']);verify_hashes(plan['old_artifact_sha256'])
    pairs=[];blocks=[];attacks=[];evaluations=[];steps=[];completed=[]
    for task in plan['tasks']:
        raw_a,raw_e,_=validate_task(plan,task,Path(task['old_attack']),Path(task['old_evaluation']),old=True,require_complete=True)
        new_a,new_e,full=validate_task(plan,task,source/'attacks'/task['id'],source/'evaluations'/task['id'])
        if full:completed.append(task['id'])
        blocks.append({'task':task['id'],'seed':task['seed'],'block':task['block'],'complete':full,'attack_images':len(new_a),'evaluation_images':len(new_e),
            'source_forward_images':sum(r['forward_images'] for r in new_a),'source_gradient_images':sum(r['input_gradient_images'] for r in new_a),'target_forward_images':2*len(new_e),
            'attack_seconds':sum(r['elapsed_seconds'] for r in new_a),'peak_cuda_allocated_bytes':max((r['peak_cuda_allocated_bytes'] for r in new_a),default=0),**{k:sum(r[k] for r in new_a)/len(new_a) if new_a else None for k in PERTURBATION_KEYS}})
        attacks.extend({'task':task['id'],'seed':task['seed'],**r} for r in new_a);evaluations.extend({'task':task['id'],'seed':task['seed'],**r} for r in new_e)
        steps.extend({'task':task['id'],'seed':task['seed'],'id':r['id'],**step} for r in new_a for step in r['steps_norm_summary'])
        raw={(r['model'],r['id']):r for r in raw_e}
        for row in new_e:
            old=raw[row['model'],row['id']]
            pairs.append({'seed':task['seed'],'id':row['id'],'target':row['model'],'task':task['id'],'complete_block':full,
                **{f'{name}_{key}':r[key] for name,r in [('raw',old),('control',row)] for key in ('success','adversarial_correct','adversarial_ce','adversarial_margin')}})
    full=len(completed)==12;summary_rows=[];single=[]
    for target in plan['target_identities']:
        for population,ids in [*plan['populations'][target].items(),('all276',plan['selected_ids'])]:
            for endpoint in ('success','adversarial_correct','adversarial_ce','adversarial_margin'):
                selected=[{**r,'difference':float(r['control_'+endpoint])-float(r['raw_'+endpoint])} for r in pairs if r['target']==target and r['complete_block']]
                summary_rows.append({'target':target,'population':population,'endpoint':endpoint,**contrast(selected,ids,SEEDS,full)})
                for seed in SEEDS:
                    complete_seed=all(t['id'] in completed for t in plan['tasks'] if t['seed']==seed)
                    value={'target':target,'population':population,'endpoint':endpoint,'seed':seed,**contrast(selected,ids,[seed],complete_seed)}
                    eligible=[r for r in selected if r['seed']==seed and r['id'] in ids]
                    b=sum(bool(r['raw_success']) and not bool(r['control_success']) for r in eligible);c=sum(bool(r['control_success']) and not bool(r['raw_success']) for r in eligible)
                    value.update(raw_only=b,control_only=c,exact_mcnemar_two_sided=exact_mcnemar(b,c) if complete_seed and population=='target_clean_ids' and endpoint=='success' else None)
                    single.append(value)
    report={'status':'complete' if full else 'partial','complete_blocks':completed,'planned_blocks':12,'main_three_seed':summary_rows,'single_seed_descriptive_uncorrected':single,
        'blocks':blocks,'source_forward_images':sum(r['source_forward_images'] for r in blocks),'source_gradient_images':sum(r['source_gradient_images'] for r in blocks),
        'target_forward_images':sum(r['target_forward_images'] for r in blocks),'planned_source_forward_images':215280,'planned_target_forward_images':3312,
        'protocol_sha256':plan['protocol_sha256'],'compatibility_sha256':plan['compatibility_sha256'],'timing_sha256':plan['timing_sha256'],
        'new_code_sha256':plan['new_code_sha256'],'source_archive':plan['source_archive'],
        'state':read_json(source/'state.json') if (source/'state.json').exists() else None,
        'interpretation':'Known-principle control; three fixed seeds averaged within each image; intervals condition on these seeds. Previously explored dev set. All12 blocks required for every main estimate/CI; no two-seed substitute. Single-seed tests descriptive and uncorrected.'}
    write_json(output/'report.json',report)
    for name,rows in [('blocks',blocks),('paired_images',pairs),('attack_images',attacks),('evaluation_images',evaluations),('step_norm_summary',steps),('main_three_seed',summary_rows),('single_seed',single)]:_csv(output/(name+'.csv'),rows)
    lines=['# View-L2 continuation','',f'Status: {report["status"]}; complete blocks {len(completed)}/12.','',report['interpretation'],'', '| Target | Population | Endpoint | L2−raw | Image CI |','|---|---|---|---:|---|']
    for row in summary_rows:lines.append(f'| {row["target"]} | {row["population"]} | {row["endpoint"]} | {row["difference"]} | {row["ci"]} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plt=_plot_setup();fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for axis,target in zip(axes,plan['target_identities']):
        row=next(r for r in summary_rows if r['target']==target and r['population']=='target_clean_ids' and r['endpoint']=='success')
        if row['ci']:
            ci=row['ci'];axis.errorbar([0],[100*row['difference']],yerr=[[100*(row['difference']-ci['ci95_low'])],[100*(ci['ci95_high']-row['difference'])]],fmt='o')
        axis.axhline(0,color='gray',linewidth=.8);axis.set_xticks([0],['Three fixed seeds']);axis.set_ylabel('ASR L2−raw (percentage points)');axis.set_title(target)
    fig.suptitle(f'{report["status"]}: {len(completed)}/12 complete blocks; image-paired interval')
    fig.savefig(output/'view_l2_continuation.png',dpi=180);fig.savefig(output/'view_l2_continuation.pdf');plt.close(fig)
    return {'output':str(output),'status':report['status'],'main_CI_produced':full}
