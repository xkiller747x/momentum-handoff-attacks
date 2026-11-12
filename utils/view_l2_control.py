"""Read-only source probe, engineering and frozen known-principle control flow."""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone,timedelta
import hashlib
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from baseline.norm_control import PROFILE,control_attack,view_gradient_mean
from baseline.ops import apply_ops
from baseline.ops_refresh import OPSRefreshContext
from .io import append_jsonl,image_tensor,read_json,read_jsonl,read_pixels,save_adversarial,sha256,write_json
from .metrics import image_seed
from .models import ROOT,load_model
from .ops_profile import _source_archive,gradient_comparison,profile_metadata
from .ops_seed_replication import FROZEN_KERNELS,cpu_pair_identity
from .pool_diagnostic import _hash_map,namespace_seed


def utc():return datetime.now(timezone.utc).isoformat()
def parsed(value):return datetime.fromisoformat(value.replace('Z','+00:00')).astimezone(timezone.utc)


def metadata(args):
    old_root=Path(args.profile_run).resolve();old=read_json(old_root/'plan.json')
    protocol,freeze=Path(args.protocol).resolve(),read_json(args.freeze)
    if sha256(protocol)!=freeze['protocol_sha256']:raise ValueError('Frozen view-L2 protocol SHA mismatch')
    base,_related,records,_clean=profile_metadata(old['related_run'],old['protocol'],old['freeze'])
    prior=_hash_map(old['new_code_sha256']);kernels={key:sha256(ROOT/key) for key in FROZEN_KERNELS}
    if any(value!=prior.get(key) for key,value in kernels.items()):raise ValueError('Old numerical kernel SHA changed')
    if old['selected_ids']!=base['selected_ids'] or old['manifest_sha256']!=base['manifest_sha256']:raise ValueError('Old raw-fresh dataset identity differs')
    inputs={};artifacts={}
    for block,ids in enumerate(old['blocks'],1):
        attack_root=old_root/'attacks'/f'b{block}'
        for group in ('attacks','evaluations'):
            for name in ('config.json','progress.jsonl'):
                path=old_root/group/f'b{block}'/name;artifacts[str(path)]=sha256(path)
        config=read_json(attack_root/'config.json');arguments=config['arguments']
        wanted={'method':'ops-refresh','profile':'ops-author-noiserefresh','source':'rn50','steps':10,'views':1,'ne':5,'np':5,
                'beta':2.,'epsilon_bytes':8.,'alpha_mult':1.,'seed':20261027,'view_batch':1}
        if any(arguments.get(key)!=value for key,value in wanted.items()) or config['selected_ids']!=ids or config['source_identity']!=base['source_identity']:
            raise ValueError('Old raw fresh configuration differs')
        if _hash_map(config['code_sha256'])!=_hash_map(old['new_code_sha256']):raise ValueError('Old raw code/plan identity differs')
        rows=read_jsonl(attack_root/'progress.jsonl')
        if len(rows)!=69 or {r['id'] for r in rows}!=set(ids):raise ValueError('Old raw-fresh block incomplete')
        for row in rows:
            if row['image_seed']!=image_seed(20261027,row['id']) or row['noise_pool_generation_count']!=10:raise ValueError('Old raw-fresh sampler differs')
            inputs[row['id']]={'png':str(attack_root/'images'/f'{row["id"]}.png'),'row':row}
    anchor=old_root.parent/'ops_seed_replication_v1'/'plan.json'
    if anchor.exists():
        anchored={str(Path(key).resolve()).casefold():value for key,value in read_json(anchor)['old_artifact_sha256'].items()}
        if any(anchored.get(str(Path(key).resolve()).casefold())!=value for key,value in artifacts.items()):
            raise ValueError('Old raw config/progress SHA differs from frozen seed-replication artifact anchor')
        anchor_note={'path':str(anchor),'sha256':sha256(anchor),'status':'matched historical seed-plan old_artifact_sha256; root separately verifies anchor freeze'}
    else:
        preflight=old_root.parent/'view_l2_old_raw_integrity_v1.json'
        anchor_note={'status':'current SHA recorded; historical match is external root preflight',
                     'root_preflight_path':str(preflight),'root_preflight_sha256':sha256(preflight) if preflight.exists() else None}
    result={'profile_run':str(old_root),'old_profile_plan_sha256':sha256(old_root/'plan.json'),'old_artifact_sha256':artifacts,'old_artifact_anchor':anchor_note,
        'old_code_sha256':old['new_code_sha256'],'kernel_sha256':kernels,'protocol':str(protocol),'protocol_sha256':sha256(protocol),
        'freeze_sha256':sha256(args.freeze),'freeze':freeze,'manifest':base['manifest'],'manifest_sha256':base['manifest_sha256'],
        'selected_ids':base['selected_ids'],'blocks':base['blocks'],'source_identity':base['source_identity'],'target_identities':base['target_identities'],
        'clean_reference':base['clean_reference'],'populations':base['populations'],'method':'ops-view-l2','profile':PROFILE,
        'seed':20261027,'source_forward_images':71760,'source_gradient_images':71760,'target_forward_images':1104}
    return result,records,inputs


def open_output(args,mode,ids):
    from . import runner
    output=runner._output(args.output)
    if not output.relative_to(ROOT/'results').parts[0].startswith('view_l2_'):raise ValueError('Use results/view_l2_* output directories')
    plan,records,inputs=metadata(args)
    config={**plan,'mode':mode,'ids':ids(plan),'device':getattr(args,'device','cpu'),'new_code_sha256':runner._code_identity()}
    config['source_archive']=_source_archive(output,config['new_code_sha256'])
    path=output/'config.json'
    if path.exists() and read_json(path)!=config:raise ValueError('View-L2 output configuration differs')
    write_json(path,config)
    return output,config,records,inputs


def checked_images(record,entry,device):
    if sha256(record['canonical_path'])!=record['canonical_sha256'] or sha256(entry['png'])!=entry['row']['png_sha256']:
        raise ValueError('Canonical/old raw PNG SHA mismatch')
    if record['label']!=entry['row']['label'] or record['canonical_sha256']!=entry['row']['canonical_sha256']:
        raise ValueError('Old raw image/label/canonical identity differs')
    return image_tensor(record['canonical_path'],device),image_tensor(entry['png'],device)


def source_model(config,device_name):
    from .runner import _device
    device=_device(device_name)
    identity=config['source_identity']
    if sha256(identity['checkpoint'])!=identity['checkpoint_sha256']:raise ValueError('Source weight SHA mismatch')
    model,loaded=load_model('rn50',device)
    if loaded!=identity:raise ValueError('Source model identity differs')
    return model,device


def counter():return {'forward_images':0,'input_gradient_images':0,'forward_pixels':0,'input_gradient_pixels':0,'resolution_counts':{}}


def actual_hook(model):
    actual={'forward_images':0}
    def count(_module,inputs):actual['forward_images']+=inputs[0].shape[0]
    return actual,model.register_forward_pre_hook(count)


def complete(output,rows,extra=None):
    summary={'count':len(rows),'forward_images':sum(r.get('forward_images',0) for r in rows),
             'input_gradient_images':sum(r.get('input_gradient_images',0) for r in rows),'target_forward_images':0,
             'completed_utc':utc(),**(extra or {})}
    write_json(output/'summary.json',summary);return summary


def sampling_check_command(args):
    output,config,records,inputs=open_output(args,'all276_CPU_sampling_identity',lambda p:p['selected_ids'])
    rows=read_jsonl(output/'progress.jsonl');done={row['id'] for row in rows}
    for record in records:
        if record['id'] in done:continue
        value=cpu_pair_identity(20261027,record['id'])
        if value['refresh_pool_sha256_by_pool']!=inputs[record['id']]['row']['noise_pool_sha256_by_pool']:
            raise ValueError('CPU reconstructed10 pools differ from actual old raw-fresh pools')
        write_json(output/'identities'/f'{record["id"]}.json',value)
        append_jsonl(output/'progress.jsonl',{'id':record['id'],'forward_images':0,'input_gradient_images':0,
            'identity_sha256':sha256(output/'identities'/f'{record["id"]}.json'),'reconstruction_seconds':value['reconstruction_seconds']})
        rows.append({'id':record['id'],'forward_images':0,'input_gradient_images':0,'reconstruction_seconds':value['reconstruction_seconds']})
    return complete(output,rows,{'all276_passed':len(rows)==276,'CPU_reconstruction_seconds':sum(r['reconstruction_seconds'] for r in rows)})


def source_probe_command(args):
    output,config,records,inputs=open_output(args,'source24_fixed_points',lambda p:p['selected_ids'][:24])
    rows=read_jsonl(output/'progress.jsonl');done={row['id'] for row in rows};model,device=source_model(config,args.device)
    for record in records[:24]:
        if record['id'] in done:continue
        start=time.perf_counter();clean,png=checked_images(record,inputs[record['id']],device)
        seed=namespace_seed('view-l2-source-probe',20261109,record['id'])
        context=OPSRefreshContext(clean.shape,8/255,5,5,2.,seed);plans=context.step_plans()
        descriptions=[{'pool_index':p.pool_index,'operators':[asdict(op) for op in p.operators]} for p in plans]
        plan_sha=hashlib.sha256(json.dumps(descriptions,sort_keys=True).encode()).hexdigest()
        actual,handle=actual_hook(model);stats=counter();states={};details=[]
        try:
            for name,image in [('clean',clean),('old_raw_png',png)]:
                _mean,description,views=view_gradient_mean(model,image,torch.tensor([record['label']],device=device),plans,stats)
                states[name]=description;details.extend({'state':name,**view} for view in views)
        finally:handle.remove()
        if actual['forward_images']!=52 or stats['input_gradient_images']!=52:raise ValueError('Source probe actual52F/B mismatch')
        row={'id':record['id'],'label':record['label'],'namespace_seed':seed,'plan_sha256':plan_sha,'states':states,
             'views':details,'sampler':context.lifecycle_record(),'actual_hook':actual,**stats,'target_forward_images':0,
             'elapsed_seconds':time.perf_counter()-start,'committed_utc':utc(),
             'interpretation':'Two fixed points only, not intermediate training trajectory; participation ratio is not independent effective sample size'}
        write_json(output/'details'/f'{record["id"]}.json',row);append_jsonl(output/'progress.jsonl',row);rows.append(row)
        print(f'view-L2 source probe {record["id"]}:52F/B,0 targetF',flush=True)
    return complete(output,rows,{'planned_forward_images':1248,'planned_input_gradient_images':1248})


def engineering_command(args):
    output,config,records,inputs=open_output(args,f'engineering_{args.phase}',lambda p:p['selected_ids'][:2])
    rows=read_jsonl(output/'progress.jsonl');done={row['id'] for row in rows};model,device=source_model(config,args.device)
    for record in records[:2]:
        if record['id'] in done:continue
        start=time.perf_counter();clean,_old_png=checked_images(record,inputs[record['id']],device)
        label=torch.tensor([record['label']],device=device);seed=image_seed(20261027,record['id'])
        actual,handle=actual_hook(model)
        try:
            if args.phase=='first-step':
                plans=OPSRefreshContext(clean.shape,8/255,5,5,2.,seed).step_plans();stats=counter()
                gradient,description,_views=view_gradient_mean(model,clean,label,plans,stats)
                # Independent reference: CE gradient, FP64 norm/division, then mean.
                independent=torch.zeros_like(clean)
                for plan in plans:
                    leaf=clean.detach().requires_grad_(True);loss=F.cross_entropy(model(apply_ops(leaf,plan)),label)
                    raw=torch.autograd.grad(loss,leaf)[0];norm=raw.double().square().sum().sqrt()
                    if norm.item()!=0:independent+=(raw.double()/norm).float()
                independent/=26
                row={'id':record['id'],'phase':args.phase,'comparison':gradient_comparison(independent,gradient),
                     'forward_images':52,'input_gradient_images':52,'step_norm_summary':description}
            else:
                adv,stats=control_attack(model,clean,label,seed=seed,identity=args.phase=='identity')
                path=output/'images'/f'{record["id"]}.png'
                png_stats=save_adversarial(path,adv,read_pixels(record['canonical_path']),8)
                row={'id':record['id'],'phase':args.phase,**stats,**png_stats}
                if args.phase=='identity':
                    a,b=read_pixels(path),read_pixels(inputs[record['id']]['png'])
                    difference=(a.astype('int16')-b.astype('int16'))
                    row['old_raw_png_comparison']={'same_sha':png_stats['png_sha256']==inputs[record['id']]['row']['png_sha256'],
                        'different_channel_elements':int((difference!=0).sum()),'max_abs_byte':int(abs(difference).max()),
                        'old_sha256':inputs[record['id']]['row']['png_sha256']}
        finally:handle.remove()
        expected=52 if args.phase=='first-step' else 260
        if actual['forward_images']!=expected or row['input_gradient_images']!=expected:raise ValueError('Engineering actual F/B mismatch')
        row.update({'actual_hook':actual,'target_forward_images':0,'total_wall_seconds':time.perf_counter()-start,'committed_utc':utc()})
        write_json(output/'details'/f'{record["id"]}.json',row);append_jsonl(output/'progress.jsonl',row);rows.append(row)
    return complete(output,rows,{'phase':args.phase,'planned_forward_images':104 if args.phase=='first-step' else 520})


def engineering_gate(config,args):
    """Persist evidence only; no norm/effect threshold selects formal execution."""
    expected=[('sampling_run','all276_CPU_sampling_identity',276,0),('probe_run','source24_fixed_points',24,1248),
              ('first_step_run','engineering_first-step',2,104),('control_run','engineering_control',2,520),
              ('identity_run','engineering_identity',2,520)]
    evidence={}
    for argument,mode,count,forwards in expected:
        value=getattr(args,argument,None)
        if not value:raise ValueError(f'Formal requires --{argument.replace("_","-")} evidence')
        root=Path(value).resolve();old_config=read_json(root/'config.json');summary=read_json(root/'summary.json')
        if old_config['mode']!=mode or old_config['protocol_sha256']!=config['protocol_sha256'] or old_config['old_profile_plan_sha256']!=config['old_profile_plan_sha256']:
            raise ValueError('Engineering/sampling proof uses another protocol or raw baseline')
        if old_config['new_code_sha256']!=config['new_code_sha256']:raise ValueError('Engineering code identity differs from formal')
        if summary['count']!=count or summary['forward_images']!=forwards or summary['input_gradient_images']!=forwards:
            raise ValueError('Engineering/sampling proof count incomplete')
        if parsed(summary['completed_utc'])>parsed(config['freeze']['engineering_must_complete_utc']):
            raise ValueError('Engineering/sampling was not complete before frozen03:50:11 gate')
        evidence[argument]={'path':str(root),'config_sha256':sha256(root/'config.json'),'summary_sha256':sha256(root/'summary.json'),
                            'completed_utc':summary['completed_utc']}
    identity_rows=read_jsonl(Path(args.identity_run)/'progress.jsonl')
    if any(not row['old_raw_png_comparison']['same_sha'] for row in identity_rows):
        raise ValueError('Identity PNG differs; investigate exact discrepancy before any formal execution')
    control_rows=read_jsonl(Path(args.control_run)/'progress.jsonl')
    seconds=sum(row['total_wall_seconds'] for row in control_rows)
    evidence['conservative_formal_projection_seconds']=1.2*seconds/520*71760
    return evidence


def may_start_block(config,state,now,elapsed):
    freeze=config['freeze']
    if elapsed>=2400:return False,'2400-second cumulative boundary cap'
    if state.get('first_block_projected_over_limit'):return False,'first complete block×4 exceeds2400seconds'
    if now>parsed(freeze['last_new_block_utc']):return False,'last new block UTC reached'
    longest=max((b.get('total_attempt_wall_seconds',0) for b in state.get('blocks',{}).values() if b.get('status')=='complete'),default=0)
    if longest and now+timedelta(seconds=longest)>parsed(freeze['last_estimated_block_end_utc']):
        return False,'longest completed block predicts end after04:45:11UTC'
    return True,None


def matrix_command(args):
    from . import runner
    start=time.perf_counter();output,config,records,inputs=open_output(args,'formal276',lambda p:p['selected_ids'])
    write_json(output/'plan.json',config)
    if not args.execute:return {'output':str(output),'blocks':4,'source_forward_images':71760,'source_gradient_images':71760,'target_forward_images':1104,'executed':False}
    now=datetime.now(timezone.utc)
    if now>parsed(config['freeze']['formal_first_start_latest_utc']):raise ValueError('Frozen formal first-start deadline passed; no formal run')
    evidence=engineering_gate(config,args)
    if now+timedelta(seconds=evidence['conservative_formal_projection_seconds'])>parsed(config['freeze']['last_estimated_block_end_utc']):
        raise ValueError('Control engineering conservative throughput projection exceeds04:45:11UTC')
    write_json(output/'engineering_gate.json',evidence)
    old_state=read_json(Path(config['profile_run'])/'state.json')
    if len(old_state['blocks'])!=4 or any(b.get('status')!='complete' for b in old_state['blocks'].values()):raise ValueError('Old raw fresh baseline incomplete')
    state_path=output/'state.json';state=read_json(state_path) if state_path.exists() else {'config_sha256':sha256(output/'config.json'),'blocks':{},'attempts':[]}
    if state['config_sha256']!=sha256(output/'config.json'):raise ValueError('Formal state/config identity differs')
    attempt={'started_utc':utc(),'status':'running'};state['attempts'].append(attempt);write_json(state_path,state)
    clean={(r['model'],r['id']):r for r in read_jsonl(Path(config['clean_reference']['path'])/'progress.jsonl')}
    try:
        for block,ids in enumerate(config['blocks'],1):
            if state['blocks'].get(str(block),{}).get('status')=='complete':continue
            elapsed=sum(a.get('wall_seconds',0) for a in state['attempts'])+time.perf_counter()-start
            allowed,reason=may_start_block(config,state,datetime.now(timezone.utc),elapsed)
            if not allowed:attempt['stop_reason']=reason;break
            entry=state['blocks'].setdefault(str(block),{'attempts':[]});group={'started_utc':utc(),'status':'running'};entry['attempts'].append(group);entry['status']='running';write_json(state_path,state)
            block_start=time.perf_counter()
            try:
                selection=dict(manifest=config['manifest'],split='dev',allow_holdout=False,limit=None,ids=','.join(ids),device=args.device)
                attack_output=output/'attacks'/f'b{block}';eval_output=output/'evaluations'/f'b{block}'
                attack_args=argparse.Namespace(command='attack',**selection,output=str(attack_output),method='ops-view-l2',source='rn50',steps=10,views=1,
                    epsilon_bytes=8.,alpha_mult=1.,seed=20261027,view_batch=1,di_probability=.5,resize_rate=1.1,save_float=False,ne=5,np=5,beta=2.,profile=PROFILE)
                eval_args=argparse.Namespace(command='evaluate',**selection,output=str(eval_output),attack_run=str(attack_output),targets='dn121,vit_b_16',
                    population='target-clean',source_clean=config['clean_reference']['path'])
                runner.attack_command(attack_args)
                generated=read_jsonl(attack_output/'progress.jsonl')
                for row in generated:
                    if row['noise_pool_sha256_by_pool']!=inputs[row['id']]['row']['noise_pool_sha256_by_pool']:
                        raise ValueError('New control sampler actual10-pool SHA differs from old raw')
                runner.evaluate_command(eval_args)
                if any(row['clean_prediction']!=clean[(row['model'],row['id'])]['prediction'] for row in read_jsonl(eval_output/'progress.jsonl')):
                    raise ValueError('New target clean argmax differs from frozen reference')
                entry['status']=group['status']='complete'
            except BaseException as error:
                entry['status']=group['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed';group['error']=f'{type(error).__name__}:{error}';raise
            finally:
                group['ended_utc']=utc();group['wall_seconds']=time.perf_counter()-block_start
                entry['total_attempt_wall_seconds']=sum(a.get('wall_seconds',0) for a in entry['attempts'])
                if block==1 and entry['status']=='complete':
                    state['first_block_projected_seconds']=4*entry['total_attempt_wall_seconds'];state['first_block_projected_over_limit']=state['first_block_projected_seconds']>2400
                write_json(state_path,state);append_jsonl(output/'events.jsonl',{'block':block,**group})
        attempt['status']='complete' if sum(b.get('status')=='complete' for b in state['blocks'].values())==4 else 'partial'
    except BaseException as error:
        attempt['status']='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed';attempt['error']=f'{type(error).__name__}:{error}';raise
    finally:
        attempt['ended_utc']=utc();attempt['wall_seconds']=time.perf_counter()-start;write_json(state_path,state)
    return {'output':str(output),'status':attempt['status'],'complete_blocks':sum(b.get('status')=='complete' for b in state['blocks'].values())}
