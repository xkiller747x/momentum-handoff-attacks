"""CPU-only fixed-pair FTM contrasts and same-mode repeatability reports."""
import copy
from pathlib import Path
import numpy as np

from baseline.ftm import layer_digest
from .ftm_update import compare_images
from .io import read_json,read_jsonl,sha256,write_json
from .metrics import image_seed
from .targeted_runner import verify_saved

SEEDS=(20261030,20261031)
MODES=('full','first-nonzero-only')
ATOL=1e-6
RTOL=1e-5


def recorded_file(run_root,path):
    value=Path(path)
    if value.is_absolute():return value
    for parent in (run_root,*run_root.parents):
        if parent.name=='results':return parent.parent/value
    return value.resolve()


def canonical_config(config,mode=False,seed=False,subset=False):
    """Remove only explicitly allowed differences, retaining unknown fields."""
    value=copy.deepcopy(config);arguments=value['arguments'];actual=value.get('update_mode',arguments.get('ftm_update_mode','full'))
    if actual not in MODES or arguments.get('ftm_update_mode',actual)!=actual:raise ValueError('Inconsistent FTM mode')
    pairs=value['pairs'];ids=value['selected_ids']
    if len(set(ids))!=len(ids) or set(value['target_labels'])!=set(ids):raise ValueError('Invalid selected IDs/target map')
    members=[i for p in pairs for i in p['image_ids']]
    if len(set(members))!=len(members) or set(members)!=set(ids) or layer_digest(pairs)!=value['pair_table_sha256']:raise ValueError('Invalid fixed pair table/SHA')
    for pair in pairs:
        if pair['rng_seed']!=image_seed(arguments['seed'],pair['batch_id']):raise ValueError('Pair RNG seed derivation differs')
    from .targeted_ftm import pair_table
    if pairs!=pair_table([{'id':i} for i in ids],arguments['batch_size'],arguments['seed'],cpu_audit=arguments.get('ftm_batch20_cpu_audit',False)):raise ValueError('Pair table is not the original sorted adjacent cohort grouping')
    if mode:
        suffix='-update-'+actual
        if not value['profile'].endswith(suffix):raise ValueError('Profile is not derived from update mode')
        value['profile']=value['profile'][:-len(suffix)]+'-update-MODE';value['update_mode']='MODE'
        if 'ftm_update_mode' in arguments:arguments['ftm_update_mode']='MODE'
    if seed:
        arguments['seed']='SEED'
        for pair in pairs:pair['rng_seed']='DERIVED_SEED'
        value['pair_table_sha256']=layer_digest(pairs)
    if subset:
        for key in ('selected_ids','target_labels','pairs','pair_table_sha256'):value.pop(key,None)
        for key in ('ids','limit'):arguments.pop(key,None)
    return value


def check_configs(a,b,allow_mode=False,allow_seed=False,subset=False):
    if canonical_config(a,allow_mode,allow_seed,subset)!=canonical_config(b,allow_mode,allow_seed,subset):raise ValueError('FTM configurations differ beyond explicitly allowed mode/seed/cohort fields')
    if subset:
        left={p['batch_id']:p for p in a['pairs']};right={p['batch_id']:p for p in b['pairs']}
        if not (set(left).issubset(right) or set(right).issubset(left)):raise ValueError('Repeat cohorts are not complete-pair subsets')
        for identity in set(left)&set(right):
            if left[identity]!=right[identity]:raise ValueError('Repeat pair membership/RNG differs')
        for identity in set(a['selected_ids'])&set(b['selected_ids']):
            if a['target_labels'][identity]!=b['target_labels'][identity]:raise ValueError('Repeat target labels differ')


def load_attack(root):
    root=Path(root).resolve()
    if not (root/'config.json').exists():return {'root':str(root),'config':None,'rows':{},'batches':{},'complete':False,'cost':{'not_started':True}}
    config=read_json(root/'config.json')
    if config.get('kind')!='targeted-attack' or config['arguments'].get('method')!='ftm':raise ValueError('Expected FTM targeted attack')
    canonical_config(config)
    records=None
    if config.get('manifest_path'):
        if sha256(config['manifest_path'])!=config['manifest_sha256']:raise ValueError('Attack manifest changed')
        records={r['id']:r for r in read_json(config['manifest_path'])['records']}
    if config['arguments'].get('labels_csv') and sha256(recorded_file(root,config['arguments']['labels_csv']))!=config['labels_csv_sha256']:raise ValueError('Attack labels CSV changed')
    rows=read_jsonl(root/'progress.jsonl');mapping={r['id']:r for r in rows};batches={};pairs={p['batch_id']:p for p in config['pairs']}
    if len(mapping)!=len(rows) or not set(mapping).issubset(config['selected_ids']):raise ValueError('Unexpected/duplicate attack IDs')
    for row in rows:
        pair=pairs.get(row['batch_id'])
        if pair is None or row['id'] not in pair['image_ids'] or row['batch_rng_seed']!=pair['rng_seed'] or row['target_label']!=config['target_labels'][row['id']] or row['layer_table_sha256']!=config['layer_table_sha256']:raise ValueError('Attack row pair/target/layer differs')
        if records and (row['label']!=records[row['id']]['label'] or row['canonical_sha256']!=records[row['id']]['canonical_sha256']):raise ValueError('Attack row canonical/label differs')
        verify_saved(row)
        path=Path(row['shared_batch_total_path'])
        if sha256(path)!=row['shared_batch_total_sha256']:raise ValueError('Batch metadata SHA changed')
        metadata=read_json(path)
        if metadata['batch_id']!=pair['batch_id'] or metadata['image_ids']!=pair['image_ids'] or metadata['rng_seed']!=pair['rng_seed'] or metadata['layer_table_sha256']!=config['layer_table_sha256']:raise ValueError('Batch metadata identity differs')
        if not metadata.get('final_generator_state_sha256'):raise ValueError('Final pair RNG SHA missing')
        batches[pair['batch_id']]=metadata
    preparations=read_jsonl(root/'preparation.jsonl');attempts=read_jsonl(root/'attempts.jsonl')
    failed=[r for r in attempts if r.get('status')!='complete']
    cost={'committed_image_F':sum(r['forward_images'] for r in rows),'committed_image_B':sum(r['input_gradient_images'] for r in rows),
        'committed_image_forward_pixels':sum(r.get('forward_pixels',0) for r in rows),'committed_image_gradient_pixels':sum(r.get('input_gradient_pixels',0) for r in rows),
        'qualification_F':sum(r['forward_images'] for r in preparations),'qualification_pixels':sum(r['forward_pixels'] for r in preparations),
        'all_attempt_observed_F':sum(r['actual_forward_images'] for r in attempts) if attempts else None,
        'all_attempt_observed_forward_pixels':sum(r.get('actual_forward_pixels',0) for r in attempts) if attempts else None,
        'all_attempt_observed_completed_B':sum(r.get('completed_input_gradient_images',0) for r in attempts) if attempts else None,
        'failed_attempt_observed_F':sum(r['actual_forward_images'] for r in failed),'failed_attempt_count':len(failed),
        'attempt_wall_seconds':sum(r['elapsed_seconds'] for r in attempts),'qualification_seconds':sum(r['elapsed_seconds'] for r in preparations),
        'committed_batch_wall_seconds':sum(r['elapsed_seconds'] for r in batches.values()),
        'max_shared_batch_peak_cuda_bytes':max((r['peak_cuda_allocated_bytes'] for r in batches.values()),default=0),
        'max_qualification_peak_cuda_bytes':max((r.get('peak_cuda_allocated_bytes',0) for r in preparations),default=0),
        'preparation':preparations,'attempts':attempts,'cost_note':'Image wall is a batch share; batch peak uses max, not sum. Unrecorded in-flight failure backward/evaluation work is unknown.'}
    return {'root':str(root),'config':config,'rows':mapping,'batches':batches,'complete':set(mapping)==set(config['selected_ids']),
        'config_sha256':sha256(root/'config.json'),'progress_sha256':sha256(root/'progress.jsonl') if (root/'progress.jsonl').exists() else None,'cost':cost}


def final_entry(attack,identity):return attack['rows'][identity]['checkpoints'][str(attack['config']['arguments']['steps'])]

def attack_evidence(attack):
    if not attack:return None
    return {key:attack.get(key) for key in ('root','config_sha256','progress_sha256','complete')}
def comparison_tier(value):
    if not value['png_exact_equal']:return 'PNG different; float different' if value['float_available'] and not value['float_exact_equal'] else 'PNG different; float equal' if value['float_available'] else 'PNG different; float missing'
    if value['float_available']:return 'float and PNG same' if value['float_exact_equal'] else 'PNG same, float different'
    return 'PNG same, float missing'


def repeat_report(args):
    from .runner import _output
    from .matrix_report import _csv
    output=_output(args.output);errors=[];attacks=[];comparisons=[];rng=[]
    for path in (args.run_a,args.run_b):
        try:attacks.append(load_attack(path))
        except (ValueError,KeyError,FileNotFoundError) as error:errors.append(str(error));attacks.append(None)
    if all(a and a['config'] for a in attacks):
        try:check_configs(attacks[0]['config'],attacks[1]['config'],subset=True)
        except ValueError as error:errors.append(str(error))
        left={p['batch_id']:p for p in attacks[0]['config']['pairs']};right={p['batch_id']:p for p in attacks[1]['config']['pairs']}
        common=set(left)&set(right);complete_pairs=[]
        for pair_id in sorted(common):
            pair=left[pair_id]
            full=all(set(pair['image_ids']).issubset(a['rows']) for a in attacks)
            if not full:continue
            complete_pairs.append(pair_id);states=[a['batches'][pair_id]['final_generator_state_sha256'] for a in attacks]
            rng.append({'pair_id':pair_id,'equal':states[0]==states[1],'run_a':states[0],'run_b':states[1]})
            if states[0]!=states[1]:errors.append('Repeat final RNG differs: '+pair_id)
            for identity in pair['image_ids']:
                value=compare_images(final_entry(attacks[0],identity),final_entry(attacks[1],identity))
                comparisons.append({'id':identity,'pair_id':pair_id,'tier':comparison_tier(value),**value})
        common_complete=len(complete_pairs)==len(common) and bool(common)
        complete=common_complete and all(a['complete'] for a in attacks)
    else:common=set();complete_pairs=[];complete=False;common_complete=False
    report={'status':'invalid' if errors else 'complete' if complete else 'partial','invalid_reasons':errors,
        'common_pair_count':len(common),'complete_pairs':complete_pairs,'image_comparisons':comparisons,'rng':rng,
        'common_pair_comparison_complete':common_complete,'source_run_complete':[a['complete'] if a else False for a in attacks],
        'attack_artifact_bindings':[attack_evidence(a) for a in attacks],
        'costs':[a['cost'] if a else None for a in attacks],
        'interpretation':'Same-mode same-numeric-profile repeat on common original complete pairs. Differences retained; no target-transfer or equivalence inference.'}
    write_json(output/'report.json',report);_csv(output/'images.csv',comparisons);_csv(output/'rng.csv',rng)
    (output/'report.md').write_text('# FTM same-mode repeat\n\nStatus: '+report['status']+'\n\n'+report['interpretation']+'\n\n'+'\n'.join(errors)+'\n',encoding='utf-8')
    return {'output':str(output),'status':report['status'],'compared_images':len(comparisons)}


def cluster_estimate(records,pairs,seeds,indices,complete):
    """Paired ratio-of-sums; seed mean within fixed pair before resampling."""
    grouped={(r['seed'],r['id']):r for r in records}
    matched=set.intersection(*[{r['id'] for r in records if r['seed']==seed} for seed in seeds]) if seeds else set()
    grouped={key:value for key,value in grouped.items() if key[1] in matched}
    numerator=[];denominator=[];full_count=[];control_count=[]
    full_only=control_only=0;discordant=opposite=0;observed=[]
    for pair in pairs:
        nums=[];dens=[];fs=[];cs=[];b=c=0
        for seed in seeds:
            selected=[grouped[seed,i] for i in pair['image_ids'] if (seed,i) in grouped]
            eligible=[r for r in selected if r['eligible']]
            nums.append(sum(r['full']-r['control'] for r in eligible));dens.append(len(eligible))
            fs.append(sum(r['full'] for r in eligible));cs.append(sum(r['control'] for r in eligible))
            b+=sum(r['full'] and not r['control'] for r in eligible);c+=sum(r['control'] and not r['full'] for r in eligible)
            observed.extend((seed,r['id']) for r in selected)
        numerator.append(float(np.mean(nums)));denominator.append(float(np.mean(dens)));full_count.append(float(np.mean(fs)));control_count.append(float(np.mean(cs)))
        full_only+=b;control_only+=c;discordant+=int(b+c>0);opposite+=int(b>0 and c>0)
    num,den=np.asarray(numerator),np.asarray(denominator);total=float(den.sum())
    result={'difference':float(num.sum()/total) if total else None,'full_rate':float(sum(full_count)/total) if total else None,
        'control_rate':float(sum(control_count)/total) if total else None,'eligible_image_seed_mean_count':total,
        'observed_image_seed_pairs':len(set(observed)),'seeds':list(seeds),'pair_count':len(pairs),
        'matched_image_ids_across_selected_seeds':sorted(matched),
        'complete_observed_pairs':[p['batch_id'] for p in pairs if set(p['image_ids']).issubset(matched)],
        'full_only_image_seed_count':int(full_only),'control_only_image_seed_count':int(control_only),'discordant_pair_count':discordant,
        'opposite_flips_within_pair_count':opposite,'eligible_pair_count':int((den>0).sum()),'bootstrap':None,
        'zero_discordant_pair_binomial_reference_upper':None,'pair_values':[{'pair_id':p['batch_id'],'difference_numerator':float(n),'eligible_denominator':float(d)} for p,n,d in zip(pairs,num,den)]}
    if complete:
        draws_den=den[indices].sum(1);valid=draws_den>0;draws=num[indices].sum(1)[valid]/draws_den[valid]
        low,high=np.percentile(draws,[2.5,97.5]) if len(draws) else (None,None)
        result['bootstrap']={'low':float(low) if low is not None else None,'high':float(high) if high is not None else None,
            'valid_draws':int(valid.sum()),'empty_denominator_draws':int((~valid).sum()),'draws':len(indices),'seed':20261028,
            'degenerate':bool(low==high) if low is not None else None,
            'interpretation':'Fixed-seed/dev-pair percentile range; degenerate ranges are not reliable population coverage or equivalence evidence.'}
        n=result['eligible_pair_count']
        if discordant==0 and n:
            result['zero_discordant_pair_binomial_reference_upper']={'upper':float(1-.025**(1/n)),'n_pairs':n,
                'assumption':'Independent pair Bernoulli occurrence reference, two-sided95%; NOT a targeted-success-rate difference interval.'}
    return result


def load_evaluations(paths,attack,targets):
    mapping={};identities={};numeric={};cost=[]
    for path in paths:
        root=Path(path).resolve()
        if not (root/'config.json').exists():continue
        cfg=read_json(root/'config.json')
        if cfg.get('kind')!='targeted-evaluate' or cfg['attack_config_sha256']!=attack['config_sha256'] or cfg['attack_progress_sha256']!=attack['progress_sha256']:raise ValueError('Evaluation/attack config or progress SHA binding differs')
        if cfg['code_sha256']!=attack['config']['code_sha256'] or cfg['numeric_settings']!=attack['config']['numeric_settings']:raise ValueError('Evaluation code/numeric identity differs from bound attack')
        if 'source_identity' in cfg and cfg['source_identity']!=attack['config']['source_identity']:raise ValueError('Evaluation source identity differs from bound attack')
        source_name=attack['config']['arguments']['source']
        if source_name in cfg['target_identities'] and cfg['target_identities'][source_name]!=attack['config']['source_identity']:raise ValueError('Evaluation source target identity differs from bound attack')
        if cfg['manifest_sha256']!=attack['config']['manifest_sha256'] or cfg['labels_csv_sha256']!=attack['config']['labels_csv_sha256'] or cfg['selected_ids']!=attack['config']['selected_ids'] or cfg['target_labels']!=attack['config']['target_labels']:raise ValueError('Evaluation cohort/targets differ')
        if cfg['arguments'].get('checkpoint') not in (None,attack['config']['arguments']['steps']):raise ValueError('Expected final checkpoint evaluation')
        rows=read_jsonl(root/'progress.jsonl')
        for name,identity in cfg['target_identities'].items():
            if name not in targets or name in identities:raise ValueError('Unexpected/duplicate target evaluation model')
            identities[name]=identity;numeric[name]=cfg['numeric_settings']
        for row in rows:
            key=row['model'],row['id']
            if key in mapping or row['model'] not in cfg['target_identities'] or row['id'] not in attack['rows']:raise ValueError('Unexpected/duplicate evaluation image')
            source=attack['rows'][row['id']];entry=final_entry(attack,row['id'])
            if row['target_label']!=source['target_label'] or row['label']!=source['label'] or row['png_sha256']!=entry['png_sha256'] or row['checkpoint_step']!=attack['config']['arguments']['steps'] or row['source_clean_correct']!=source['source_clean']['true_correct']:raise ValueError('Evaluation image/PNG/target/source-clean join differs')
            if row['adversarial']['targeted_success']!=(row['adversarial']['prediction']==row['target_label']):raise ValueError('Non-targeted failure misreported as targeted success')
            mapping[key]=row
        cost.append({'path':str(root),'committed_F':sum(r['forward_images'] for r in rows),'committed_rows':len(rows),
            'config_sha256':sha256(root/'config.json'),'progress_sha256':sha256(root/'progress.jsonl') if (root/'progress.jsonl').exists() else None,
            'wall_seconds':sum(r['elapsed_seconds'] for r in rows),'max_peak_cuda_bytes':max((r.get('peak_cuda_allocated_bytes',0) for r in rows),default=0),
            'cost_note':'Only committed clean+PNG rows; interrupted unrecorded work is unknown; no reuse or new model call.'})
    return mapping,identities,numeric,cost


def pair_report(args):
    from .runner import _output
    from .matrix_report import _csv
    plan_path=Path(args.plan).resolve();plan=read_json(plan_path);output=_output(args.output);errors=[]
    ids=plan['cohort_ids'];seeds=plan.get('seeds',list(SEEDS));targets=plan.get('targets',['rn50','dn121','vit_b_16'])
    if len(set(ids))!=len(ids) or not ids or len(seeds)!=2 or len(set(seeds))!=2:raise ValueError('Unique cohort and two predefined seeds required')
    if plan.get('freeze'):
        if sha256(plan['protocol'])!=read_json(plan['freeze'])['protocol_sha256']:raise ValueError('Frozen protocol SHA mismatch')
    expected={(seed,mode) for seed in seeds for mode in MODES};cells={};costs=[];base=None;target_ids={};numeric={};eligibility={};same_png={};checks=[];clean_checks=[]
    for cell in plan['cells']:
        key=cell['seed'],cell['mode']
        if key not in expected or key in cells:raise ValueError('Unexpected/duplicate plan cell')
        value={'cell':cell,'attack':None,'evaluation':{},'complete':False};cells[key]=value
        cost_entry={'seed':cell['seed'],'mode':cell['mode'],'source':None,'evaluation':[]};costs.append(cost_entry)
        try:
            attack=load_attack(cell['attack_run']);value['attack']=attack
            cost_entry['source']=attack['cost']
            if not attack['config']:continue
            cfg=attack['config']
            if cfg['selected_ids']!=ids or cfg['arguments']['seed']!=cell['seed'] or cfg['update_mode']!=cell['mode']:raise ValueError('Plan and attack cell identity differs')
            ns=cfg['numeric_settings']
            if plan.get('deterministic',True) and not (cfg['arguments'].get('deterministic') and ns['deterministic_algorithms'] and not ns['deterministic_warn_only'] and ns['cudnn_deterministic'] and not ns['cudnn_benchmark'] and not ns['cuda_matmul_allow_tf32'] and not ns['cudnn_allow_tf32'] and ns['cublas_workspace_config'] in (':4096:8',':16:8')):raise ValueError('Expected strict deterministic dev profile')
            if base is None:base=cfg
            else:check_configs(base,cfg,allow_mode=True,allow_seed=True)
            paths=cell.get('evaluation_runs',[cell['evaluation_run']] if cell.get('evaluation_run') else [])
            evaluation,identities,settings,evaluation_cost=load_evaluations(paths,attack,targets);value['evaluation']=evaluation
            cost_entry['evaluation']=evaluation_cost
            for model,identity in identities.items():
                if model in target_ids and (target_ids[model]!=identity or numeric[model]!=settings[model]):raise ValueError('Target weights/numeric profile differ across cells')
                target_ids[model]=identity;numeric[model]=settings[model]
            for (model,identity),row in evaluation.items():
                signature=bool(row['clean']['true_correct']),bool(row['source_clean_correct'])
                clean_checks.append({'seed':cell['seed'],'mode':cell['mode'],'model':model,'id':identity,'target_clean':signature[0],'source_clean':signature[1]})
                if (model,identity) in eligibility and eligibility[model,identity]!=signature:raise ValueError(f'Clean populations differ across mode/seed: {model}/{identity}')
                eligibility[model,identity]=signature
                png_key=model,row['png_sha256'],row['target_label']
                previous=same_png.get(png_key)
                if previous:
                    changed=[];violations=[]
                    for metric in ('prediction','target_rank','target_logit','target_margin','target_log_odds','target_confidence'):
                        a,b=previous['adversarial'][metric],row['adversarial'][metric]
                        if a!=b:changed.append({'metric':metric,'a':a,'b':b,'absolute_difference':abs(a-b)})
                        equal=a==b if metric in ('prediction','target_rank') else bool(np.isclose(a,b,atol=ATOL,rtol=RTOL))
                        if not equal:violations.append(metric)
                    if changed:checks.append({'model':model,'id':identity,'png_sha256':row['png_sha256'],'differences':changed,'violations':violations})
                    if violations:errors.append(f'Identical PNG/model metric inconsistency: {model}/{identity}/{violations}')
                else:same_png[png_key]=row
            value['complete']=attack['complete'] and set(evaluation)=={(m,i) for m in targets for i in ids}
        except (ValueError,KeyError,FileNotFoundError) as error:errors.append(f'{key}: {error}')
    if set(cells)!=expected:errors.append('Plan does not contain exactly four predefined cells')
    pairs=base['pairs'] if base else [];rng=[];images=[];joined=[]
    for seed in seeds:
        a=cells.get((seed,'full'),{}).get('attack');b=cells.get((seed,'first-nonzero-only'),{}).get('attack')
        if not a or not b or not a['config'] or not b['config']:continue
        for pair in pairs:
            left,right=a['batches'].get(pair['batch_id']),b['batches'].get(pair['batch_id'])
            if left and right:
                equal=left['final_generator_state_sha256']==right['final_generator_state_sha256']
                rng.append({'seed':seed,'pair_id':pair['batch_id'],'equal':equal,'full':left['final_generator_state_sha256'],'control':right['final_generator_state_sha256']})
                if not equal:errors.append(f'Final pair RNG mismatch: {seed}/{pair["batch_id"]}')
        for identity in ids:
            if identity not in a['rows'] or identity not in b['rows']:continue
            image=compare_images(final_entry(a,identity),final_entry(b,identity));images.append({'seed':seed,'id':identity,'tier':comparison_tier(image),**image})
            for model in targets:
                full=cells[seed,'full']['evaluation'].get((model,identity));control=cells[seed,'first-nonzero-only']['evaluation'].get((model,identity))
                if full and control:joined.append({'seed':seed,'id':identity,'model':model,'full':int(full['adversarial']['targeted_success']),'control':int(control['adversarial']['targeted_success']),
                    **{f'{mode}_{metric}':row['adversarial'][metric] for mode,row in [('full',full),('control',control)] for metric in ('target_logit','target_margin','target_log_odds','target_rank')}})
    complete=set(cells)==expected and all(v['complete'] for v in cells.values());valid=not errors;statistics=[]
    indices=np.random.default_rng(20261028).integers(0,len(pairs),size=(10000,len(pairs))) if pairs else np.empty((10000,0),dtype=int)
    for model in targets:
        for population in ('all','target-clean','source-target-clean'):
            eligible=[i for i in ids if population=='all' or (model,i) in eligibility and eligibility[model,i][0] and (population!='source-target-clean' or eligibility[model,i][1])]
            values=[{**r,'eligible':r['id'] in eligible} for r in joined if r['model']==model]
            for selected in [[s] for s in seeds]+[seeds]:
                estimate=cluster_estimate(values,pairs,selected,indices,complete and valid)
                estimate.update(model=model,population=population,eligible_ids=eligible,clean_eligibility_unknown_ids=[i for i in ids if (model,i) not in eligibility],
                    estimand='seed-average' if len(selected)>1 else 'per-seed',expected_images=len(ids),valid_identity=valid)
                continuous={}
                observed=[r for r in values if r['eligible'] and r['seed'] in selected and r['id'] in estimate['matched_image_ids_across_selected_seeds']]
                for metric in ('target_logit','target_margin','target_log_odds','target_rank'):
                    continuous[metric]={'full_mean':float(np.mean([r['full_'+metric] for r in observed])) if observed else None,
                        'control_mean':float(np.mean([r['control_'+metric] for r in observed])) if observed else None,
                        'full_minus_control_mean':float(np.mean([r['full_'+metric]-r['control_'+metric] for r in observed])) if observed else None}
                estimate['within_model_descriptive_metrics']=continuous;statistics.append(estimate)
    report={'status':'invalid' if errors else 'complete' if complete else 'partial','invalid_reasons':errors,'plan_sha256':sha256(plan_path),
        'cells':[{'seed':s,'mode':m,'complete':v['complete'],'completed_attack_ids':list(v['attack']['rows']) if v['attack'] else [],
            'evaluation_rows':len(v['evaluation']),'attack_artifact_binding':attack_evidence(v['attack'])} for (s,m),v in cells.items()],
        'statistics':statistics,'image_comparisons':images,'rng':rng,'costs':costs,'same_png_metric_checks':checks,'clean_population_checks':clean_checks,
        'metric_tolerance':{'atol':ATOL,'rtol':RTOL,'argmax_and_rank':'exact'},'repeat_run_reference':plan.get('repeat_run'),
        'interpretation':'Full minus first-nonzero-only. PNG primary; pair clusters and fixed seeds, never seed IID. Partial/invalid comparisons have no complete bootstrap. Degenerate ranges are descriptive finite-sample facts, not global equivalence or reliable95% coverage. Continuous metrics only within model; no new attack or causal claim.'}
    write_json(output/'report.json',report)
    for name,rows in [('statistics',statistics),('images',images),('rng',rng),('joined_evaluation',joined),('same_png_checks',checks),('clean_populations',clean_checks)]:_csv(output/(name+'.csv'),rows)
    _csv(output/'costs.csv',[{'seed':r['seed'],'mode':r['mode'],**{k:v for k,v in (r['source'] or {}).items() if k not in ('preparation','attempts')},
        'committed_eval_F':sum(e['committed_F'] for e in r['evaluation']),'evaluation_evidence':r['evaluation']} for r in costs])
    _csv(output/'pair_values.csv',[{'model':r['model'],'population':r['population'],'estimand':r['estimand'],'seeds':r['seeds'],**p} for r in statistics for p in r['pair_values']])
    lines=['# FTM fixed-pair known-control report','','Status: '+report['status'],'',report['interpretation'],'',
        '| Model | Seeds | Full rate | Control rate | Full − control | Pair percentile range | Observed / expected image-seed rows |',
        '|---|---|---:|---:|---:|---|---:|']
    def number(value):return 'missing' if value is None else f'{value:.6f}'
    for row in statistics:
        if row['population']!='all':continue
        interval=row['bootstrap']
        text=(f"[{number(interval['low'])}, {number(interval['high'])}]"+(' (degenerate)' if interval['degenerate'] else '')) if interval else report['status']+'; no complete interval'
        lines.append(f"| {row['model']} | {','.join(map(str,row['seeds']))} | {number(row['full_rate'])} | {number(row['control_rate'])} | {number(row['difference'])} | {text} | {row['observed_image_seed_pairs']} / {len(ids)*len(row['seeds'])} |")
    if errors:lines+=['','Identity failures:','']+['- '+error for error in errors]
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'output':str(output),'status':report['status'],'complete_bootstrap_produced':complete and valid}
