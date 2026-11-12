"""Image-unit two-new-seed estimands and exact per-seed discordant checks."""
import math
from pathlib import Path

import numpy as np

from .io import read_json,read_jsonl,sha256,write_json
from .matrix_report import _csv,_mean,PERTURBATION_KEYS
from .metrics import paired_bootstrap,image_seed
from .ops_profile_report import collect_profile
from .ops_seed_replication import ALL_SEEDS,NEW_SEEDS,_check_unit_outputs
from .pool_diagnostic import _hash_map


def exact_mcnemar(fixed_only,refresh_only):
    if fixed_only<0 or refresh_only<0:raise ValueError('Discordants must be nonnegative counts')
    n=fixed_only+refresh_only
    if n==0:return 1.
    tail=sum(math.comb(n,k) for k in range(min(fixed_only,refresh_only)+1))
    return min(1.,2*tail/(2**n))


def image_seed_mean(differences,seeds,ids):
    if any(seed not in differences or set(differences[seed])!=set(ids) for seed in seeds):
        raise ValueError('Seed mean requires the complete fixed image population for every required seed')
    return np.array([[differences[seed][identity] for identity in ids] for seed in seeds],dtype=np.float64).mean(axis=0)


def summarize_seed_results(plan,data,seed_complete):
    per_seed,estimands,ranges,readings=[],[],[],[]
    for target in plan['targets']:
        for population,eligible in [*plan['populations'][target].items(),('all276',plan['selected_ids'])]:
            for seed in ALL_SEEDS:
                fixed=data.get(seed,{}).get('fixed',{}).get(target,{})
                refresh=data.get(seed,{}).get('refresh',{}).get(target,{})
                available=[identity for identity in eligible if identity in fixed and identity in refresh]
                b=sum(fixed[i]['success'] and not refresh[i]['success'] for i in available)
                c=sum(refresh[i]['success'] and not fixed[i]['success'] for i in available)
                row={'seed':seed,'target':target,'population':population,'observed_images':len(available),'expected_images':len(eligible),
                     'complete_seed':seed_complete.get(seed,False),'original_seed_posthoc':seed==20261027,
                     'fixed_successes':sum(fixed[i]['success'] for i in available),'refresh_successes':sum(refresh[i]['success'] for i in available),
                     'fixed_only':b,'refresh_only':c,'fixed_asr':float(np.mean([fixed[i]['success'] for i in available])) if available else None,
                     'refresh_asr':float(np.mean([refresh[i]['success'] for i in available])) if available else None,
                     'difference':float(np.mean([float(refresh[i]['success'])-float(fixed[i]['success']) for i in available])) if available else None,
                     'exact_mcnemar_two_sided':exact_mcnemar(b,c) if seed_complete.get(seed,False) and population=='target_clean_ids' else None,
                     'p_interpretation':'uncorrected descriptive sensitivity; seed27 posthoc, new seeds prespecified; not applied to seed means'}
                for profile,mapping in [('fixed',fixed),('refresh',refresh)]:
                    for endpoint in ('adversarial_margin','adversarial_ce','adversarial_correct'):
                        row[f'{profile}_mean_{endpoint}']=float(np.mean([mapping[i][endpoint] for i in available])) if available else None
                per_seed.append(row)
            for name,seeds in [('two_new_seeds',NEW_SEEDS),('three_seeds_secondary',ALL_SEEDS)]:
                full=all(seed_complete.get(seed,False) for seed in seeds)
                for endpoint in ('success','adversarial_margin','adversarial_ce'):
                    result={'estimand':name,'seeds':list(seeds),'target':target,'population':population,'endpoint':endpoint,
                            'status':'complete' if full else 'missing_required_seed_images','expected_images':len(eligible),
                            'complete_seed_count':sum(seed_complete.get(seed,False) for seed in seeds),
                            'role':'primary replication' if name=='two_new_seeds' and population=='target_clean_ids' and endpoint=='success' else 'secondary/sensitivity',
                            'ci_condition':'fixed seed realizations, seed averaging within image before10000 image resamples; not seed-population variance',
                            'estimate':None,'ci':None}
                    if full:
                        differences={seed:{identity:float(data[seed]['refresh'][target][identity][endpoint])-float(data[seed]['fixed'][target][identity][endpoint]) for identity in eligible} for seed in seeds}
                        values=image_seed_mean(differences,seeds,eligible)
                        result['estimate']=float(values.mean());result['ci']=paired_bootstrap(values,seed=20261104)
                    estimands.append(result)
        for profile in ('fixed','refresh'):
            rows=[row for row in per_seed if row['target']==target and row['population']=='target_clean_ids' and row['complete_seed']]
            rates=[row[f'{profile}_asr'] for row in rows]
            ranges.append({'target':target,'profile':profile,'required_seed_count':3,'complete_seed_count':len(rates),
                           'seeds':[row['seed'] for row in rows],'ASR_min':min(rates) if rates else None,
                           'ASR_max':max(rates) if rates else None,'ASR_range':max(rates)-min(rates) if rates else None,
                           'interpretation':'observed fixed-seed range, not a confidence interval for seed population'})
        primary=next(row for row in estimands if row['target']==target and row['population']=='target_clean_ids' and row['estimand']=='two_new_seeds' and row['endpoint']=='success')
        new_rows=[next(row for row in per_seed if row['target']==target and row['population']=='target_clean_ids' and row['seed']==seed) for seed in NEW_SEEDS]
        label='incomplete two-new-seed replication'
        if primary['ci'] is not None:
            ci=primary['ci'];both_positive=all(row['difference']>0 for row in new_rows)
            if ci['ci95_high']<0:label='reverse direction under these conditions'
            elif both_positive and ci['ci95_low']>0:label='positive direction replicated under these fixed conditions'
            elif both_positive:label='direction agrees, magnitude uncertain'
            else:label='no consistent positive direction across the two new seeds'
        readings.append({'target':target,'description':label,'not_a_method_or_stopping_rule':True})
    return {'per_seed':per_seed,'estimands':estimands,'profile_ASR_ranges':ranges,'descriptive_readings':readings}


def collect_replication(directory):
    directory=Path(directory);plan=read_json(directory/'plan.json')
    for path,digest in plan['old_artifact_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Frozen old task artifact changed')
    if sha256(Path(plan['related_run'])/'plan.json')!=plan['related_plan_sha256'] or sha256(Path(plan['profile_run'])/'plan.json')!=plan['profile_plan_sha256']:
        raise ValueError('Old baseline plan identity changed')
    _old_plan,_old_table,old_pairs,old_blocks,old_attacks,old_evaluations=collect_profile(plan['profile_run'])
    if old_blocks!=[1,2,3,4]:raise ValueError('Original seed27 must retain all276 paired records')
    data={seed:{profile:{target:{} for target in plan['targets']} for profile in ('fixed','refresh')} for seed in ALL_SEEDS}
    pairs=[]
    for row in old_pairs:
        for profile in ('fixed','refresh'):
            data[20261027][profile][row['target']][row['id']]={key:row[f'{profile}_{key}'] for key in ('success','adversarial_correct','adversarial_margin','adversarial_ce')}
        pairs.append({'seed':20261027,'complete_unit':True,**row})
    reference=Path(plan['clean_reference']['path'])
    if sha256(reference/'config.json')!=plan['clean_reference']['config_sha256'] or sha256(reference/'progress.jsonl')!=plan['clean_reference']['progress_sha256']:
        raise ValueError('Frozen clean reference changed')
    clean={(r['model'],r['id']):r for r in read_jsonl(reference/'progress.jsonl')}
    if sha256(plan['manifest'])!=plan['manifest_sha256']:raise ValueError('Frozen manifest changed')
    records={r['id']:r for r in read_json(plan['manifest'])['records'] if r['id'] in plan['selected_ids']}
    attacks=[{**row,'seed':20261027,'historical_actual_hook_recorded':row.get('actual_forward_hook') is not None} for row in old_attacks]
    evaluations=[{**row,'seed':20261027} for row in old_evaluations]
    units=[];metadata=[];complete_units=[]
    for unit in plan['units']:
        mappings={};attack_maps={};component_full={}
        for profile in ('fixed','refresh'):
            attack_root=directory/'attacks'/unit['id']/profile;eval_root=directory/'evaluations'/unit['id']/profile
            a=read_jsonl(attack_root/'progress.jsonl');e=read_jsonl(eval_root/'progress.jsonl')
            if len({r['id'] for r in a})!=len(a) or any(r['id'] not in unit['selected_ids'] for r in a):raise ValueError('New attack IDs invalid')
            if len({(r['model'],r['id']) for r in e})!=len(e) or any(r['id'] not in unit['selected_ids'] or r['model'] not in plan['targets'] for r in e):raise ValueError('New evaluation IDs invalid')
            if (attack_root/'config.json').exists():
                cfg=read_json(attack_root/'config.json');args=cfg['arguments']
                expected={**plan['parameters'],'method':'ops' if profile=='fixed' else 'ops-refresh','seed':unit['seed'],
                          'profile':'ops-author-31ops-93pool' if profile=='fixed' else 'ops-author-noiserefresh'}
                if any(args.get(k)!=v for k,v in expected.items()) or cfg['selected_ids']!=unit['selected_ids'] or cfg['source_identity']!=plan['source_identity'] or cfg['manifest_sha256']!=plan['manifest_sha256'] or _hash_map(cfg['code_sha256'])!=_hash_map(plan['new_code_sha256']):
                    raise ValueError('New task configuration/model/code identity differs')
            if (eval_root/'config.json').exists():
                cfg=read_json(eval_root/'config.json')
                if cfg['selected_ids']!=unit['selected_ids'] or cfg['target_identities']!=plan['target_identities'] or _hash_map(cfg['code_sha256'])!=_hash_map(plan['new_code_sha256']):raise ValueError('New evaluation model/code identity differs')
            for row in a:
                record=records[row['id']]
                if row['image_seed']!=image_seed(unit['seed'],row['id']) or row['label']!=record['label'] or row['canonical_sha256']!=record['canonical_sha256'] or sha256(attack_root/'images'/f'{row["id"]}.png')!=row['png_sha256'] or sha256(record['canonical_path'])!=record['canonical_sha256']:
                    raise ValueError('New PNG/label/global seed identity differs')
                if row['actual_forward_hook']['forward_images']!=260 or row['forward_images']!=260 or row['input_gradient_images']!=260:raise ValueError('New source actual260 hook cost differs')
            for row in e:
                if row['clean_prediction']!=clean[(row['model'],row['id'])]['prediction'] or row['label']!=records[row['id']]['label']:raise ValueError('New clean argmax/label differs')
            mappings[profile]={target:{r['id']:r for r in e if r['model']==target} for target in plan['targets']};attack_maps[profile]={r['id']:r for r in a}
            component_full[profile]=len(a)==69 and all(len(mappings[profile][target])==69 for target in plan['targets'])
            attacks.extend({**r,'attack_profile':r.get('profile'),'profile':profile,'seed':unit['seed'],'unit':unit['id'],'historical_actual_hook_recorded':True} for r in a)
            evaluations.extend({**r,'profile':profile,'seed':unit['seed'],'unit':unit['id']} for r in e)
        full=all(component_full.values())
        if full:_check_unit_outputs(directory,unit);complete_units.append(unit['id'])
        units.append({**{key:value for key,value in unit.items() if key!='selected_ids'},'status':'complete' if full else 'partial_or_missing',
                      'fixed_attack_images':len(attack_maps['fixed']),'refresh_attack_images':len(attack_maps['refresh']),
                      'fixed_evaluation_images':sum(len(v) for v in mappings['fixed'].values()),'refresh_evaluation_images':sum(len(v) for v in mappings['refresh'].values())})
        for identity in unit['selected_ids']:
            path=directory/'cpu_identity'/f's{unit["seed"]}'/f'{identity}.json'
            if path.exists():metadata.append(read_json(path))
        for target in plan['targets']:
            for identity in unit['selected_ids']:
                fixed=mappings['fixed'][target].get(identity);refresh=mappings['refresh'][target].get(identity)
                if fixed is None or refresh is None:continue
                pairs.append({'seed':unit['seed'],'id':identity,'unit':unit['id'],'target':target,'complete_unit':full,
                    'target_clean_correct':clean[(target,identity)]['correct'],'source_clean_correct':clean[('rn50',identity)]['correct'],
                    'image_seed':image_seed(unit['seed'],identity),'canonical_sha256':records[identity]['canonical_sha256'],
                    'fixed_png_sha256':attack_maps['fixed'][identity]['png_sha256'],'refresh_png_sha256':attack_maps['refresh'][identity]['png_sha256'],
                    **{f'{name}_{key}':row[key] for name,row in [('fixed',fixed),('refresh',refresh)] for key in ('success','adversarial_correct','adversarial_margin','adversarial_ce')}})
                if full:
                    data[unit['seed']]['fixed'][target][identity]=fixed;data[unit['seed']]['refresh'][target][identity]=refresh
    seed_complete={20261027:True,**{seed:all(u['id'] in complete_units for u in plan['units'] if u['seed']==seed) for seed in NEW_SEEDS}}
    return plan,data,seed_complete,units,pairs,attacks,evaluations,metadata


def ops_seed_report_command(args):
    from .runner import _output
    source,output=Path(args.seed_run).resolve(),_output(args.output)
    plan,data,seed_complete,units,pairs,attacks,evaluations,metadata=collect_replication(source)
    analysis=summarize_seed_results(plan,data,seed_complete)
    costs=[]
    state_path=source/'state.json';state=read_json(state_path) if state_path.exists() else None
    for seed in ALL_SEEDS:
        for profile in ('fixed','refresh'):
            rows=[r for r in attacks if r['seed']==seed and r['profile']==profile]
            costs.append({'seed':seed,'profile':profile,'images':len(rows),'planned_images':276,
                'actual_source_forward_images':sum(r['forward_images'] for r in rows),'actual_source_gradient_images':sum(r['input_gradient_images'] for r in rows),
                'actual_target_forward_images':2*sum(r['seed']==seed and r['profile']==profile for r in evaluations),
                'actual_hook_recorded_images':sum(r.get('actual_forward_hook') is not None for r in rows),
                'attack_seconds':sum(r['elapsed_seconds'] for r in rows),'peak_cuda_allocated_bytes':max((r['peak_cuda_allocated_bytes'] for r in rows),default=0),
                **{key:_mean(rows,key) for key in PERTURBATION_KEYS}})
    summary={'status':'complete' if all(seed_complete.values()) else 'partial','seed_complete':seed_complete,'units':units,
        'protocol_sha256':plan['protocol_sha256'],'new_code_sha256':plan['new_code_sha256'],'source_archive':plan['source_archive'],
        'old_runner_sha256':plan['old_runner_sha256'],'new_runner_sha256':plan['new_runner_sha256'],'state':state,'costs':costs,
        'CPU_identity_images':len(metadata),'CPU_reconstruction_seconds':sum(r['reconstruction_seconds'] for r in metadata),
        'extra_metadata_forward_images':0,'extra_metadata_gradient_images':0,**analysis,
        'interpretation':'Exploratory prospective fixed-new-seed replication on previously used dev276; not independent-data confirmation or a new attack. '
            'Both profiles and both targets retained. Main estimate averages seed28/29 differences within image; seed27 was already observed. '
            'McNemar is exact two-sided binomial on each seed discordants only, uncorrected descriptive sensitivity. '
            'No seed-mean binomial test, discordant pooling across seeds, seed-population CI or equivalence inference. '
            'CPU identities are reconstructions, not fixed GPU trajectory capture. Instrumented timing is not pure refresh-cost causality.'}
    write_json(output/'report.json',summary)
    for name,rows in [('units',units),('per_seed',analysis['per_seed']),('estimands',analysis['estimands']),('profile_ASR_ranges',analysis['profile_ASR_ranges']),
                      ('paired_images',pairs),('attack_images',attacks),('evaluation_images',evaluations),('CPU_identity',metadata),('costs',costs)]:_csv(output/f'{name}.csv',rows)
    lines=['# OPS lifecycle seed replication','',f"Status:**{summary['status']}**. Complete seeds:{seed_complete}.",'',summary['interpretation'],'',
        '| Target | Seed | Population | Observed/expected | Fixed ASR | Refresh ASR | Difference pp | Fixed-only/refresh-only | Exact two-sided p |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for row in analysis['per_seed']:
        difference=None if row['difference'] is None else 100*row['difference']
        lines.append(f"| {row['target']} | {row['seed']}{' posthoc' if row['original_seed_posthoc'] else ''} | {row['population']} | {row['observed_images']}/{row['expected_images']} | {row['fixed_asr']} | {row['refresh_asr']} | {difference} | {row['fixed_only']}/{row['refresh_only']} | {row['exact_mcnemar_two_sided']} |")
    lines.extend(['','## Image-averaged estimands','',
        '| Target | Estimand | Population | Endpoint | Estimate |95% image CI | Status |','|---|---|---|---|---:|---|---|'])
    for row in analysis['estimands']:
        scale=100 if row['endpoint']=='success' else 1;ci=row['ci']
        interval=f"[{scale*ci['ci95_low']:.6g},{scale*ci['ci95_high']:.6g}]" if ci else 'withheld until both new seeds full276'
        estimate=None if row['estimate'] is None else scale*row['estimate']
        lines.append(f"| {row['target']} | {row['estimand']} | {row['population']} | {row['endpoint']} | {estimate} | {interval} | {row['status']} |")
    lines.extend(['']+[f"{r['target']}: {r['description']}." for r in analysis['descriptive_readings']]+['',
        f"CPU reconstruction:{len(metadata)} images,{summary['CPU_reconstruction_seconds']:.3f}s,0 extraF/B. Old fixed historical rows without hooks remain without hooks.",
        'All PNG perturbations, per-seed CE/margin, ASR ranges, cost and actual unit/task UTC/wall time are retained in CSV/JSON. Timing includes instrumentation and is not randomized.',''])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for axis,target in zip(axes,plan['targets']):
        rows=[r for r in analysis['per_seed'] if r['target']==target and r['population']=='target_clean_ids']
        for profile,color in [('fixed','#2864a0'),('refresh','#c65526')]:
            axis.plot(range(3),[100*r[f'{profile}_asr'] if r[f'{profile}_asr'] is not None else np.nan for r in rows],marker='o',label=profile,color=color)
        axis.set_xticks(range(3),['27(posthoc)','28','29']);axis.set_ylim(0,100);axis.set_ylabel('Target-own-clean ASR (%)');axis.set_title(target);axis.legend();axis.grid(alpha=.2)
    figure.suptitle(f'{summary["status"]}: fixed seeds, observed complete-unit images; main CI requires both new full276')
    figure.savefig(output/'ops_seed_replication.png',dpi=180);figure.savefig(output/'ops_seed_replication.pdf');plt.close(figure)
    return {'output':str(output),'status':summary['status'],'complete_seeds':seed_complete,'whole_population_main_CI':all(seed_complete.values())}
