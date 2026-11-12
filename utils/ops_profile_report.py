"""Both OPS lifecycle profiles, paired images and complete-population reporting."""
from pathlib import Path

import numpy as np

from .io import read_json, read_jsonl, sha256, write_json
from .matrix_report import PERTURBATION_KEYS, _csv, _mean
from .metrics import image_seed, paired_bootstrap
from .pool_diagnostic import _hash_map


def paired_population(pairs, eligible_ids, complete=False):
    selected=[row for row in pairs if row['id'] in set(eligible_ids)]
    fixed_only=sum(r['fixed_success'] and not r['refresh_success'] for r in selected)
    refresh_only=sum(r['refresh_success'] and not r['fixed_success'] for r in selected)
    differences=np.array([float(r['refresh_success'])-float(r['fixed_success']) for r in selected])
    result={'observed_images':len(selected),'expected_images':len(eligible_ids),
            'fixed_success_count':sum(r['fixed_success'] for r in selected),
            'refresh_success_count':sum(r['refresh_success'] for r in selected),
            'fixed_success_refresh_failure':fixed_only,'fixed_failure_refresh_success':refresh_only,
            'fixed_asr':_mean(selected,'fixed_success'),'refresh_asr':_mean(selected,'refresh_success'),
            'refresh_minus_fixed_asr':float(differences.mean()) if len(differences) else None,
            'ci':paired_bootstrap(differences,seed=20261103) if complete and len(selected)==len(eligible_ids) and len(selected) else None,
            'ci_condition':'single attack seed20261027 and realized per-image noise streams; image is statistical unit'}
    for endpoint in ('adversarial_margin','adversarial_ce'):
        result[f'fixed_mean_{endpoint}']=_mean(selected,f'fixed_{endpoint}')
        result[f'refresh_mean_{endpoint}']=_mean(selected,f'refresh_{endpoint}')
        result[f'refresh_minus_fixed_mean_{endpoint}']=(_mean(selected,f'refresh_{endpoint}')-_mean(selected,f'fixed_{endpoint}')) if selected else None
    return result


def collect_profile(directory):
    directory=Path(directory)
    plan=read_json(directory/'plan.json')
    old=read_json(Path(plan['related_run'])/'plan.json')
    if sha256(Path(plan['related_run'])/'plan.json')!=plan['related_plan_sha256']:
        raise ValueError('Original related plan changed')
    clean_path=Path(plan['clean_reference']['path'])
    if sha256(clean_path/'config.json')!=plan['clean_reference']['config_sha256'] or sha256(clean_path/'progress.jsonl')!=plan['clean_reference']['progress_sha256']:
        raise ValueError('Frozen clean reference changed')
    clean={(r['model'],r['id']):r for r in read_jsonl(clean_path/'progress.jsonl')}
    manifest=read_json(plan['manifest'])
    if sha256(plan['manifest'])!=plan['manifest_sha256']:
        raise ValueError('Manifest changed')
    records={r['id']:r for r in manifest['records'] if r['id'] in plan['selected_ids']}
    tasks={t['block']:t for t in old['tasks'] if t['configuration']=='ops260'}
    new_state=read_json(directory/'state.json') if (directory/'state.json').exists() else {'blocks':{}}
    old_state_path=Path(plan['related_run'])/'state.json'
    old_state=read_json(old_state_path) if old_state_path.exists() else {'tasks':{}}
    complete_blocks,table,all_pairs,raw_attacks,raw_evaluations=[],[],[],[],[]
    for block,ids in enumerate(plan['blocks'],1):
        mappings,attack_sets,attack_maps={},{},{}
        for profile,attack_root,eval_root in [
            ('fixed',Path(plan['related_run'])/'attacks'/tasks[block]['id'],Path(plan['related_run'])/'evaluations'/tasks[block]['id']),
            ('refresh',directory/'attacks'/f'b{block}',directory/'evaluations'/f'b{block}')]:
            attack_rows=read_jsonl(attack_root/'progress.jsonl');eval_rows=read_jsonl(eval_root/'progress.jsonl')
            if len({r['id'] for r in attack_rows})!=len(attack_rows) or any(r['id'] not in ids for r in attack_rows):
                raise ValueError('Lifecycle attack IDs invalid')
            if len({(r['model'],r['id']) for r in eval_rows})!=len(eval_rows) or any(r['id'] not in ids or r['model'] not in plan['targets'] for r in eval_rows):
                raise ValueError('Lifecycle evaluation IDs invalid')
            if (attack_root/'config.json').exists():
                cfg=read_json(attack_root/'config.json');arguments=cfg['arguments']
                expected={'method':'ops' if profile=='fixed' else 'ops-refresh','profile':'ops-author-31ops-93pool' if profile=='fixed' else 'ops-author-noiserefresh',
                          'source':'rn50','steps':10,'views':1,'ne':5,'np':5,'beta':2.,'epsilon_bytes':8.,'alpha_mult':1.,'seed':20261027,'view_batch':1}
                if any(arguments.get(k)!=v for k,v in expected.items()) or cfg['selected_ids']!=ids or cfg['manifest_sha256']!=plan['manifest_sha256'] or cfg['source_identity']!=plan['source_identity']:
                    raise ValueError('Lifecycle attack configuration/identity differs')
                expected_code=plan['old_code_sha256'] if profile=='fixed' else plan['new_code_sha256']
                if _hash_map(cfg['code_sha256'])!=_hash_map(expected_code):
                    raise ValueError('Lifecycle attack code identity differs')
            if (eval_root/'config.json').exists():
                cfg=read_json(eval_root/'config.json')
                expected_code=plan['old_code_sha256'] if profile=='fixed' else plan['new_code_sha256']
                if cfg['selected_ids']!=ids or cfg['manifest_sha256']!=plan['manifest_sha256'] or _hash_map(cfg['code_sha256'])!=_hash_map(expected_code) or cfg['target_identities']!=plan['target_identities']:
                    raise ValueError('Lifecycle evaluation identity differs')
            for row in attack_rows:
                if row['label']!=records[row['id']]['label'] or row['canonical_sha256']!=records[row['id']]['canonical_sha256'] or row['forward_images']!=260 or row['input_gradient_images']!=260 or row['image_seed']!=image_seed(20261027,row['id']):
                    raise ValueError('Lifecycle image/label/cost record differs')
                if sha256(attack_root/'images'/f'{row["id"]}.png')!=row['png_sha256'] or sha256(records[row['id']]['canonical_path'])!=row['canonical_sha256']:
                    raise ValueError('Lifecycle PNG/canonical content SHA mismatch')
                if profile=='refresh' and (row['noise_pool_generation_count']!=10 or row['noise_pool_steps_used']!=10 or row['actual_forward_hook']['forward_images']!=260):
                    raise ValueError('Refresh lifecycle/hook record differs from10 pools/260 images')
            for row in eval_rows:
                if row['label']!=records[row['id']]['label'] or row['clean_correct']!=clean[(row['model'],row['id'])]['correct']:
                    raise ValueError('Lifecycle evaluation label/clean denominator differs')
            mappings[profile]={target:{r['id']:r for r in eval_rows if r['model']==target} for target in plan['targets']}
            attack_sets[profile]={r['id'] for r in attack_rows}
            attack_maps[profile]={r['id']:r for r in attack_rows}
            raw_attacks.extend({**r,'attack_profile':r.get('profile'),'profile':profile,'block':block} for r in attack_rows)
            raw_evaluations.extend({**r,'profile':profile,'block':block} for r in eval_rows)
            for target in plan['targets']:
                rows=list(mappings[profile][target].values());own=[r for r in rows if r['clean_correct']]
                common=[r for r in own if clean[('rn50',r['id'])]['correct']]
                table.append({'profile':profile,'block':block,'target':target,'planned_images':69,'attack_images':len(attack_rows),'evaluated_images':len(rows),
                    'status':'complete' if len(attack_rows)==69 and all(len(mappings[profile][t])==69 for t in plan['targets']) else 'partial_or_missing',
                    'success_count':sum(r['success'] for r in own),'target_clean_denominator_observed':len(own),
                    'target_clean_denominator_expected':sum(clean[(target,identity)]['correct'] for identity in ids),
                    'asr_target_clean':_mean(own,'success'),'adv_accuracy_all':_mean(rows,'adversarial_correct'),
                    'source_target_clean_count':len(common),'asr_source_target_clean':_mean(common,'success'),
                    'mean_adversarial_margin':_mean(own,'adversarial_margin'),'mean_adversarial_ce':_mean(own,'adversarial_ce'),
                    'actual_forward_images':sum(r['forward_images'] for r in attack_rows),'actual_input_gradient_images':sum(r['input_gradient_images'] for r in attack_rows),
                    'target_forward_images':len(rows)*2,'attack_seconds':sum(r['elapsed_seconds'] for r in attack_rows),
                    'whole_block_wall_seconds':(old_state['tasks'].get(tasks[block]['id'],{}).get('total_attempt_wall_seconds') if profile=='fixed'
                        else new_state['blocks'].get(str(block),{}).get('total_attempt_wall_seconds')),
                    'noise_pool_generation_count':sum(r.get('noise_pool_generation_count',1) for r in attack_rows),
                    'noise_pool_generation_seconds':sum(r.get('noise_pool_generation_seconds',r.get('pool_init_seconds',0)) for r in attack_rows),
                    **{key:_mean(attack_rows,key) for key in PERTURBATION_KEYS}})
        full=all(attack_sets[profile]==set(ids) and all(set(mappings[profile][target])==set(ids) for target in plan['targets']) for profile in ('fixed','refresh'))
        if full:complete_blocks.append(block)
        for target in plan['targets']:
            for identity in ids:
                fixed,refresh=mappings['fixed'][target].get(identity),mappings['refresh'][target].get(identity)
                if fixed is None or refresh is None:
                    continue
                all_pairs.append({'id':identity,'block':block,'complete_predefined_block':full,'target':target,'label':records[identity]['label'],
                    'target_clean_correct':clean[(target,identity)]['correct'],'source_clean_correct':clean[('rn50',identity)]['correct'],
                    'canonical_sha256':records[identity]['canonical_sha256'],'image_seed':image_seed(20261027,identity),
                    'fixed_png_sha256':attack_maps['fixed'][identity]['png_sha256'],'refresh_png_sha256':attack_maps['refresh'][identity]['png_sha256'],
                    **{f'{name}_{key}':row[key] for name,row in [('fixed',fixed),('refresh',refresh)]
                       for key in ('success','adversarial_correct','adversarial_prediction','adversarial_margin','adversarial_ce','clean_margin','clean_ce')}})
    return plan,table,all_pairs,complete_blocks,raw_attacks,raw_evaluations


def ops_profile_report_command(args):
    from .runner import _output
    source,output=Path(args.profile_run).resolve(),_output(args.output)
    plan,table,pairs,blocks,attacks,evaluations=collect_profile(source)
    complete=len(blocks)==4
    populations={}
    for target in plan['targets']:
        available=[r for r in pairs if r['target']==target and r['complete_predefined_block']]
        populations[target]={name:paired_population(available,ids,complete) for name,ids in plan['populations'][target].items()}
        all_pair_values=[r for r in available]
        populations[target]['all276_adversarial_accuracy']={
            'observed_images':len(all_pair_values),'expected_images':276,
            'fixed':_mean(all_pair_values,'fixed_adversarial_correct'),'refresh':_mean(all_pair_values,'refresh_adversarial_correct'),
            'refresh_minus_fixed':float(np.mean([float(r['refresh_adversarial_correct'])-float(r['fixed_adversarial_correct']) for r in all_pair_values])) if all_pair_values else None}
    state_path=source/'state.json'
    report={'status':'complete' if complete else 'partial','complete_predefined_blocks':blocks,'planned_blocks':4,
            'protocol_sha256':plan['protocol_sha256'],'plan_sha256':sha256(source/'plan.json'),
            'new_code_sha256':plan['new_code_sha256'],'old_code_sha256':plan['old_code_sha256'],'source_archive':plan['source_archive'],
            'source_identity':plan['source_identity'],'target_identities':plan['target_identities'],
            'state':read_json(state_path) if state_path.exists() else None,'populations':populations,'block_results':table,
            'actual_new_forward_images':sum(r['forward_images'] for r in attacks if r['profile']=='refresh'),
            'actual_new_input_gradient_images':sum(r['input_gradient_images'] for r in attacks if r['profile']=='refresh'),
            'actual_new_target_forward_images':2*sum(r['profile']=='refresh' for r in evaluations),
            'planned_new_forward_images':71760,'planned_new_input_gradient_images':71760,'planned_new_target_forward_images':1104,
            'interpretation':'Both author-code fixed-pool and author-noiserefresh lifecycle profiles are retained. Exploratory, one attack seed, local224/EPS8/true-label protocol. '
                'Paper K_list1/2/3/4 is not ported: both use author31×levels2/3/4=93 composites. '
                'No whole-population CI before all276 complete; partial pairs are preserved and complete blocks described. '
                'Not a new method, numerical paper replication, SOTA, random-stability or transfer-causality claim.'}
    write_json(output/'report.json',report)
    _csv(output/'blocks.csv',table);_csv(output/'paired_images.csv',pairs)
    _csv(output/'attack_images.csv',attacks);_csv(output/'evaluation_images.csv',evaluations)
    summary_rows=[]
    for target,values in populations.items():
        for population in ('target_clean_ids','source_target_clean_ids'):
            summary_rows.append({'target':target,'population':population,**values[population]})
    _csv(output/'paired_summary.csv',summary_rows)
    lines=['# OPS noise lifecycle profiles','',f"Status:**{report['status']}**; complete predefined blocks:{blocks}/[1,2,3,4].",'',report['interpretation'],'',
        '| Target | Population | Observed/expected | Fixed ASR | Refresh ASR | Difference pp[95% CI] | Fixed-only / refresh-only successes |',
        '|---|---|---:|---:|---:|---|---:|']
    for row in summary_rows:
        ci=row['ci']; effect=row['refresh_minus_fixed_asr']
        value='missing' if effect is None else f'{100*effect:.4f}'
        if ci:value+=f" [{100*ci['ci95_low']:.4f},{100*ci['ci95_high']:.4f}]"
        else:value+=' [CI withheld until complete276]'
        lines.append(f"| {row['target']} | {row['population']} | {row['observed_images']}/{row['expected_images']} | {row['fixed_asr']} | {row['refresh_asr']} | {value} | {row['fixed_success_refresh_failure']}/{row['fixed_failure_refresh_success']} |")
    lines.extend(['','## Complete and partial block descriptions','',
        '| Profile | Block | Target | Images | ASR successes/denominator | Adv accuracy | Margin | CE | Source F/B | Attack seconds | PNG mean abs bytes | PNG L2 | Boundary fraction |',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|'])
    for row in table:
        lines.append(f"| {row['profile']} | {row['block']} | {row['target']} | {row['evaluated_images']}/69 | {row['success_count']}/{row['target_clean_denominator_observed']} | {row['adv_accuracy_all']} | {row['mean_adversarial_margin']} | {row['mean_adversarial_ce']} | {row['actual_forward_images']}/{row['actual_input_gradient_images']} | {row['attack_seconds']:.3f} | {row['png_mean_abs_bytes']} | {row['png_l2_unit']} | {row['png_boundary_element_fraction']} |")
    lines.extend(['','Partial image pairs remain in paired_images.csv but are not silently promoted to whole276 inference. '
        'Discordants are fixed-success/refresh-failure and fixed-failure/refresh-success. Margin/CE and both profiles remain reported regardless of ASR direction.',''])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for axis,target in zip(axes,plan['targets']):
        value=populations[target]['target_clean_ids']
        for index,name in enumerate(('fixed_asr','refresh_asr')):
            if value[name] is not None:axis.bar(index,100*value[name],color=('#2864a0','#c65526')[index])
        axis.set_xticks([0,1],['fixed-code','noise-refresh']);axis.set_ylim(0,100)
        axis.set_ylabel('Target-own-clean ASR (%)');axis.set_title(f'{target};{value["observed_images"]}/{value["expected_images"]}')
    figure.suptitle(f'{report["status"]}: both lifecycle profiles, complete blocks{blocks}; single-seed exploratory')
    figure.savefig(output/'ops_lifecycle.png',dpi=180);figure.savefig(output/'ops_lifecycle.pdf');plt.close(figure)
    return {'output':str(output),'status':report['status'],'complete_predefined_blocks':blocks,'whole_population_ci_produced':complete}
