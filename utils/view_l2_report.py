"""Known-principle control and fixed-source-point descriptive reports."""
from pathlib import Path

import numpy as np

from .io import read_json,read_jsonl,sha256,write_json
from .matrix_report import _csv,_mean,PERTURBATION_KEYS
from .metrics import paired_bootstrap,image_seed
from .ops_seed_report import exact_mcnemar
from .pool_diagnostic import _hash_map


def population_difference(pairs,ids,full):
    wanted=set(ids);rows=[r for r in pairs if r['id'] in wanted]
    raw_only=sum(r['raw_success'] and not r['control_success'] for r in rows)
    control_only=sum(r['control_success'] and not r['raw_success'] for r in rows)
    result={'observed_images':len(rows),'expected_images':len(ids),'raw_successes':sum(r['raw_success'] for r in rows),
            'control_successes':sum(r['control_success'] for r in rows),'raw_only':raw_only,'control_only':control_only,
            'raw_asr':_mean(rows,'raw_success'),'control_asr':_mean(rows,'control_success'),
            'exact_mcnemar_two_sided':exact_mcnemar(raw_only,control_only) if full and len(rows)==len(ids) else None,
            'p_interpretation':'single-seed exploratory, uncorrected descriptive sensitivity; not a stable-benefit/innovation criterion',
            'endpoints':{}}
    for endpoint in ('success','adversarial_margin','adversarial_ce','adversarial_correct'):
        values=np.array([float(r[f'control_{endpoint}'])-float(r[f'raw_{endpoint}']) for r in rows])
        result['endpoints'][endpoint]={'raw_mean':_mean(rows,f'raw_{endpoint}'),'control_mean':_mean(rows,f'control_{endpoint}'),
            'difference':float(values.mean()) if len(values) else None,
            'ci':paired_bootstrap(values,seed=20261110) if full and len(rows)==len(ids) and len(rows) else None}
    return result


def _plot_setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def source_report(source,output,config,rows):
    views=[];states=[]
    for row in rows:
        views.extend({'id':row['id'],**view} for view in row['views'])
        states.extend({'id':row['id'],'state':name,**value} for name,value in row['states'].items())
    report={'status':'complete' if len(rows)==24 else 'partial','count':len(rows),'planned_images':24,
            'actual_forward_images':sum(r['forward_images'] for r in rows),'actual_input_gradient_images':sum(r['input_gradient_images'] for r in rows),
            'target_forward_images':0,'protocol_sha256':config['protocol_sha256'],'per_image_states':states,
            'interpretation':'Independent namespace20261109; shared complete plans at clean and old raw-fresh PNG fixed points. '
            'Norm participation ratio is not independent effective sample size. Base norm share is not directional contribution. '
            'CE/norm observations do not separate Jacobian and confidence causes, and do not describe intermediate attack trajectories. No expansion/effect threshold.'}
    write_json(output/'report.json',report);_csv(output/'states.csv',states);_csv(output/'views.csv',views)
    lines=['# Fixed-source-point norm probe','',f"Status:{report['status']}; {len(rows)}/24 images; {report['actual_forward_images']}/1248 sourceF/B;0 targetF.",'',report['interpretation'],'',
           '| Image | State | P | Base norm share | Base tied max | Max norm share | Min norm | Zero views | Tiny<1e-12 | Raw/unit cos |',
           '|---|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for row in states:
        lines.append(f"| {row['id']} | {row['state']} | {row['norm_participation_ratio']} | {row['base_norm_share']} | {row['base_is_tied_max']} | {row['max_norm_share']} | {row['min_view_l2']} | {row['zero_view_count']} | {row['view_l2_below_1e_minus_12_count']} | {row['raw_unit_cosine']} |")
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plt=_plot_setup();figure,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for name,color in [('clean','#2864a0'),('old_raw_png','#c65526')]:
        selected=[r for r in states if r['state']==name]
        axes[0].plot(range(len(selected)),[r['norm_participation_ratio'] for r in selected],'.',label=name,color=color)
        axes[1].plot(range(len(selected)),[r['base_norm_share'] for r in selected],'.',label=name,color=color)
    axes[0].set_ylabel('Norm participation ratio P');axes[1].set_ylabel('Base-view norm share')
    for axis in axes:axis.set_xlabel('Fixed manifest image order');axis.grid(alpha=.2)
    handles,labels=axes[0].get_legend_handles_labels()
    figure.legend(handles,labels,loc='outside lower center',ncol=2)
    figure.suptitle('Source-only24 fixed points; descriptive, not trajectories or effective sample size')
    figure.savefig(output/'source_norm_probe.png',dpi=180);figure.savefig(output/'source_norm_probe.pdf');plt.close(figure)
    return {'output':str(output),'status':report['status'],'target_forward_images':0}


def collect_control(source,config):
    old=Path(config['profile_run']);raw_config=read_json(old/'plan.json')
    if sha256(old/'plan.json')!=config['old_profile_plan_sha256']:raise ValueError('Old raw plan changed')
    for path,digest in config['old_artifact_sha256'].items():
        if sha256(path)!=digest:raise ValueError('Old raw config/progress artifact changed')
    if sha256(config['manifest'])!=config['manifest_sha256']:raise ValueError('Manifest changed')
    record_map={r['id']:r for r in read_json(config['manifest'])['records'] if r['id'] in config['selected_ids']}
    clean_root=Path(config['clean_reference']['path'])
    if sha256(clean_root/'config.json')!=config['clean_reference']['config_sha256'] or sha256(clean_root/'progress.jsonl')!=config['clean_reference']['progress_sha256']:
        raise ValueError('Frozen clean reference changed')
    clean={(r['model'],r['id']):r for r in read_jsonl(clean_root/'progress.jsonl')}
    attacks=[];evaluations=[];pairs=[];complete_blocks=[];table=[];steps=[]
    for block,ids in enumerate(config['blocks'],1):
        mappings={};attack_maps={}
        for name,root in [('raw',old),('control',source)]:
            attack_root=root/'attacks'/f'b{block}';eval_root=root/'evaluations'/f'b{block}'
            a=read_jsonl(attack_root/'progress.jsonl');e=read_jsonl(eval_root/'progress.jsonl')
            if len({r['id'] for r in a})!=len(a) or any(r['id'] not in ids for r in a):raise ValueError('Attack IDs invalid')
            if len({(r['model'],r['id']) for r in e})!=len(e) or any(r['id'] not in ids or r['model'] not in config['target_identities'] for r in e):raise ValueError('Evaluation IDs invalid')
            if (attack_root/'config.json').exists():
                saved=read_json(attack_root/'config.json');args=saved['arguments']
                method='ops-refresh' if name=='raw' else 'ops-view-l2';profile='ops-author-noiserefresh' if name=='raw' else config['profile']
                wanted={'method':method,'profile':profile,'source':'rn50','steps':10,'views':1,'ne':5,'np':5,'beta':2.,'epsilon_bytes':8.,'alpha_mult':1.,'seed':20261027,'view_batch':1}
                expected_code=config['old_code_sha256'] if name=='raw' else config['new_code_sha256']
                if any(args.get(key)!=value for key,value in wanted.items()) or saved['selected_ids']!=ids or saved['manifest_sha256']!=config['manifest_sha256'] or saved['source_identity']!=config['source_identity'] or _hash_map(saved['code_sha256'])!=_hash_map(expected_code):
                    raise ValueError('Raw/control attack identity differs')
            if (eval_root/'config.json').exists():
                saved=read_json(eval_root/'config.json');expected_code=config['old_code_sha256'] if name=='raw' else config['new_code_sha256']
                if saved['selected_ids']!=ids or saved['manifest_sha256']!=config['manifest_sha256'] or saved['target_identities']!=config['target_identities'] or _hash_map(saved['code_sha256'])!=_hash_map(expected_code):raise ValueError('Evaluation identity differs')
            for row in a:
                record=record_map[row['id']]
                if row['label']!=record['label'] or row['image_seed']!=image_seed(20261027,row['id']) or row['canonical_sha256']!=record['canonical_sha256'] or row['forward_images']!=260 or row['input_gradient_images']!=260:raise ValueError('Image/cost identity differs')
                if sha256(record['canonical_path'])!=record['canonical_sha256'] or sha256(attack_root/'images'/f'{row["id"]}.png')!=row['png_sha256']:raise ValueError('PNG/canonical content changed')
                if name=='control':
                    if row['actual_forward_hook']['forward_images']!=260 or len(row['steps_norm_summary'])!=10:raise ValueError('Control actual hook/step count differs')
                    steps.extend({'id':row['id'],'block':block,**step} for step in row['steps_norm_summary'])
            for row in e:
                if row['clean_prediction']!=clean[(row['model'],row['id'])]['prediction'] or row['label']!=record_map[row['id']]['label']:raise ValueError('Clean argmax/label identity differs')
            mappings[name]={target:{r['id']:r for r in e if r['model']==target} for target in config['target_identities']}
            attack_maps[name]={r['id']:r for r in a}
            attacks.extend({**r,'method_profile':r.get('profile'),'group':name,'block':block} for r in a)
            evaluations.extend({**r,'group':name,'block':block} for r in e)
            for target in config['target_identities']:
                target_rows=list(mappings[name][target].values());own=[r for r in target_rows if clean[(target,r['id'])]['correct']]
                table.append({'group':name,'block':block,'target':target,'attack_images':len(a),'evaluation_images':len(target_rows),'planned_images':69,
                    'successes':sum(r['success'] for r in own),'target_clean_denominator':len(own),'asr':_mean(own,'success'),
                    'all_adversarial_accuracy':_mean(target_rows,'adversarial_correct'),'mean_adversarial_ce':_mean(own,'adversarial_ce'),
                    'mean_adversarial_margin':_mean(own,'adversarial_margin'),'forward_images':sum(r['forward_images'] for r in a),
                    'input_gradient_images':sum(r['input_gradient_images'] for r in a),'target_forward_images':2*len(target_rows),
                    'attack_seconds':sum(r['elapsed_seconds'] for r in a),'peak_cuda_allocated_bytes':max((r['peak_cuda_allocated_bytes'] for r in a),default=0),
                    **{key:_mean(a,key) for key in PERTURBATION_KEYS}})
        full=all(set(attack_maps[name])==set(ids) and all(set(mappings[name][target])==set(ids) for target in config['target_identities']) for name in ('raw','control'))
        if full:complete_blocks.append(block)
        for target in config['target_identities']:
            for identity in ids:
                raw,controlled=mappings['raw'][target].get(identity),mappings['control'][target].get(identity)
                if raw is None or controlled is None:continue
                if attack_maps['control'][identity]['noise_pool_sha256_by_pool']!=attack_maps['raw'][identity]['noise_pool_sha256_by_pool']:raise ValueError('Actual paired sampler10-pool SHA differs')
                pairs.append({'id':identity,'target':target,'block':block,'complete_block':full,
                    'canonical_sha256':record_map[identity]['canonical_sha256'],'image_seed':image_seed(20261027,identity),
                    'raw_png_sha256':attack_maps['raw'][identity]['png_sha256'],'control_png_sha256':attack_maps['control'][identity]['png_sha256'],
                    **{f'{group}_{key}':row[key] for group,row in [('raw',raw),('control',controlled)] for key in ('success','adversarial_correct','adversarial_margin','adversarial_ce')}})
    return table,pairs,complete_blocks,attacks,evaluations,steps


def report_command(args):
    from .runner import _output
    source,output=Path(args.run).resolve(),_output(args.output)
    config=read_json(source/'config.json');rows=read_jsonl(source/'progress.jsonl')
    if config['mode']=='source24_fixed_points':return source_report(source,output,config,rows)
    if config['mode']!='formal276':
        summary={'mode':config['mode'],'engineering_only':True,'count':len(rows),'whole_population_main_CI':None,'rows':rows}
        write_json(output/'report.json',summary);_csv(output/'engineering_images.csv',rows)
        return {'output':str(output),'engineering_only':True,'count':len(rows)}
    table,pairs,blocks,attacks,evaluations,steps=collect_control(source,config)
    full=len(blocks)==4;populations={}
    for target in config['target_identities']:
        available=[r for r in pairs if r['target']==target and r['complete_block']]
        populations[target]={name:population_difference(available,ids,full) for name,ids in [*config['populations'][target].items(),('all276',config['selected_ids'])]}
    summary={'status':'complete' if full else 'partial','complete_blocks':blocks,'planned_blocks':4,'populations':populations,'blocks':table,
        'source_forward_images':sum(r['forward_images'] for r in attacks if r['group']=='control'),
        'source_gradient_images':sum(r['input_gradient_images'] for r in attacks if r['group']=='control'),
        'target_forward_images':2*sum(r['group']=='control' for r in evaluations),'planned_source_forward_images':71760,'planned_target_forward_images':1104,
        'protocol_sha256':config['protocol_sha256'],'kernel_sha256':config['kernel_sha256'],'old_code_sha256':config['old_code_sha256'],
        'new_code_sha256':config['new_code_sha256'],'source_archive':config['source_archive'],
        'state':read_json(source/'state.json') if (source/'state.json').exists() else None,
        'interpretation':'Known-principle per-view L2 control, not a new method or complete SimAttack replication. '
        'Compare only raw-fresh seed20261027. Both targets retained; single-seed previously used development set. '
        'All276 must complete before whole-population CI. No variance reduction/unbiased-gradient/causal or stable-benefit claim.'}
    write_json(output/'report.json',summary)
    for name,values in [('blocks',table),('paired_images',pairs),('attack_images',attacks),('evaluation_images',evaluations),('step_norm_summary',steps)]:_csv(output/f'{name}.csv',values)
    stats=[]
    for target,values in populations.items():
        for population,value in values.items():
            for endpoint,estimate in value['endpoints'].items():stats.append({'target':target,'population':population,'endpoint':endpoint,'observed_images':value['observed_images'],
                'expected_images':value['expected_images'],'raw_only':value['raw_only'],'control_only':value['control_only'],'exact_mcnemar_two_sided':value['exact_mcnemar_two_sided'],**estimate})
    _csv(output/'paired_summary.csv',stats)
    lines=['# Known-principle view-L2 control','',f"Status:{summary['status']}; complete blocks:{blocks}.",'',summary['interpretation'],'',
        '| Target | Population | Endpoint | Observed/expected | Control−raw |95% image CI | Raw-only/control-only | Exact McNemar p |',
        '|---|---|---|---:|---:|---|---:|---:|']
    for row in stats:
        ci=row['ci'];interval=f"[{ci['ci95_low']:.6g},{ci['ci95_high']:.6g}]" if ci else 'null until complete276'
        lines.append(f"| {row['target']} | {row['population']} | {row['endpoint']} | {row['observed_images']}/{row['expected_images']} | {row['difference']} | {interval} | {row['raw_only']}/{row['control_only']} | {row['exact_mcnemar_two_sided']} |")
    lines.extend(['','PNG perturbation/cost/time and every generated image, partial pair and step norm summary are retained in CSV/JSON. Engineering/source24 do not enter this table.',''])
    (output/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    plt=_plot_setup();figure,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for axis,target in zip(axes,config['target_identities']):
        value=populations[target]['target_clean_ids']
        for index,key in enumerate(('raw_asr','control_asr')):
            if value[key] is not None:axis.bar(index,100*value[key],color=('#2864a0','#c65526')[index])
        axis.set_xticks([0,1],['raw fresh27','view L2 control27']);axis.set_ylim(0,100);axis.set_ylabel('Target-own-clean ASR (%)');axis.set_title(target)
    figure.suptitle(f"{summary['status']}: complete blocks{blocks}, fixed-seed exploratory control")
    figure.savefig(output/'view_l2_control.png',dpi=180);figure.savefig(output/'view_l2_control.pdf');plt.close(figure)
    return {'output':str(output),'status':summary['status'],'whole_population_main_CI_produced':full}
