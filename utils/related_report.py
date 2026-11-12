"""Recent baseline report with strictly predefined complete-block comparisons."""
from pathlib import Path

import numpy as np

from .io import read_json, read_jsonl, sha256, write_json
from .matrix_report import PERTURBATION_KEYS, _csv, _mean
from .metrics import paired_bootstrap


def collect_related(directory):
    directory = Path(directory)
    plan = read_json(directory / 'plan.json')
    state_path = directory / 'state.json'
    state = read_json(state_path) if state_path.exists() else {'tasks': {}}
    clean_path = Path(plan['clean_reference']['path'])
    if sha256(clean_path / 'config.json') != plan['clean_reference']['config_sha256'] or sha256(clean_path / 'progress.jsonl') != plan['clean_reference']['progress_sha256']:
        raise ValueError('Frozen external clean reference changed')
    clean_config = read_json(clean_path / 'config.json')
    clean = {(r['model'], r['id']): r for r in read_jsonl(clean_path / 'progress.jsonl')}
    table, attacks_all, evaluations_all, mappings, complete = [], [], [], {}, {}
    for task in plan['tasks']:
        attack_path = directory / 'attacks' / task['id']
        eval_path = directory / 'evaluations' / task['id']
        attacks, evaluations = read_jsonl(attack_path / 'progress.jsonl'), read_jsonl(eval_path / 'progress.jsonl')
        ids = set(task['selected_ids'])
        if len({r['id'] for r in attacks}) != len(attacks) or any(r['id'] not in ids for r in attacks):
            raise ValueError(f'Attack IDs invalid for {task["id"]}')
        if len({(r['model'], r['id']) for r in evaluations}) != len(evaluations) or any(r['id'] not in ids or r['model'] not in plan['targets'] for r in evaluations):
            raise ValueError(f'Evaluation IDs invalid for {task["id"]}')
        if (attack_path / 'config.json').exists():
            cfg = read_json(attack_path / 'config.json')
            arguments = cfg['arguments']
            expected = {'method': task['method'], 'source': 'rn50', 'steps': task['steps'], 'views': task['views'],
                        'epsilon_bytes': 8., 'alpha_mult': 1., 'seed': plan['seed'], 'view_batch': 1,
                        'ne': task.get('ne', 5), 'np': task.get('np', 5), 'beta': task.get('beta', 2.), 'profile': task['profile']}
            if cfg['selected_ids'] != task['selected_ids'] or cfg['manifest_sha256'] != plan['manifest_sha256'] or cfg['code_sha256'] != plan['code_sha256']:
                raise ValueError(f'Attack sample/version mismatch: {task["id"]}')
            if any(arguments.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Attack configuration mismatch: {task["id"]}')
            if cfg['source_identity'] != clean_config['model_identities']['rn50']:
                raise ValueError('Source weights differ from frozen clean reference')
        if (eval_path / 'config.json').exists():
            cfg = read_json(eval_path / 'config.json')
            if cfg['selected_ids'] != task['selected_ids'] or cfg['code_sha256'] != plan['code_sha256'] or cfg['manifest_sha256'] != plan['manifest_sha256']:
                raise ValueError(f'Evaluation sample/version mismatch: {task["id"]}')
            if any(cfg['target_identities'][target] != clean_config['model_identities'][target] for target in plan['targets']):
                raise ValueError('Target weights differ from frozen clean reference')
        attacks_all.extend({'task_id': task['id'], 'configuration': task['configuration'], **row} for row in attacks)
        evaluations_all.extend({'task_id': task['id'], 'configuration': task['configuration'], **row} for row in evaluations)
        mappings[task['id']] = {target: {r['id']: r for r in evaluations if r['model'] == target} for target in plan['targets']}
        complete[task['id']] = len(attacks) == 69 and all(set(mappings[task['id']][t]) == ids for t in plan['targets'])
        for target in plan['targets']:
            rows = list(mappings[task['id']][target].values())
            for row in rows:
                if row['clean_correct'] != clean[(target, row['id'])]['correct']:
                    raise ValueError('Clean denominator differs from frozen reference')
            own = [r for r in rows if clean[(target, r['id'])]['correct']]
            common = [r for r in own if clean[('rn50', r['id'])]['correct']]
            entry = {**{k: v for k, v in task.items() if k != 'selected_ids'}, 'target': target,
                     'status': 'complete' if complete[task['id']] else 'partial_or_missing',
                     'attack_images': len(attacks), 'evaluated_images': len(rows), 'planned_images': 69,
                     'target_clean_successes': sum(r['success'] for r in own), 'target_clean_denominator_observed': len(own),
                     'target_clean_denominator_expected': sum(clean[(target, identity)]['correct'] for identity in task['selected_ids']),
                     'asr_target_clean': _mean(own, 'success'), 'adv_accuracy_all': _mean(rows, 'adversarial_correct'),
                     'all_adv_correct': sum(r['adversarial_correct'] for r in rows),
                     'source_target_clean_successes': sum(r['success'] for r in common),
                     'source_target_clean_denominator': len(common), 'asr_source_target_clean': _mean(common, 'success'),
                     'mean_adv_margin_target_clean': _mean(own, 'adversarial_margin'),
                     'mean_adv_ce_target_clean': _mean(own, 'adversarial_ce'),
                     'actual_gradient_images': sum(r['input_gradient_images'] for r in attacks),
                     'actual_forward_images': sum(r['forward_images'] for r in attacks),
                     'attack_seconds': sum(r['elapsed_seconds'] for r in attacks),
                     'group_wall_seconds': state['tasks'].get(task['id'], {}).get('total_attempt_wall_seconds'),
                     'zero_grad_count': sum(r['zero_grad_count'] for r in attacks),
                     **{key: _mean(attacks, key) for key in PERTURBATION_KEYS}}
            table.append(entry)
    return plan, clean, table, mappings, complete, attacks_all, evaluations_all


def related_analysis(plan, clean, mappings, complete):
    tasks = {(t['configuration'], t['block']): t for t in plan['tasks']}
    matched_blocks = [block for block in range(1, 5)
                      if all(complete.get(tasks[(name, block)]['id'], False) for name in ('ops260', 'din260', 'sia260', 'bsr260'))]
    ids = [identity for block in matched_blocks for identity in plan['blocks'][block - 1]]
    rows, contrasts = [], []
    for target in plan['targets']:
        for population in ('target-clean', 'source-target-clean'):
            eligible = [identity for identity in ids if clean[(target, identity)]['correct'] and
                        (population != 'source-target-clean' or clean[('rn50', identity)]['correct'])]
            endpoints = {}
            for name in ('ops260', 'din260', 'sia260', 'bsr260'):
                mapping = {identity: row for block in matched_blocks
                           for identity, row in mappings[tasks[(name, block)]['id']][target].items()}
                endpoints[name] = mapping
                selected = [mapping[identity] for identity in eligible]
                rows.append({'analysis': 'S3_matched260', 'configuration': name, 'target': target, 'population': population,
                             'complete_blocks': matched_blocks, 'all_images': len(ids), 'eligible_images': len(eligible),
                             'successes': sum(r['success'] for r in selected), 'asr': _mean(selected, 'success'),
                             'adv_accuracy_all': _mean(list(mapping.values()), 'adversarial_correct'),
                             'mean_adv_margin': _mean(selected, 'adversarial_margin'),
                             'status': 'complete' if len(matched_blocks) == 4 else 'complete_predefined_blocks_only' if matched_blocks else 'missing'})
            for control in ('din260', 'sia260', 'bsr260'):
                for endpoint in ('success', 'adversarial_margin'):
                    result = {'comparison': f'ops260_minus_{control}', 'target': target, 'population': population,
                              'endpoint': endpoint, 'complete_blocks': matched_blocks, 'all_images': len(ids),
                              'eligible_images': len(eligible), 'seed': plan['seed'], 'seed_count': 1,
                              'status': 'complete_predefined_blocks_only' if eligible else 'missing',
                              'matrix_complete': len(matched_blocks) == 4,
                              'ci_condition': 'exploratory, conditional on single fixed seed and included complete predefined blocks'}
                    if eligible:
                        difference = np.array([float(endpoints['ops260'][identity][endpoint]) -
                                               float(endpoints[control][identity][endpoint]) for identity in eligible])
                        result.update(paired_bootstrap(difference))
                    contrasts.append(result)
    # S2 default cost points are shown separately; sets need not be identical and
    # these summaries are never used as cost-matched effect contrasts.
    for name in ('ops260', 'bsr200'):
        blocks = [block for block in range(1, 5) if complete.get(tasks[(name, block)]['id'], False)]
        for target in plan['targets']:
            mapping = {identity: row for block in blocks for identity, row in mappings[tasks[(name, block)]['id']][target].items()}
            own = [r for r in mapping.values() if clean[(target, r['id'])]['correct']]
            rows.append({'analysis': 'S2_default_cost_point', 'configuration': name, 'target': target,
                         'gradient_images_per_image': tasks[(name, 1)]['gradient_images_per_image'],
                         'population': 'target-clean', 'complete_blocks': blocks, 'all_images': len(mapping),
                         'eligible_images': len(own), 'successes': sum(r['success'] for r in own),
                         'asr': _mean(own, 'success'), 'adv_accuracy_all': _mean(list(mapping.values()), 'adversarial_correct'),
                         'mean_adv_margin': _mean(own, 'adversarial_margin'),
                         'status': 'complete' if len(blocks) == 4 else 'complete_predefined_blocks_only' if blocks else 'missing'})
    return rows, contrasts, matched_blocks


def related_report_command(args):
    from .runner import _output
    output = _output(args.output)
    plan, clean, table, mappings, complete, attacks, evaluations = collect_related(args.related_run)
    summaries, contrasts, matched = related_analysis(plan, clean, mappings, complete)
    report = {'status': 'complete' if all(complete.values()) else 'partial', 'complete_tasks': sum(complete.values()),
              'planned_tasks': len(plan['tasks']), 'matched_S3_blocks': matched,
              'planned_gradient_images': plan['planned_gradient_images'],
              'actual_gradient_images': sum(r['input_gradient_images'] for r in attacks),
              'protocol_sha256': plan['protocol_sha256'], 'upstream_versions': plan['upstream_versions'],
              'task_results': table, 'block_summaries': summaries, 'paired_contrasts': contrasts,
              'interpretation': 'Single-seed exploratory intervals conditional on complete predefined blocks. '
                                'No claim of random stability, SOTA or architecture-family generalization. '
                                'CI crossing zero is not equivalence. S2 default cost points are separate from S3 matched260 comparisons.'}
    write_json(output / 'report.json', report)
    _csv(output / 'tasks.csv', table)
    _csv(output / 'block_summaries.csv', summaries)
    _csv(output / 'contrasts.csv', contrasts)
    _csv(output / 'attack_images.csv', attacks)
    _csv(output / 'evaluation_images.csv', evaluations)
    _csv(output / 'clean_reference.csv', [r for (_, identity), r in clean.items() if identity in set(plan['selected_ids'])])
    lines = ['# Recent baseline anchors', '', f"Status: **{report['status']}**. Complete tasks {sum(complete.values())}/20; "
             f"gradient images {report['actual_gradient_images']}/{plan['planned_gradient_images']}.", '',
             report['interpretation'], '', f'S3 common complete predefined blocks: {matched}.', '',
             '| Task | Target | Images | Target-clean successes/observed denominator | Expected denominator | ASR | Adv accuracy | Gradients | Attack seconds | Wall seconds | Status |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    def fmt(value):
        return 'missing' if value is None else f'{value:.6g}'
    for row in table:
        lines.append(f"| {row['id']} | {row['target']} | {row['evaluated_images']}/69 | "
                     f"{row['target_clean_successes']}/{row['target_clean_denominator_observed']} | {row['target_clean_denominator_expected']} | "
                     f"{fmt(row['asr_target_clean'])} | {fmt(row['adv_accuracy_all'])} | {row['actual_gradient_images']} | "
                     f"{fmt(row['attack_seconds'])} | {fmt(row['group_wall_seconds'])} | {row['status']} |")
    lines.extend(['', '## Matched260 paired contrasts', '', 'ASR effects are percentage points; margins remain model-specific native logit units.', '',
                  '| Target | Population | Contrast | Endpoint | Eligible images | Effect [95% CI] |', '|---|---|---|---|---:|---|'])
    for row in contrasts:
        scale = 100 if row['endpoint'] == 'success' else 1
        effect = (f"{row['estimate'] * scale:.4f} [{row['ci95_low'] * scale:.4f}, {row['ci95_high'] * scale:.4f}]") if 'estimate' in row else 'missing complete blocks'
        lines.append(f"| {row['target']} | {row['population']} | {row['comparison']} | {row['endpoint']} | {row['eligible_images']} | {effect} |")
    lines.extend(['', 'All partial image rows and PNG perturbation/margin/cost scalars are retained in CSV/JSON. '
                  'S2 OPS260 predictions are reused directly in S3; no extra generation/evaluation is implied. '
                  'EPS16 pilot24 is excluded from this report.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for target in plan['targets']:
        figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
        for axis, endpoint, scale in [(axes[0], 'success', 100), (axes[1], 'adversarial_margin', 1)]:
            data = [r for r in contrasts if r['target'] == target and r['population'] == 'target-clean' and r['endpoint'] == endpoint]
            for index, row in enumerate(data):
                if 'estimate' in row:
                    axis.plot(row['estimate'] * scale, index, 'o', color='#2864a0')
                    axis.hlines(index, row['ci95_low'] * scale, row['ci95_high'] * scale, color='#2864a0')
            axis.set_yticks(range(len(data)), [r['comparison'].replace('ops260_minus_', 'OPS - ') for r in data])
            axis.axvline(0, color='gray', linewidth=.7)
            axis.set_xlabel('ASR difference (pp)' if endpoint == 'success' else 'True-class margin difference')
            axis.grid(alpha=.2)
        figure.suptitle(f'{target}: complete blocks {matched}; single-seed exploratory95% CI')
        figure.savefig(output / f'{target}_matched260.png', dpi=180)
        figure.savefig(output / f'{target}_matched260.pdf')
        plt.close(figure)
    return {'output': str(output), 'status': report['status'], 'matched_S3_blocks': matched}
