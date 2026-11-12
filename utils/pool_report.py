"""Frozen image-level pool observation endpoints and descriptive noise usage."""
from pathlib import Path

import numpy as np

from .io import read_json, read_jsonl, sha256, write_json
from .matrix_report import _csv
from .metrics import paired_bootstrap
from .pool_diagnostic import BOOTSTRAP_SEED, FORWARDS_PER_IMAGE


def summarize_endpoints(plan, rows):
    ids = plan['formal_all_ids']
    selected = {row['id']: row for row in rows}
    if len(ids) != 276 or len(set(ids)) != 276 or len(plan['source_clean_ids']) != 267 or not set(plan['source_clean_ids']).issubset(ids):
        raise ValueError('Formal source-clean267/all276 population definition differs')
    if len(selected) != len(rows) or any(identity not in plan['selected_ids'] for identity in selected):
        raise ValueError('Duplicate/unplanned observation image')
    if any(row['source_clean_correct'] != (row['id'] in plan['source_clean_ids']) or
           row['forward_images'] != FORWARDS_PER_IMAGE or row.get('input_gradient_images', 0) != 0 or
           row.get('target_forward_images', 0) != 0 or not row.get('all_no_grad', True) for row in rows):
        raise ValueError('Observed source-clean identity/forward count differs from plan')
    complete = plan['mode'] == 'formal276' and len(ids) == 276 and set(selected) == set(ids)
    result = {'status': 'complete' if complete else 'engineering_only' if plan['mode'] != 'formal276' else 'partial',
              'completed_images': len(rows), 'planned_images': len(plan['selected_ids']),
              'formal_image_count': 276, 'forward_images': sum(row['forward_images'] for row in rows),
              'planned_forward_images': plan['planned_forward_images'], 'input_gradient_images': 0,
              'target_forward_images': 0, 'populations': {},
              'ci_condition': 'single original attack seed20261027, fixed three fresh pools and operator namespace; image is statistical unit'}
    for name, eligible in [('source_clean267', plan['source_clean_ids']), ('all276', ids)]:
        available = [selected[identity] for identity in eligible if identity in selected]
        description = {'observed_images': len(available), 'expected_images': len(eligible), 'endpoints': {},
                       'D_median': float(np.median([row['D'] for row in available])) if available else None,
                       'D_positive_fraction': float(np.mean([row['D'] > 0 for row in available])) if available else None}
        for endpoint in ('D', 'G', 'C', 'A'):
            values = np.array([row[endpoint] for row in available], dtype=np.float64)
            entry = {'mean_observed': float(values.mean()) if len(values) else None,
                     'role': 'primary' if endpoint == 'D' and name == 'source_clean267' else 'secondary/sensitivity',
                     'ci': paired_bootstrap(values, seed=BOOTSTRAP_SEED) if complete else None}
            description['endpoints'][endpoint] = entry
        a = description['endpoints']['A']
        description['D_over_A_descriptive_point_ratio'] = (
            description['endpoints']['D']['mean_observed'] / a['mean_observed']
            if complete and a['mean_observed'] > 0 and a['ci']['ci95_low'] > 0 else None)
        description['ratio_condition'] = 'Only if A mean and lower95% CI bound are positive; no ratio CI or threshold decision'
        result['populations'][name] = description
    return result


def describe_usage(details):
    """Descriptive per-image aggregation; no regression or noise-level inference."""
    noise_rows, count_rows, radius_rows, within_radius = [], [], [], []
    for detail in details:
        usage = detail['pool_metadata']['usage_count']
        last = detail['pool_metadata']['last_step']
        residual = detail['noise_residual_d']
        for index in range(30):
            noise_rows.append({'id': detail['id'], 'source_clean_correct': detail['source_clean_correct'],
                               'pool_index': index, 'radius_index': index // 5,
                               'usage_count': usage[index], 'used': usage[index] > 0,
                               'last_step': last[index], 'CE_gain_residual_d': residual[index]})
        for radius in range(6):
            indices = list(range(radius * 5, radius * 5 + 5))
            used = [residual[i] for i in indices if usage[i] > 0]
            unused = [residual[i] for i in indices if usage[i] == 0]
            within_radius.append({'id': detail['id'], 'source_clean_correct': detail['source_clean_correct'],
                                  'radius_index': radius, 'used_noise_count': len(used), 'unused_noise_count': len(unused),
                                  'eligible_both_groups': bool(used and unused),
                                  'used_mean_d': float(np.mean(used)) if used else None,
                                  'unused_mean_d': float(np.mean(unused)) if unused else None,
                                  'used_minus_unused_d': float(np.mean(used) - np.mean(unused)) if used and unused else None})
    for population in ('source_clean', 'all_observed'):
        chosen = [row for row in noise_rows if population == 'all_observed' or row['source_clean_correct']]
        for count in range(11):
            per_image = {}
            for row in chosen:
                if row['usage_count'] == count:
                    per_image.setdefault(row['id'], []).append(row['CE_gain_residual_d'])
            count_rows.append({'population': population, 'usage_count': count, 'available_images': len(per_image),
                               'noise_count': sum(len(values) for values in per_image.values()),
                               'mean_within_image_mean_d': float(np.mean([np.mean(values) for values in per_image.values()])) if per_image else None,
                               'noise_weighted_mean_d_descriptive': float(np.mean([v for values in per_image.values() for v in values])) if per_image else None})
        for radius in range(6):
            candidates = [row for row in within_radius if row['radius_index'] == radius and
                          (population == 'all_observed' or row['source_clean_correct'])]
            eligible = [row for row in candidates if row['eligible_both_groups']]
            radius_rows.append({'population': population, 'radius_index': radius,
                                'candidate_images': len(candidates), 'available_images': len(eligible),
                                'missing_images': len(candidates) - len(eligible),
                                'used_noise_count_eligible': sum(row['used_noise_count'] for row in eligible),
                                'unused_noise_count_eligible': sum(row['unused_noise_count'] for row in eligible),
                                'mean_used_minus_unused_d': float(np.mean([r['used_minus_unused_d'] for r in eligible])) if eligible else None})
    return {'noise_rows': noise_rows, 'usage_count_summary': count_rows,
            'within_image_radius_rows': within_radius, 'radius_used_unused_summary': radius_rows}


def pool_report_command(args):
    from .runner import _output
    source, output = Path(args.pool_run).resolve(), _output(args.output)
    plan, rows = read_json(source / 'plan.json'), read_jsonl(source / 'progress.jsonl')
    summary = summarize_endpoints(plan, rows)
    details = []
    for row in rows:
        path = source / row['detail_path']
        if sha256(path) != row['detail_sha256']:
            raise ValueError('Committed observation detail SHA mismatch')
        detail = read_json(path)
        if detail['id'] != row['id'] or any(detail['ce_endpoints'][key] != row[key] for key in ('D', 'G', 'C', 'A')):
            raise ValueError('Committed observation endpoint/detail mismatch')
        if detail['actual_counter']['forward_images'] != FORWARDS_PER_IMAGE or not detail['actual_counter']['all_no_grad']:
            raise ValueError('Actual source count/no_grad differs from protocol')
        if detail['actual_counter']['input_gradient_images'] != 0 or detail['actual_counter']['target_forward_images'] != 0:
            raise ValueError('Observation must have0 input gradients and0 target forwards')
        details.append(detail)
    usage = describe_usage(details)
    placebo_rows, state_rows, margin_rows = [], [], []
    for detail in details:
        for row in detail['ce_endpoints']['fresh_placebo']:
            placebo_rows.append({'id': detail['id'], 'source_clean_correct': detail['source_clean_correct'], **row})
        state_rows.extend({'id': detail['id'], **row} for row in detail['noise_state_details'])
        margin_rows.append({'id': detail['id'], 'source_clean_correct': detail['source_clean_correct'],
                            **{key: detail['margin_descriptive'][key] for key in ('D', 'G', 'C', 'A')}})
    placebo_summary = []
    complete = summary['status'] == 'complete'
    for population in ('source_clean267', 'all276'):
        for pool in ('fresh1', 'fresh2', 'fresh3'):
            selected = [r for r in placebo_rows if r['fresh_pool'] == pool and
                        (population == 'all276' or r['source_clean_correct'])]
            for endpoint in ('clean', 'png', 'gain_difference'):
                values = np.array([r[endpoint] for r in selected])
                placebo_summary.append({'population': population, 'fresh_pool': pool, 'endpoint': endpoint,
                    'image_count': len(values), 'mean_observed': float(values.mean()) if len(values) else None,
                    'ci': paired_bootstrap(values, seed=BOOTSTRAP_SEED) if complete else None})
    summary.update({'protocol_sha256': plan['protocol_sha256'], 'kernel_sha256': plan['kernel_sha256'],
                    'plan_sha256': sha256(source / 'plan.json'),
                    'config_sha256': sha256(source / 'config.json') if (source / 'config.json').exists() else None,
                    'runtime': read_json(source / 'runtime.json') if (source / 'runtime.json').exists() else None,
                    'actual_forward_pixels': sum(d['actual_counter']['forward_pixels'] for d in details),
                    'summed_image_elapsed_seconds': sum(d['elapsed_seconds'] for d in details),
                    'peak_cuda_allocated_bytes': max((d['peak_cuda_allocated_bytes'] for d in details), default=0),
                    'placebo_summary': placebo_summary, 'usage_count_summary': usage['usage_count_summary'],
                    'radius_used_unused_summary': usage['radius_used_unused_summary'],
                    'margin_descriptive_means': {population: {key: float(np.mean([row[key] for row in margin_rows if
                        population == 'all_observed' or row['source_clean_correct']])) if any(population == 'all_observed' or row['source_clean_correct'] for row in margin_rows) else None
                        for key in ('D', 'G', 'C', 'A')} for population in ('source_clean', 'all_observed')},
                    'interpretation': 'Post-target-results exploratory source observation, not an attack method. '
                    'D is primary; G/C/A are secondary. Three fresh-placebo rotations sum to zero algebraically and are not independent validation. '
                    'Random clean/placebo gaps are not deterministic failure gates. Source adaptation cannot establish transfer harm or novelty. '
                    'Partial/engineering runs do not receive full-population CIs. No noise-level causal inference or ratio CI.'})
    write_json(output / 'report.json', summary)
    _csv(output / 'images.csv', rows)
    _csv(output / 'noise_state_Kmeans.csv', state_rows)
    _csv(output / 'noise_usage_residuals.csv', usage['noise_rows'])
    _csv(output / 'usage_count_summary.csv', usage['usage_count_summary'])
    _csv(output / 'within_image_radius.csv', usage['within_image_radius_rows'])
    _csv(output / 'radius_used_unused_summary.csv', usage['radius_used_unused_summary'])
    _csv(output / 'placebo_images.csv', placebo_rows)
    _csv(output / 'placebo_summary.csv', placebo_summary)
    _csv(output / 'margin_descriptive_images.csv', margin_rows)
    lines = ['# OPS finite-pool source observation', '', f"Status: **{summary['status']}**, "
             f"{len(rows)}/{len(plan['selected_ids'])} images; {summary['forward_images']}/{plan['planned_forward_images']} source forwards;0 gradients;0 target forwards.", '',
             summary['interpretation'], '',
             '| Population | Endpoint | Observed/expected images | Mean |95% image CI |', '|---|---|---:|---:|---|']
    descriptions = []
    for name, population in summary['populations'].items():
        for key, endpoint in population['endpoints'].items():
            ci = endpoint['ci']
            ci_text = f"[{ci['ci95_low']:.6g}, {ci['ci95_high']:.6g}]" if ci else 'not produced before complete276'
            mean = endpoint['mean_observed']
            lines.append(f"| {name} | {key} | {population['observed_images']}/{population['expected_images']} | "
                         f"{mean if mean is not None else 'missing'} | {ci_text} |")
        descriptions.append(f"{name}: median D={population['D_median']}; positive-D fraction={population['D_positive_fraction']}; "
                            f"descriptive D/A={population['D_over_A_descriptive_point_ratio']} (only if A mean/lower CI positive).")
    lines.extend([''] + descriptions)
    lines.extend(['', '## Usage and radius descriptions', '',
                  '| Population | Radius index | Available/candidate images | Missing | Used/unused noise counts | Mean used-minus-unused residual |',
                  '|---|---:|---:|---:|---:|---:|'])
    for row in usage['radius_used_unused_summary']:
        lines.append(f"| {row['population']} | {row['radius_index']} | {row['available_images']}/{row['candidate_images']} | "
                     f"{row['missing_images']} | {row['used_noise_count_eligible']}/{row['unused_noise_count_eligible']} | {row['mean_used_minus_unused_d']} |")
    lines.extend(['', 'Fresh1/2/3 rotations, ordinary views, complete operator parameters, pool/noise SHA and training usage/last-step remain in per-image details and companion CSV. '
                  'The three placebo rotations sum to zero; that identity is not validation. '
                  'Usage0..10 is descriptive within image; missing radius groups are not zero-filled.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    values = [row['D'] for row in rows if row['source_clean_correct']]
    if values:
        axes[0].hist(values, bins=min(25, max(3, len(values))), color='#2864a0', alpha=.8)
    axes[0].axvline(0, color='gray', linewidth=.8)
    axes[0].set_xlabel('Per-image D (source CE)')
    axes[0].set_ylabel('Images; fixed source-clean population observed')
    for index, key in enumerate(('D', 'G', 'C', 'A')):
        endpoint = summary['populations']['source_clean267']['endpoints'][key]
        if endpoint['mean_observed'] is not None:
            axes[1].plot(index, endpoint['mean_observed'], 'o', color='#c65526')
            if endpoint['ci']:
                axes[1].vlines(index, endpoint['ci']['ci95_low'], endpoint['ci']['ci95_high'], color='#c65526')
    axes[1].set_xticks(range(4), ['D(primary)', 'G', 'C', 'A'])
    axes[1].set_ylabel('Source CE; CI only when formal276 complete')
    axes[1].axhline(0, color='gray', linewidth=.8)
    figure.suptitle(f"{summary['status']}: finite-pool source observation, fixed-seed exploratory")
    figure.savefig(output / 'pool_observation.png', dpi=180)
    figure.savefig(output / 'pool_observation.pdf')
    plt.close(figure)
    return {'output': str(output), 'status': summary['status'], 'completed_images': len(rows),
            'full_population_CI_produced': complete}
