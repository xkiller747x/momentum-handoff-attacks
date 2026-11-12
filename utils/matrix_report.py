"""Prespecified diagnostic summaries, paired contrasts, and scientific figures."""
import csv
import json
from pathlib import Path

import numpy as np

from .io import read_json, read_jsonl, write_json
from .matrix import SEEDS, TARGETS
from .metrics import paired_bootstrap, seed_average

PERTURBATION_KEYS = ['png_mean_abs_bytes', 'png_mean_abs_unit', 'png_l2_unit', 'png_boundary_element_fraction']


def _mean(rows, key):
    values = [r[key] for r in rows if key in r and r[key] is not None]
    return float(np.mean(values)) if values else None


def _csv(path, rows):
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with open(path, 'w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _key(job):
    return (job['budget'], job['method'], job['steps'], job['views'], job['alpha_mult'])


def collect(matrix_dir):
    matrix_dir = Path(matrix_dir)
    plan = read_json(matrix_dir / 'plan.json')
    state_path = matrix_dir / 'state.json'
    state = read_json(state_path) if state_path.exists() else {'jobs': {}}
    clean_rows = read_jsonl(matrix_dir / 'clean' / 'progress.jsonl')
    clean = {(r['model'], r['id']): r for r in clean_rows}
    ids = plan['selected_ids']
    per_seed, groups, attack_rows, evaluation_rows = [], {}, [], []
    known_clean = {target: all((target, identity) in clean for identity in ids) for target in TARGETS}
    eligible = {target: [identity for identity in ids if clean.get((target, identity), {}).get('correct')]
                for target in TARGETS}
    for job in plan['jobs']:
        attacks = read_jsonl(matrix_dir / 'attacks' / job['id'] / 'progress.jsonl')
        evaluations = read_jsonl(matrix_dir / 'evaluations' / job['id'] / 'progress.jsonl')
        attack_rows.extend({'job_id': job['id'], **row} for row in attacks)
        evaluation_rows.extend({'job_id': job['id'], **row} for row in evaluations)
        groups.setdefault(_key(job), {})[job['seed']] = {}
        for target in TARGETS:
            rows = [row for row in evaluations if row['model'] == target]
            mapping = {r['id']: r for r in rows}
            groups[_key(job)][job['seed']][target] = mapping
            own = [r for r in rows if r['clean_correct']]
            common = [r for r in rows if r.get('source_target_clean_eligible')]
            entry = {**job, 'target': target, 'planned_images': len(ids), 'attack_images': len(attacks),
                     'evaluated_images': len(rows), 'evaluation_complete': set(mapping) == set(ids),
                     'clean_reference_complete': known_clean[target],
                     'expected_target_clean_denominator': len(eligible[target]) if known_clean[target] else None,
                     'target_clean_numerator': sum(r['success'] for r in own),
                     'target_clean_denominator_observed': len(own),
                     'asr_target_clean': _mean(own, 'success'),
                     'adversarial_correct_numerator': sum(r['adversarial_correct'] for r in rows),
                     'adv_accuracy_denominator': len(rows), 'adv_accuracy_all': _mean(rows, 'adversarial_correct'),
                     'source_target_clean_numerator': sum(r['success'] for r in common),
                     'source_target_clean_denominator': len(common), 'asr_source_target_clean': _mean(common, 'success'),
                     'mean_clean_margin': _mean(own, 'clean_margin'), 'mean_adv_margin': _mean(own, 'adversarial_margin'),
                     'mean_clean_ce': _mean(own, 'clean_ce'), 'mean_adv_ce': _mean(own, 'adversarial_ce'),
                     'actual_forward_images': sum(r['forward_images'] for r in attacks),
                     'actual_gradient_images': sum(r['input_gradient_images'] for r in attacks),
                     'attack_seconds': sum(r['elapsed_seconds'] for r in attacks),
                     'group_cli_wall_seconds': state['jobs'].get(job['id'], {}).get('total_attempt_wall_seconds'),
                     'zero_grad_count': sum(r['zero_grad_count'] for r in attacks),
                     'max_png_byte_difference': max((r['png_max_byte_difference'] for r in attacks), default=None),
                     'status': state['jobs'].get(job['id'], {}).get('status', 'not_started'),
                     'exploratory_single_seed': job['budget'] == 'B200',
                     **{key: _mean(attacks, key) for key in PERTURBATION_KEYS}}
            per_seed.append(entry)
    return plan, state, clean, known_clean, eligible, groups, per_seed, attack_rows, evaluation_rows


def analyze(plan, known_clean, eligible, groups, per_seed):
    aggregate, contrasts = [], []

    def get_values(key, target, seeds, endpoint):
        target_ids = eligible[target]
        if not known_clean[target] or not target_ids:
            return None
        seed_rows = groups.get(key, {})
        values = {}
        for seed in seeds:
            mapping = seed_rows.get(seed, {}).get(target, {})
            # A CI needs the complete selected sample, not a smaller intersection
            # obtained by silently dropping unfinished images.
            if set(mapping) != set(plan['selected_ids']):
                return None
            if any(endpoint not in mapping[identity] for identity in target_ids):
                return None
            values[seed] = {identity: float(mapping[identity][endpoint]) for identity in target_ids}
        return seed_average(values, seeds, target_ids)

    for key, seed_rows in groups.items():
        budget, method, steps, views, alpha = key
        required_seeds = SEEDS if budget == 'B20' or (budget == 'classic' and method == 'di') else SEEDS[:1]
        for target in TARGETS:
            asr = get_values(key, target, required_seeds, 'success')
            margin = get_values(key, target, required_seeds, 'adversarial_margin')
            rows = [r for r in per_seed if _key(r) == key and r['target'] == target]
            complete_seeds = [r['seed'] for r in rows if r['evaluation_complete']]
            finite_rates = [r['asr_target_clean'] for r in rows if r['evaluation_complete'] and r['asr_target_clean'] is not None]
            row = {'budget': budget, 'method': method, 'steps': steps, 'views': views, 'alpha_mult': alpha,
                   'alpha_bytes': 8 * alpha / steps, 'target': target, 'required_seeds': required_seeds,
                   'complete_seeds': complete_seeds, 'seed_count_complete': len(complete_seeds),
                   'status': 'complete' if asr is not None else 'partial_or_missing',
                   'missing_seeds': [s for s in required_seeds if s not in complete_seeds],
                   'eligible_image_count': len(eligible[target]) if known_clean[target] else None,
                   'seed_asr_min': min(finite_rates) if finite_rates else None,
                   'seed_asr_max': max(finite_rates) if finite_rates else None,
                   'asr_seed_mean': float(asr.mean()) if asr is not None else None,
                   'adv_margin_seed_mean': float(margin.mean()) if margin is not None else None,
                   'adv_accuracy_seed_mean': _mean([r for r in rows if r['evaluation_complete']], 'adv_accuracy_all'),
                   'exploratory_single_seed': budget == 'B200',
                   'ci_condition': f'conditional on {len(required_seeds)} fixed seed(s), empirical image distribution',
                   'asr_ci': paired_bootstrap(asr) if asr is not None else None,
                   **{key: _mean(rows, key) for key in PERTURBATION_KEYS},
                   'attack_seconds_all_observed_seeds': sum(r['attack_seconds'] for r in rows),
                   'actual_gradient_images_all_observed_seeds': sum(r['actual_gradient_images'] for r in rows)}
            aggregate.append(row)

    specs = [
        ('primary_B20_alpha1', 'B20', (5, 4, 1), (20, 1, 1), SEEDS),
        ('robustness_B20_alpha2', 'B20', (5, 4, 2), (20, 1, 2), SEEDS),
        ('fixed_step_0.8byte', 'B20', (10, 2, 1), (20, 1, 2), SEEDS),
        ('fixed_step_1.6byte', 'B20', (5, 4, 1), (10, 2, 2), SEEDS),
        # B200 direction: more views (T10,N20) minus more iterations (T40,N5).
        ('exploratory_B200_alpha1', 'B200', (10, 20, 1), (40, 5, 1), SEEDS[:1]),
    ]
    for name, budget, high_views, low_views, seeds in specs:
        for target in TARGETS:
            for endpoint in ('success', 'adversarial_margin'):
                cells, missing = {}, []
                for method in ('sia', 'di-n'):
                    for cell, config in [('high_views', high_views), ('low_views', low_views)]:
                        key = (budget, method, *config)
                        value = get_values(key, target, seeds, endpoint)
                        cells[(method, cell)] = value
                        if value is None:
                            missing.append({'method': method, 'steps_views_alpha': config,
                                            'required_seeds': seeds})
                saturation = (endpoint == 'success' and not missing and
                              all(float(value.mean()) > .95 for value in cells.values()))
                for comparison in ('sia_allocation', 'di-n_allocation', 'difference_in_differences'):
                    needed_methods = ['sia', 'di-n'] if comparison == 'difference_in_differences' else [comparison.split('_')[0]]
                    needed_missing = [m for m in missing if m['method'] in needed_methods]
                    result = {'contrast': name, 'target': target, 'endpoint': endpoint, 'comparison': comparison,
                              'direction': 'high_views minus low_views; DID=SIA allocation minus DI-N allocation',
                              'high_views_steps_views_alpha': high_views, 'low_views_steps_views_alpha': low_views,
                              'seed_count': len(seeds), 'seeds': seeds, 'eligible_image_count': len(eligible[target]),
                              'exploratory': True, 'exploratory_single_seed': budget == 'B200',
                              'status': 'missing' if needed_missing else 'complete', 'missing_cells': needed_missing,
                              'all_four_asr_cells_above_95pct': saturation,
                              'ci_condition': 'fixed seeds; paired empirical image IDs; no architecture-family inference'}
                    if not needed_missing:
                        differences = {method: cells[(method, 'high_views')] - cells[(method, 'low_views')]
                                       for method in needed_methods}
                        values = differences['sia'] - differences['di-n'] if len(needed_methods) == 2 else differences[needed_methods[0]]
                        result.update(paired_bootstrap(values))
                        seed_estimates = {}
                        for seed in seeds:
                            seed_differences = {}
                            for method in needed_methods:
                                high = get_values((budget, method, *high_views), target, [seed], endpoint)
                                low = get_values((budget, method, *low_views), target, [seed], endpoint)
                                seed_differences[method] = high - low
                            seed_values = (seed_differences['sia'] - seed_differences['di-n'] if len(needed_methods) == 2
                                           else seed_differences[needed_methods[0]])
                            seed_estimates[str(seed)] = float(seed_values.mean())
                        result['per_seed_estimates'] = seed_estimates
                        result['seed_estimate_min'] = min(seed_estimates.values())
                        result['seed_estimate_max'] = max(seed_estimates.values())
                    contrasts.append(result)
    return aggregate, contrasts


def _figures(output, aggregate):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for target in TARGETS:
        figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
        for column, alpha in enumerate((1, 2)):
            for method, color in [('di-n', '#2864a0'), ('sia', '#c65526')]:
                rows = [next((r for r in aggregate if r['target'] == target and r['budget'] == 'B20'
                              and r['method'] == method and r['steps'] == steps and r['alpha_mult'] == alpha), None)
                        for steps in (20, 10, 5)]
                for axis, endpoint, scale in [(axes[0, column], 'asr_seed_mean', 100),
                                               (axes[1, column], 'adv_margin_seed_mean', 1)]:
                    y = [r[endpoint] * scale if r and r[endpoint] is not None else np.nan for r in rows]
                    axis.plot(range(3), y, marker='o', color=color, label=method)
                    if endpoint == 'asr_seed_mean':
                        for index, row in enumerate(rows):
                            if row and row['asr_ci']:
                                ci = row['asr_ci']
                                axis.vlines(index, ci['ci95_low'] * 100, ci['ci95_high'] * 100, color=color, alpha=.65)
                    axis.set_xticks(range(3), ['T20 N1', 'T10 N2', 'T5 N4'])
                    axis.grid(alpha=.2)
            axes[0, column].set_title(f'alpha={alpha} epsilon/T')
            axes[0, column].set_ylabel('Target-clean ASR (%)')
            axes[0, column].set_ylim(0, 101)
            axes[1, column].set_ylabel('True-class adversarial margin')
            axes[1, column].set_xlabel('Allocation at 20 gradient images')
            axes[0, column].legend()
        figure.suptitle(f'{target}: fixed-seed image means; missing cells left blank')
        figure.savefig(output / f'{target}_B20.png', dpi=180)
        figure.savefig(output / f'{target}_B20.pdf')
        plt.close(figure)


def matrix_report_command(args):
    from .runner import _output
    output = _output(args.output)
    plan, state, clean, known_clean, eligible, groups, per_seed, attack_rows, evaluation_rows = collect(args.matrix_run)
    aggregate, contrasts = analyze(plan, known_clean, eligible, groups, per_seed)
    observed = {(r['job_id'], r['model'], r['id']) for r in evaluation_rows}
    complete_jobs = sum(all((job['id'], target, identity) in observed
                            for identity in plan['selected_ids'] for target in TARGETS)
                        for job in plan['jobs'])
    summary = {'status': 'complete' if complete_jobs == len(plan['jobs']) else 'partial',
               'complete_jobs': complete_jobs, 'planned_jobs': len(plan['jobs']),
               'planned_gradient_images': plan['planned_total_gradient_images'],
               'actual_gradient_images': sum(r['input_gradient_images'] for r in attack_rows),
               'protocol_sha256': plan['protocol_sha256'], 'image_count': len(plan['selected_ids']),
               'clean_reference_complete': known_clean,
               'interpretation': 'Exploratory intervals conditional on fixed seeds. CI crossing zero is not evidence of equivalence. '
                                 '5 percentage points is only a magnitude reference. B200 is single-seed exploratory. '
                                 'Gradient-image matching is not matching FLOPs or wall time.',
               'per_seed': per_seed, 'seed_averaged': aggregate, 'contrasts': contrasts}
    write_json(output / 'report.json', summary)
    _csv(output / 'per_seed.csv', per_seed)
    _csv(output / 'seed_averaged.csv', aggregate)
    _csv(output / 'contrasts.csv', contrasts)
    _csv(output / 'attack_images.csv', attack_rows)
    _csv(output / 'evaluation_images.csv', evaluation_rows)
    _csv(output / 'clean_images.csv', list(clean.values()))
    lines = ['# Frozen diagnostic matrix report', '',
             f"Status: **{summary['status']}**; complete jobs {complete_jobs}/{len(plan['jobs'])}; "
             f"gradient images {summary['actual_gradient_images']}/{plan['planned_total_gradient_images']}.", '',
             summary['interpretation'], '',
             'Intervals use 10,000 image-paired resamples, seed20261030, after within-image attack-seed averaging. '
             'Targets are reported independently. Missing configurations retain their rows and receive no CI.', '',
             '| Target | Configuration | Seed | Images | ASR numerator/denominator | ASR | Adv accuracy | Status |',
             '|---|---|---:|---:|---:|---:|---:|---|']
    def percentage(value):
        return f'{100 * value:.2f}%' if value is not None else 'missing'
    for row in per_seed:
        lines.append(f"| {row['target']} | {row['budget']} {row['method']} T{row['steps']} N{row['views']} a{row['alpha_mult']} "
                     f"| {row['seed']} | {row['evaluated_images']}/{row['planned_images']} "
                     f"| {row['target_clean_numerator']}/{row['target_clean_denominator_observed']} "
                     f"| {percentage(row['asr_target_clean'])} | {percentage(row['adv_accuracy_all'])} "
                     f"| {'complete' if row['evaluation_complete'] else 'partial/missing'} |")
    lines.extend(['', '## Prespecified paired contrasts', '',
                  'ASR effects below are percentage points; margin effects remain in each model’s native logit units.', '',
                  '| Target | Contrast | Endpoint | Comparison | Images | Seeds | Estimate [95% CI] | Status |',
                  '|---|---|---|---|---:|---:|---|---|'])
    saturation_notes = set()
    for row in contrasts:
        scale = 100 if row['endpoint'] == 'success' else 1
        estimate = (f"{row['estimate'] * scale:.4f} [{row['ci95_low'] * scale:.4f}, {row['ci95_high'] * scale:.4f}]; "
                    f"half-width {row['ci95_half_width'] * scale:.4f}") if row['status'] == 'complete' else 'missing required cells'
        lines.append(f"| {row['target']} | {row['contrast']} | {row['endpoint']} | {row['comparison']} "
                     f"| {row['eligible_image_count']} | {row['seed_count']} | {estimate} | {row['status']} |")
        if row['all_four_asr_cells_above_95pct']:
            saturation_notes.add(f"Saturation note: {row['target']} / {row['contrast']} has all four ASR cells above95%; "
                                 'probability-scale separation is limited. ASR and margins are retained.')
    lines.extend([''] + sorted(saturation_notes))
    lines.extend(['', '## Observed cost, margin and PNG perturbation', '',
                  'Values below describe observed images; missing/partial jobs remain explicitly marked above. '
                  'L2 is in [0,1] units without pixel-count normalization. Boundary fraction uses channel elements.', '',
                  '| Target | Job | Adv margin (target-clean) | Adv CE | Source gradients | Attack seconds | Group wall seconds | PNG mean abs bytes | PNG L2 unit | Boundary fraction |',
                  '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|'])
    def number(value):
        return f'{value:.6g}' if value is not None else 'missing'
    for row in per_seed:
        lines.append(f"| {row['target']} | {row['id']} | {number(row['mean_adv_margin'])} | {number(row['mean_adv_ce'])} "
                     f"| {row['actual_gradient_images']} | {number(row['attack_seconds'])} | {number(row['group_cli_wall_seconds'])} "
                     f"| {number(row['png_mean_abs_bytes'])} | {number(row['png_l2_unit'])} | {number(row['png_boundary_element_fraction'])} |")
    lines.extend(['', 'Full seed ranges, scalar endpoints, per-image data, cost and perturbation units are in companion JSON/CSV files.',
                  'B200 T10N20 minus T40N5 is exploratory and conditional on one seed. '
                  'Fixed-step slices partly control step size, while momentum/update counts still differ.', ''])
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    _figures(output, aggregate)
    return {'output': str(output), 'status': summary['status'], 'complete_jobs': complete_jobs,
            'planned_jobs': len(plan['jobs'])}
