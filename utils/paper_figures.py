"""Static paper figures (light mode, print) built only from saved evaluation records.

Colors are the first three validated categorical slots of the reference palette
(all-pairs CVD dE >= 9.2). Aqua is below 3:1 on the surface, so every series also
gets a distinct marker shape and a direct label; Table 3 is the table view.
"""
import json
import random
from pathlib import Path

SURFACE = '#fcfcfb'
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
SERIES = {'FFT': ('#2a78d6', 'o'), 'AaF': ('#eb6834', 's'), 'ILA': ('#1baf7a', '^')}
NAMES = {'rn50_v1': 'RN50', 'dn121': 'DN121', 'vgg16_bn': 'VGG16-BN', 'vit_b_16': 'ViT-B/16'}
HOLDOUT_TARGETS = {'rn50_v1': ['dn121', 'vgg16_bn', 'vit_b_16'], 'dn121': ['rn50_v1', 'vgg16_bn', 'vit_b_16'],
                   'vgg16_bn': ['rn50_v1', 'dn121', 'vit_b_16']}


def load_success(path):
    rows = {}
    for line in open(Path(path) / 'progress.jsonl', encoding='utf-8'):
        record = json.loads(line)
        rows[(record['model'], record['id'])] = record['adv']
    return rows


def paired_difference(a, b, model, ids, draws=4000, seed=2026):
    """Mean success difference a-b with a percentile paired-bootstrap 95% interval.

    model may be a tuple of target models; each image then contributes its mean over those targets.
    """
    models = model if isinstance(model, tuple) else (model,)
    diff = [sum(int(a[(m, i)]['targeted_success']) - int(b[(m, i)]['targeted_success']) for m in models)
            / len(models) for i in ids]
    rng = random.Random(seed)
    n = len(diff)
    boot = sorted(sum(rng.choice(diff) for _ in range(n)) / n for _ in range(draws))
    return 100 * sum(diff) / n, 100 * boot[int(0.025 * draws)], 100 * boot[int(0.975 * draws) - 1]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis='x', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def log_odds_change(a, b, models, ids):
    """Per-image change a-b in target log-odds, averaged over the given target models."""
    return [sum(a[(m, i)]['target_log_odds'] - b[(m, i)]['target_log_odds'] for m in models) / len(models)
            for i in ids]


def _boxes(ax, groups, series, width):
    """Vertical boxes: groups is a list of {series_key: values}; series maps key -> (label, color).

    Box = interquartile range, line = median, whiskers = 5th-95th percentile, outliers not drawn.
    """
    offsets = [(k - (len(series) - 1) / 2) * width * 1.15 for k in range(len(series))]
    for (key, (label, color)), offset in zip(series.items(), offsets):
        data = [group[key] for group in groups]
        ax.boxplot(data, positions=[g + offset for g in range(len(groups))], widths=width, whis=(5, 95),
                   showfliers=False, patch_artist=True,
                   boxprops={'facecolor': color, 'edgecolor': color, 'alpha': 0.85, 'linewidth': 1},
                   medianprops={'color': SURFACE, 'linewidth': 1.5},
                   whiskerprops={'color': color, 'linewidth': 1.2}, capprops={'color': color, 'linewidth': 1.2})
        ax.plot([], [], 's', color=color, markersize=6, label=label)
    ax.axhline(0, color=INK2, linewidth=1, zorder=0)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.grid(axis='x', visible=False)
    ax.set_xlim(-0.6, len(groups) - 0.4)


def holdout_box(run_dir, output):
    """Hold-out: per-image target log-odds, inherited minus original, for every surrogate-target pair."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels, groups = [], []
    for source, targets in HOLDOUT_TARGETS.items():
        for target in targets:
            labels.append(f'{NAMES[source]}\n→ {NAMES[target]}')
            entry = {}
            for method in SERIES:
                key = method.lower()
                a = load_success(Path(run_dir) / f'eval_{source}_{key}_continue')
                b = load_success(Path(run_dir) / f'eval_{source}_{key}_author')
                entry[method] = log_odds_change(a, b, (target,), sorted({i for (_, i) in a}))
            groups.append(entry)
    fig, ax = plt.subplots(figsize=(7.0, 2.7), facecolor=SURFACE)
    _style(ax)
    _boxes(ax, groups, {m: (m, color) for m, (color, _) in SERIES.items()}, width=0.24)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, color=INK)
    ax.set_ylabel('Target log-odds,\ninherited − original', fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(0.0, 1.0), ncol=3, labelcolor=INK)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(output) + suffix, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return groups


def switch_timing_box(run_dir, output):
    """Per-image change in CNN-mean target log-odds when the switch moves from iteration 160 to 130 (dev276)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cnn = {'rn50_v1': ('dn121', 'vgg16_bn'), 'dn121': ('rn50_v1', 'vgg16_bn')}
    series = {'reset': ('Reset (original)', '#eb6834'), 'keep': ('Inherited', '#2a78d6')}
    labels, groups = [], []
    for source, targets in cnn.items():
        for method in ('fft', 'aaf', 'ila'):
            labels.append(f'{NAMES[source]}\n{method.upper() if method != "aaf" else "AaF"}')
            entry = {}
            for state in series:
                early = load_success(Path(run_dir) / f'eval_{source}_{method}_s130_{state}')
                late = load_success(Path(run_dir) / f'eval_{source}_{method}_s160_{state}')
                entry[state] = log_odds_change(early, late, targets, sorted({i for (_, i) in late}))
            groups.append(entry)
    fig, ax = plt.subplots(figsize=(3.4, 2.7), facecolor=SURFACE)
    _style(ax)
    _boxes(ax, groups, series, width=0.34)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, color=INK)
    ax.set_ylabel('Target log-odds change,\nswitch at 130 − at 160', fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=7, loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=2,
              labelcolor=INK, title='Momentum at the switch', title_fontsize=7, alignment='left')
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(output) + suffix, dpi=300, facecolor=SURFACE)
    plt.close(fig)
    return groups


def decay_curve(points, output):
    """points: {series_label: [(gamma, mean_log_odds), ...]} from dev64 runs."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gammas = sorted({g for series in points.values() for g, _ in series})
    position = {g: k for k, g in enumerate(gammas)}
    fig, ax = plt.subplots(figsize=(4.6, 3.2), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    for (label, series), (color, marker) in zip(points.items(), list(SERIES.values())):
        xs = [position[g] for g, _ in series]
        ys = [v for _, v in series]
        # Solid only between adjacent measured gammas; a skipped range is drawn dotted.
        for k in range(len(xs) - 1):
            style = '-' if xs[k + 1] - xs[k] == 1 else ':'
            ax.plot(xs[k:k + 2], ys[k:k + 2], color=color, linewidth=2, linestyle=style)
        ax.plot(xs, ys, color=color, linewidth=0, marker=marker, markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(6, 0), textcoords='offset points',
                    fontsize=8, color=INK, va='center')
    ax.set_xticks(range(len(gammas)))
    ax.set_xticklabels([f'{g:g}' for g in gammas], fontsize=8)
    ax.set_xlabel('Fraction of inherited momentum kept (γ); 0 = reset, 1 = inherit', fontsize=8, color=INK2)
    ax.set_ylabel('Mean target log-odds', fontsize=8, color=INK2)
    ax.legend(frameon=False, fontsize=8, loc='lower right', labelcolor=INK)
    fig.tight_layout()
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(output) + suffix, dpi=300, facecolor=SURFACE)
    plt.close(fig)


def flips_by_label(reset, inherit, target, target_labels):
    """Images whose success on target differs between hand-offs, grouped by target class.

    reset/inherit: load_success records; target_labels: image id -> target class. Returns
    {'inherit': {label: [ids]}, 'reset': {label: [ids]}} for the hand-off that succeeds; ids sorted.
    """
    flips = {'inherit': {}, 'reset': {}}
    for i in sorted({i for (_, i) in reset}):
        r, k = reset[(target, i)]['targeted_success'], inherit[(target, i)]['targeted_success']
        if r != k:
            flips['inherit' if k else 'reset'].setdefault(target_labels[i], []).append(i)
    return flips


def example_grid(cases, manifest, output, epsilon_bytes=16):
    """Clean image, both adversarial images and both perturbations for each case.

    cases: dicts with title, id, target model, and reset/inherit (attack run dir, eval dir) pairs.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from torchvision.models import ResNet50_Weights
    names = ResNet50_Weights.IMAGENET1K_V1.meta['categories']
    paths = {r['id']: r['canonical_path'] for r in json.load(open(manifest, encoding='utf-8'))['records']}
    short = lambda k: names[k].split(',')[0]
    columns = ['Clean', 'Original (reset)', 'Inherited momentum', 'Perturbation (reset)', 'Perturbation (inherited)']
    fig, axes = plt.subplots(len(cases), 5, figsize=(7.0, 1.75 * len(cases) + 0.3), facecolor='white')
    for row, case in enumerate(cases):
        clean = np.asarray(Image.open(paths[case['id']]).convert('RGB'), dtype=np.int16)
        panels, notes = [clean.astype(np.uint8)], []
        record = None
        for state in ('reset', 'inherit'):
            run, evaluation = case[state]
            adv = np.asarray(Image.open(Path(run) / 'images' / f"{case['id']}.png").convert('RGB'), dtype=np.int16)
            panels.append(adv.astype(np.uint8))
            record = evaluation[(case['target'], case['id'])]
            hit = 'target' if record['targeted_success'] else 'not target'
            notes.append(f"{short(record['prediction'])}\n({hit})")
        for state in ('reset', 'inherit'):
            run, _ = case[state]
            adv = np.asarray(Image.open(Path(run) / 'images' / f"{case['id']}.png").convert('RGB'), dtype=np.int16)
            panels.append(np.clip(0.5 + (adv - clean) / (2.0 * epsilon_bytes), 0, 1))
        clean_note = f"true: {short(case['label'])}\ntarget: {short(case['target_label'])}"
        for col, (ax, panel) in enumerate(zip(axes[row], panels)):
            ax.imshow(panel)
            ax.set_xticks([]); ax.set_yticks([])
            for side in ax.spines.values():
                side.set_visible(False)
            if row == 0:
                ax.set_title(columns[col], fontsize=7, color=INK)
            text = clean_note if col == 0 else (notes[col - 1] if col in (1, 2) else '')
            ax.set_xlabel(text, fontsize=6.5, color=INK2 if col != 0 else INK, labelpad=2)
        axes[row][0].set_ylabel(case['title'], fontsize=7, color=INK)
    fig.tight_layout(pad=0.3, h_pad=0.6, w_pad=0.4)
    for suffix in ('.png', '.pdf'):
        fig.savefig(str(output) + suffix, dpi=300, facecolor='white')
    plt.close(fig)


def figures_command(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    holdout_box('results/holdout_final_v1', out / 'fig_holdout')
    switch_timing_box('results/early_switch_dev276_v1', out / 'fig_switch_timing')
    # Examples (dev276, switch 160, CNN targets; methods in order FFT, AaF, ILA): the first target class for which two
    # images succeed only with inherited momentum; the two smallest ids are shown.
    runs = {('rn50_v1', 'fft'): ('results/feature_dev276_v1/fft', 'results/fine_continue_dev276_v1/fft_continue'),
            ('rn50_v1', 'aaf'): ('results/feature_dev276_v1/aaf', 'results/fine_continue_dev276_v1/aaf_continue'),
            ('rn50_v1', 'ila'): ('results/ila_dev276_v1/rn50_v1_ila_author', 'results/ila_dev276_v1/rn50_v1_ila_continue')}
    for source in ('dn121',):
        for method in ('fft', 'aaf', 'ila'):
            runs[(source, method)] = (f'results/dn121_source_dev276_v1/{method}_author',
                                      f'results/dn121_source_dev276_v1/{method}_continue')
        runs[(source, 'ila')] = ('results/ila_dev276_v1/dn121_ila_author', 'results/ila_dev276_v1/dn121_ila_continue')
    cases = []
    for method in ('fft', 'aaf', 'ila'):
        for source, target in (('rn50_v1', 'dn121'), ('rn50_v1', 'vgg16_bn'), ('dn121', 'rn50_v1'), ('dn121', 'vgg16_bn')):
            prefix = f'results/early_switch_dev276_v1/eval_{source}_{method}_s160_'
            evals = [load_success(prefix + s) for s in ('reset', 'keep')]
            raw = {r['id']: r for r in map(json.loads, open(prefix + 'reset/progress.jsonl', encoding='utf-8'))}
            flips = flips_by_label(evals[0], evals[1], target, {i: r['target_label'] for i, r in raw.items()})
            shared = sorted(label for label, ids in flips['inherit'].items() if len(ids) >= 2)
            if not shared:
                continue
            label = method.upper() if method != 'aaf' else 'AaF'
            for name, image in zip(('Image A', 'Image B'), flips['inherit'][shared[0]][:2]):
                cases.append({'title': f'{name}\n{label}, {NAMES[source]} → {NAMES[target]}', 'id': image,
                              'target': target, 'label': raw[image]['label'], 'target_label': raw[image]['target_label'],
                              'reset': (runs[(source, method)][0], evals[0]),
                              'inherit': (runs[(source, method)][1], evals[1])})
            n_inherit = sum(map(len, flips['inherit'].values()))
            n_reset = sum(map(len, flips['reset'].values()))
            print(f'Examples: {label} {source}->{target}; inherit-only {n_inherit}, reset-only {n_reset}')
            break
        if cases:
            break
    example_grid(cases, 'data/original299/manifest.json', out / 'fig_examples')
    ids = open('results/feature_dev64_v1_ids.txt', encoding='utf-8').read().split(',')
    v1, v2 = Path('results/partial_decay_dev64_v1'), Path('results/partial_decay_dev64_v2')
    mean = lambda path, model: sum(load_success(path)[(model, i)]['target_log_odds'] for i in ids) / len(ids)
    rn50 = [(0.0, v1 / 'eval_fft_g0'), (0.02, v2 / 'eval_rn50_fft_g0.02'), (0.05, v2 / 'eval_rn50_fft_g0.05'),
            (0.1, v2 / 'eval_rn50_fft_g0.1'), (0.25, v1 / 'eval_fft_g0.25'), (0.5, v1 / 'eval_fft_g0.5'),
            (0.75, v1 / 'eval_fft_g0.75'), (1.0, v1 / 'eval_fft_g1')]
    dn121 = [(g, v2 / f'eval_dn121_fft_g{label}') for g, label in ((0.0, '0'), (0.02, '0.02'), (0.05, '0.05'),
                                                                    (0.1, '0.1'), (1.0, '1'))]
    decay_curve({'RN50 → DN121': [(g, mean(p, 'dn121')) for g, p in rn50],
                 'DN121 → RN50': [(g, mean(p, 'rn50_v1')) for g, p in dn121]}, out / 'fig_partial_decay')
    print(f'Wrote figures to {out}')
