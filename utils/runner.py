"""Explicit experiment commands and append-only per-image progress."""
import hashlib
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

from baseline.attacks import attack
from baseline.attacks import mean_view_gradient
from baseline.transforms import make_plans
from .data import prepare, select_records
from .io import append_jsonl, image_tensor, read_json, read_jsonl, read_pixels, save_adversarial, sha256, write_json
from .models import ROOT, TORCH_HOME, load_model
from .metrics import image_seed, logit_metrics


def environment(args):
    import PIL
    import torchvision
    result = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'pillow': PIL.__version__,
              'cuda_available': torch.cuda.is_available(), 'cuda_build': torch.version.cuda,
              'torch_home': str(TORCH_HOME), 'devices': []}
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            result['devices'].append({'index': index, 'name': properties.name, 'memory_bytes': properties.total_memory,
                                      'capability': list(torch.cuda.get_device_capability(index))})
    if args.output:
        write_json(args.output, result)
    return result


def prepare_command(args):
    result = prepare(args.images_dir, args.labels_csv, args.output, args.label_offset,
                     args.id_column, args.label_column, args.expected_count, args.development_count, args.pilot_count)
    return {'manifest': str(Path(args.output).resolve() / 'manifest.json'), 'counts': result['counts']}


def _records(args):
    return select_records(args.manifest, args.split, args.allow_holdout, args.limit, args.ids)


def _output(path):
    output = Path(path).resolve()
    if not output.is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Experiment outputs must be inside project results/')
    output.mkdir(parents=True, exist_ok=True)
    return output


def _code_identity():
    paths = sorted(list((ROOT / 'baseline').glob('*.py')) + list((ROOT / 'utils').glob('*.py')))
    return {str(p.relative_to(ROOT)): sha256(p) for p in paths}


def _config(args, records, **extra):
    values = vars(args).copy()
    values.pop('handler', None)
    values.pop('output', None)
    return {'schema_version': 1, 'arguments': values,
            'manifest_path': str(Path(args.manifest).resolve()), 'manifest_sha256': sha256(args.manifest),
            'selected_ids': [r['id'] for r in records], 'code_sha256': _code_identity(),
            'numeric_settings': numeric_settings(), **extra}


def numeric_settings():
    return {'default_dtype': str(torch.get_default_dtype()), 'numpy_version': np.__version__, 'fp32': True, 'autocast': False,
            'cuda_matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
            'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_enabled': torch.backends.cudnn.enabled,
            'cudnn_version': torch.backends.cudnn.version(),
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'deterministic_warn_only': torch.is_deterministic_algorithms_warn_only_enabled(),
            'float32_matmul_precision': torch.get_float32_matmul_precision(),
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}


def _open_run(output, config):
    path = output / 'config.json'
    if path.exists():
        if read_json(path) != config:
            raise ValueError('Existing run configuration differs; use a new output directory')
    else:
        if any(output.iterdir()):
            raise ValueError('Nonempty output has no config; choose an empty directory')
        write_json(path, config)
    rows = read_jsonl(output / 'progress.jsonl')
    keys = [(r['id'], r.get('model')) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate completed progress rows')
    return rows


def _check_clean(record):
    if sha256(record['canonical_path']) != record['canonical_sha256']:
        raise ValueError(f'Canonical image hash changed: {record["id"]}')


def _device(name):
    device = torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    # FP32 even on modern GPUs; no TF32 approximation or mixed precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    return device


def clean_command(args):
    records, output, device = _records(args), _output(args.output), _device(args.device)
    identities = {}
    # Avoid simultaneous target models in 4GB VRAM.
    for name in args.models.split(','):
        model, identities[name] = load_model(name, 'cpu')
        del model
    config = _config(args, records, model_identities=identities)
    completed = _open_run(output, config)
    done = {(r['id'], r['model']) for r in completed}
    for name in identities:
        model, _ = load_model(name, device)
        for record in records:
            if (record['id'], name) in done:
                continue
            _check_clean(record)
            start = time.perf_counter()
            with torch.no_grad():
                logits = model(image_tensor(record['canonical_path'], device))
                prediction = int(logits.argmax(1).item())
                endpoint = logit_metrics(logits, record['label'])
            row = {'id': record['id'], 'split': record['split'], 'label': record['label'], 'model': name,
                   'prediction': prediction, 'correct': prediction == record['label'],
                   'elapsed_seconds': time.perf_counter() - start, **endpoint}
            append_jsonl(output / 'progress.jsonl', row)
            completed.append(row)
            print(f'clean {name} {record["id"]}: {prediction}', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    summary = {name: {'count': sum(r['model'] == name for r in completed),
                      'accuracy': float(np.mean([r['correct'] for r in completed if r['model'] == name]))}
               for name in identities}
    write_json(output / 'summary.json', summary)
    return summary


def attack_command(args):
    records, output, device = _records(args), _output(args.output), _device(args.device)
    model, identity = load_model(args.source, device)
    profile = {'mi': 'mi-fgsm_mu1_local224', 'di': 'momentum-di_p05_r11_local224',
               'di-n': 'mean-raw-gradient-di-n_mu1_local224', 'sia': 'sit-author-10ops_3x3_local224',
               'bsr': 'bsr-transferattack-3blocks', 'ops': 'ops-author-31ops-93pool',
               'ops-refresh': 'ops-author-noiserefresh', 'ops-view-l2': 'ops-fresh-perview-l2-wrtx-control'}[args.method]
    requested_profile = getattr(args, 'profile', None)
    if requested_profile is not None and requested_profile != profile:
        raise ValueError(f'Profile mismatch: expected {profile}')
    config = _config(args, records, source_identity=identity, profile=profile,
                     gradient_aggregation=('each raw dCE/doriginal_x unit L2 then equal mean26, mean-absolute normalize and momentum' if args.method=='ops-view-l2'
                                           else 'mean raw input gradients then mean-absolute normalize then momentum'),
                     effective_views_per_step=(1 + getattr(args, 'ne', 5) * getattr(args, 'np', 5)) if args.method in {'ops', 'ops-refresh', 'ops-view-l2'} else args.views,
                     alpha_bytes=args.alpha_mult * args.epsilon_bytes / args.steps,
                     perturbation_reference='canonical uint8 PNG; absolute-byte and [0,1] units; boundary denominator is channel elements',
                     perturbation_epsilon_bytes=args.epsilon_bytes,
                     perturbation_statistics={'png_mean_abs_bytes': 'mean(abs(uint8_adv-uint8_clean))',
                                              'png_mean_abs_unit': 'mean absolute byte difference /255',
                                              'png_l2_unit': 'L2(delta/255), not pixel-normalized',
                                              'png_boundary_element_fraction': 'fraction of channel elements with abs(byte delta)==epsilon'})
    completed = _open_run(output, config)
    done = {r['id']: r for r in completed}
    for record in records:
        image_id = record['id']
        destination = output / 'images' / f'{image_id}.png'
        if image_id in done:
            if not destination.exists() or sha256(destination) != done[image_id]['png_sha256']:
                raise ValueError(f'Completed output missing/changed: {image_id}')
            if args.save_float and not (output / 'float' / f'{image_id}.npz').exists():
                raise ValueError(f'Completed float output missing: {image_id}')
            continue
        _check_clean(record)
        clean = image_tensor(record['canonical_path'], device)
        labels = torch.tensor([record['label']], device=device)
        # Stable per-image seed makes resume and subset selection invariant.
        seed = image_seed(args.seed, image_id)
        actual_hook = {'forward_images': 0, 'forward_calls': 0}
        def count_forward(_module, inputs):
            actual_hook['forward_images'] += inputs[0].shape[0]
            actual_hook['forward_calls'] += 1
        handle = model.register_forward_pre_hook(count_forward) if args.method in {'ops', 'ops-refresh', 'ops-view-l2'} else None
        if args.method=='ops-view-l2':
            from baseline.norm_control import control_attack
            attack_fn=control_attack
        else:attack_fn=attack
        try:
            adv, stats = attack_fn(model, clean, labels, method=args.method, steps=args.steps, views=args.views,
                            epsilon_bytes=args.epsilon_bytes, alpha_mult=args.alpha_mult, seed=seed,
                            view_batch=args.view_batch, di_probability=args.di_probability, resize_rate=args.resize_rate,
                            ne=getattr(args, 'ne', 5), np_=getattr(args, 'np', 5), beta=getattr(args, 'beta', 2.), profile=profile if args.method in {'bsr', 'ops', 'ops-refresh', 'ops-view-l2'} else None)
        finally:
            if handle is not None:
                handle.remove()
        if args.method in {'ops', 'ops-refresh', 'ops-view-l2'}:
            if actual_hook['forward_images'] != stats['forward_images']:
                raise RuntimeError('Refresh actual forward hook count differs')
            stats['actual_forward_hook'] = actual_hook
        png_stats = save_adversarial(destination, adv, read_pixels(record['canonical_path']), args.epsilon_bytes)
        if args.save_float:
            directory = output / 'float'
            directory.mkdir(exist_ok=True)
            with open(directory / f'{image_id}.tmp', 'wb') as stream:
                np.savez_compressed(stream, adversarial=adv.detach().cpu().numpy())
            (directory / f'{image_id}.tmp').replace(directory / f'{image_id}.npz')
        row = {'id': image_id, 'split': record['split'], 'label': record['label'], 'image_seed': seed,
               'canonical_sha256': record['canonical_sha256'], **stats, **png_stats}
        append_jsonl(output / 'progress.jsonl', row)
        completed.append(row)
        print(f'attack {args.method} {image_id}: {stats["elapsed_seconds"]:.3f}s, '
              f'{stats["input_gradient_images"]} gradient images', flush=True)
    summary = {'count': len(completed), 'forward_images': sum(r['forward_images'] for r in completed),
               'input_gradient_images': sum(r['input_gradient_images'] for r in completed),
               'forward_pixels': sum(r['forward_pixels'] for r in completed),
               'input_gradient_pixels': sum(r['input_gradient_pixels'] for r in completed),
               'attack_seconds': sum(r['elapsed_seconds'] for r in completed),
               'peak_cuda_allocated_bytes': max(r['peak_cuda_allocated_bytes'] for r in completed),
               'zero_grad_count': sum(r['zero_grad_count'] for r in completed),
               'max_png_byte_difference': max(r['png_max_byte_difference'] for r in completed)}
    for metric in ('png_mean_abs_bytes', 'png_mean_abs_unit', 'png_l2_unit', 'png_boundary_element_fraction'):
        summary[metric] = float(np.mean([r[metric] for r in completed]))
    write_json(output / 'summary.json', summary)
    return summary


def evaluate_command(args):
    records, output, device = _records(args), _output(args.output), _device(args.device)
    run = Path(args.attack_run).resolve()
    attack_config = read_json(run / 'config.json')
    if sha256(args.manifest) != attack_config['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from attack manifest')
    generated = {r['id']: r for r in read_jsonl(run / 'progress.jsonl')}
    if any(r['id'] not in generated for r in records):
        raise ValueError('Selected evaluation images are not all completed by the attack')
    identities = {}
    for name in args.targets.split(','):
        model, identities[name] = load_model(name, 'cpu')
        del model
    source_clean = {}
    if args.population == 'source-target-clean' or args.source_clean:
        if not args.source_clean:
            raise ValueError('--source-clean run directory required for source-target-clean population')
        source_config = read_json(Path(args.source_clean) / 'config.json')
        if source_config['manifest_sha256'] != attack_config['manifest_sha256']:
            raise ValueError('Source clean manifest mismatch')
        source_name = attack_config['arguments']['source']
        if source_config['model_identities'].get(source_name) != attack_config['source_identity']:
            raise ValueError('Source clean model identity mismatch')
        source_clean = {r['id']: r['correct'] for r in read_jsonl(Path(args.source_clean) / 'progress.jsonl') if r['model'] == source_name}
        if any(r['id'] not in source_clean for r in records):
            raise ValueError('Source clean predictions missing selected IDs')
    config = _config(args, records, target_identities=identities,
                     attack_config_sha256=sha256(run / 'config.json'),
                     attack_progress_sha256=sha256(run / 'progress.jsonl'))
    completed = _open_run(output, config)
    done = {(r['id'], r['model']) for r in completed}
    for name in identities:
        model, _ = load_model(name, device)
        for record in records:
            if (record['id'], name) in done:
                continue
            _check_clean(record)
            adversarial_path = run / 'images' / f'{record["id"]}.png'
            if sha256(adversarial_path) != generated[record['id']]['png_sha256']:
                raise ValueError('Adversarial PNG hash mismatch')
            with torch.no_grad():
                clean_logits = model(image_tensor(record['canonical_path'], device))
                adv_logits = model(image_tensor(adversarial_path, device))
                clean_pred = int(clean_logits.argmax(1).item())
                adv_pred = int(adv_logits.argmax(1).item())
                clean_endpoint = logit_metrics(clean_logits, record['label'])
                adv_endpoint = logit_metrics(adv_logits, record['label'])
            clean_correct = clean_pred == record['label']
            eligible = args.population == 'all' or (clean_correct and
                       (args.population != 'source-target-clean' or source_clean[record['id']]))
            row = {'id': record['id'], 'split': record['split'], 'label': record['label'], 'model': name,
                   'clean_prediction': clean_pred, 'adversarial_prediction': adv_pred,
                   'clean_correct': clean_correct, 'adversarial_correct': adv_pred == record['label'],
                   'eligible': eligible, 'success': adv_pred != record['label'],
                   'source_clean_correct': source_clean.get(record['id']),
                   'source_target_clean_eligible': clean_correct and source_clean[record['id']] if source_clean else None,
                   **{f'clean_{key}': value for key, value in clean_endpoint.items()},
                   **{f'adversarial_{key}': value for key, value in adv_endpoint.items()}}
            append_jsonl(output / 'progress.jsonl', row)
            completed.append(row)
            print(f'evaluate {name} {record["id"]}', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    summary = {}
    for name in identities:
        rows = [r for r in completed if r['model'] == name]
        eligible = [r for r in rows if r['eligible']]
        target_clean = [r for r in rows if r['clean_correct']]
        common_clean = [r for r in rows if r.get('source_target_clean_eligible')]
        summary[name] = {'count': len(rows), 'clean_accuracy': float(np.mean([r['clean_correct'] for r in rows])),
                         'adversarial_accuracy_all': float(np.mean([r['adversarial_correct'] for r in rows])),
                         'target_clean_count': len(target_clean),
                         'asr_target_clean': float(np.mean([r['success'] for r in target_clean])) if target_clean else None,
                         'target_clean_success_count': sum(r['success'] for r in target_clean),
                         'source_target_clean_count': len(common_clean) if source_clean else None,
                         'asr_source_target_clean': float(np.mean([r['success'] for r in common_clean])) if common_clean else None,
                         'all_success_rate': float(np.mean([r['success'] for r in rows])),
                         'mean_adversarial_margin_target_clean': float(np.mean([r['adversarial_margin'] for r in target_clean])) if target_clean else None,
                         'mean_adversarial_ce_target_clean': float(np.mean([r['adversarial_ce'] for r in target_clean])) if target_clean else None,
                         'selected_population': args.population, 'eligible_count': len(eligible),
                         'selected_population_success_rate': float(np.mean([r['success'] for r in eligible])) if eligible else None}
    write_json(output / 'summary.json', summary)
    return summary


def gradient_check_command(args):
    """Real-source engineering check with exactly the same sampled view plans."""
    if args.method == 'ops-refresh':
        from .ops_profile import gradient_check_refresh
        return gradient_check_refresh(args)
    import random
    records, output, device = _records(args), _output(args.output), _device(args.device)
    if args.views < 1:
        raise ValueError('--views must be positive')
    expected_profile = {'bsr': 'bsr-transferattack-3blocks', 'ops': 'ops-author-31ops-93pool'}.get(args.method)
    if getattr(args, 'profile', None) is not None and args.profile != expected_profile:
        raise ValueError('gradient-check profile does not match the selected method')
    model, identity = load_model(args.source, device)
    config = _config(args, records, source_identity=identity,
                     purpose='fixed-plan source input-gradient microbatch engineering check',
                     compared_view_batches=[1, 2])
    completed = _open_run(output, config)
    done = {row['id'] for row in completed}
    for record in records:
        if record['id'] in done:
            continue
        _check_clean(record)
        clean = image_tensor(record['canonical_path'], device)
        labels = torch.tensor([record['label']], device=device)
        image_seed = int.from_bytes(hashlib.sha256(f'{args.seed}:{record["id"]}'.encode()).digest()[:8], 'big') % (2**63)
        if args.method == 'ops':
            from baseline.ops import OPSContext
            plans = OPSContext(clean.shape, args.epsilon_bytes / 255, args.ne, args.np, args.beta, image_seed).step_plans()
        else:
            plans = make_plans(args.method, args.views, random.Random(image_seed), args.di_probability, args.resize_rate)
        gradients, counters = [], []
        for batch_size in (1, 2):
            counter = {'forward_images': 0, 'input_gradient_images': 0, 'forward_pixels': 0,
                       'input_gradient_pixels': 0, 'resolution_counts': {}, 'view_batch': batch_size}
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            gradient = mean_view_gradient(model, clean, labels, plans, batch_size, counter)
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            counter['elapsed_seconds'] = time.perf_counter() - start
            counter['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
            gradients.append(gradient.detach().cpu().double().flatten())
            counters.append(counter)
            del gradient
        first, second = gradients
        norm_first, norm_second = first.norm().item(), second.norm().item()
        difference = second - first
        cosine = float(torch.dot(first, second).item() / (norm_first * norm_second)) if norm_first and norm_second else None
        row = {'id': record['id'], 'split': record['split'], 'label': record['label'],
               'image_seed': image_seed, 'plan_seeds': [getattr(p, 'seed', None) for p in plans],
               'ops_pool_indices': [getattr(p, 'pool_index', None) for p in plans] if args.method == 'ops' else None,
               'view_batch_1': counters[0], 'view_batch_2': counters[1],
               'cosine': cosine, 'max_abs_error': difference.abs().max().item(),
               'relative_l2_error': difference.norm().item() / norm_first if norm_first else None,
               'gradient_l2_view_batch_1': norm_first, 'gradient_l2_view_batch_2': norm_second,
               'sign_agreement': (first.sign() == second.sign()).double().mean().item(),
               'both_zero': norm_first == 0 and norm_second == 0}
        append_jsonl(output / 'progress.jsonl', row)
        completed.append(row)
        print(f'gradient-check {record["id"]}: cosine={cosine}, relative_l2={row["relative_l2_error"]}', flush=True)
    summary = {'count': len(completed), 'checks': completed}
    write_json(output / 'summary.json', summary)
    return summary


def report_command(args):
    rows = []
    for directory in args.runs:
        path = Path(directory).resolve()
        rows.append({'run': str(path), 'summary': read_json(path / 'summary.json')})
    output = Path(args.output).resolve()
    if not output.is_relative_to((ROOT / 'results').resolve()):
        raise ValueError('Reports must be inside project results/')
    write_json(output, rows)
    return rows
