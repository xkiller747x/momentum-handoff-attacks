"""Read-only source observation of existing OPS PNGs and finite noise pools.

No attack generation, target model or input-gradient computation occurs here.
The existing OPS constructor and transformation implementations are used intact.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from baseline.ops import OPSContext, OPSViewPlan, RATIOS, apply_ops, composite_pool, plan_basic
from .data import select_records
from .io import append_jsonl, image_tensor, read_json, read_jsonl, sha256, write_json
from .metrics import image_seed, logit_metrics, paired_bootstrap
from .models import ROOT, load_model

K = 2
FRESH_SEEDS = (20261031, 20261032, 20261033)
OPERATOR_SEED = 20261101
BOOTSTRAP_SEED = 20261102
FORWARDS_PER_IMAGE = 4 * 2 * 30 * K + 2
REQUIRED_KERNELS = ('baseline/ops.py', 'utils/models.py', 'utils/data.py', 'utils/io.py', 'utils/metrics.py')


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _time(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00')).astimezone(timezone.utc)


def namespace_seed(namespace, seed, identity, *parts):
    key = ':'.join(map(str, ('ops-pool-observation-v1', namespace, seed, identity, *parts)))
    return int.from_bytes(hashlib.sha256(key.encode('utf-8')).digest()[:8], 'big') % (2**63)


def _tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def reconstruct_pools(identity, original_seed, shape=(1, 3, 224, 224)):
    """Reconstruct exact fit pool and replay all training choices without a model."""
    contexts = [OPSContext(shape, 8 / 255, 5, 5, 2., original_seed)]
    seeds = [original_seed] + [namespace_seed('fresh-pool', seed, identity) for seed in FRESH_SEEDS]
    contexts.extend(OPSContext(shape, 8 / 255, 5, 5, 2., seed) for seed in seeds[1:])
    if len(set(seeds)) != 4:
        raise ValueError('Probe pool seed collision')
    usage, last_step, trajectory = [0] * 30, [None] * 30, []
    for step in range(1, 11):
        views = contexts[0].step_plans()[1:]
        indices = list(dict.fromkeys(view.pool_index for view in views))
        if len(views) != 25 or len(indices) != 5 or any(sum(v.pool_index == index for v in views) != 5 for index in indices):
            raise ValueError('Original OPS neighborhood usage trajectory is invalid')
        trajectory.append(indices)
        for index in indices:
            usage[index] += 1
            last_step[index] = step
    if sum(usage) != 50:
        raise ValueError('Original OPS usage must total50, not250 operator views')
    metadata = []
    for index, (context, seed) in enumerate(zip(contexts, seeds)):
        hashes = [_tensor_hash(noise) for noise in context.pool]
        digest = hashlib.sha256()
        for noise in context.pool:
            digest.update(noise.contiguous().numpy().tobytes())
        metadata.append({'name': 'fit' if index == 0 else f'fresh{index}', 'constructor_seed': seed,
                         'namespace_seed': None if index == 0 else FRESH_SEEDS[index - 1],
                         'pool_sha256': digest.hexdigest(), 'noise_sha256': hashes,
                         'shape': list(shape), 'dtype': 'float32', 'count': len(context.pool)})
    return [context.pool for context in contexts], {'pools': metadata, 'usage_count': usage,
            'last_step': last_step, 'training_neighbor_indices_by_step': trajectory,
            'usage_sum': sum(usage)}


def evaluation_operators(identity, shape=(1, 3, 224, 224)):
    """One93-composite pool per noise, two selections, full shared parameters."""
    plans, records = [], []
    for index in range(30):
        selection_seed = namespace_seed('evaluation-composite-pool', OPERATOR_SEED, identity, index)
        rng = random.Random(selection_seed)
        pool = composite_pool(rng)
        selected_indices = rng.sample(range(len(pool)), K)
        paired, descriptions = [], []
        for k, composite_index in enumerate(selected_indices):
            parameter_seed = namespace_seed('evaluation-basic-parameters', OPERATOR_SEED, identity, index, k)
            parameter_rng = random.Random(parameter_seed)
            working_shape, operations = tuple(shape), []
            for basic_index in reversed(pool[composite_index]):
                operations.append(plan_basic(basic_index, working_shape, parameter_rng))
                if basic_index >= 21:
                    working_shape = (*working_shape[:-2], working_shape[-1], working_shape[-1])
            operators = tuple(reversed(operations))
            paired.append(operators)
            descriptions.append({'k': k, 'composite_pool_index': composite_index, 'parameter_seed': parameter_seed,
                                 'operators_outer_to_inner': [asdict(op) for op in operators]})
        plans.append(tuple(paired))
        records.append({'pool_index': index, 'selection_seed': selection_seed, 'level_counts': [31, 31, 31],
                        'selected': descriptions})
    serialized = json.dumps(records, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return plans, {'namespace': OPERATOR_SEED, 'parameters': records,
                   'plan_sha256': hashlib.sha256(serialized).hexdigest(),
                   'shared_across': 'fit/fresh1/fresh2/fresh3 and clean/PNG; no resampling between states/pools'}


def endpoint_values(pool_state_values):
    """Input shape4pools×2states, clean first. Statistical unit remains image."""
    values = np.asarray(pool_state_values, dtype=np.float64)
    if values.shape != (4, 2) or not np.isfinite(values).all():
        raise ValueError('Expected finite4×2 pool-state scalar means')
    fit_clean, fit_png = values[0]
    fresh_clean, fresh_png = values[1:].mean(axis=0)
    g, c, a = fit_png - fresh_png, fit_clean - fresh_clean, fresh_png - fresh_clean
    placebo = []
    for index in range(1, 4):
        others = np.delete(values[1:], index - 1, axis=0).mean(axis=0)
        gaps = values[index] - others
        placebo.append({'fresh_pool': f'fresh{index}', 'clean': float(gaps[0]), 'png': float(gaps[1]),
                        'gain_difference': float(gaps[1] - gaps[0])})
    return {'D': float(g - c), 'G': float(g), 'C': float(c), 'A': float(a),
            'pool_state_means': values.tolist(), 'fresh_placebo': placebo}


def measure_image(model, clean, png, label, pools, operator_plans):
    """Exactly482 no-grad source forwards; hooks count actual model inputs."""
    counter = {'forward_images': 0, 'forward_calls': 0, 'input_gradient_images': 0, 'target_forward_images': 0,
               'forward_pixels': 0, 'resolution_counts': {}, 'all_no_grad': True}
    def count(_module, inputs):
        x = inputs[0]
        counter['forward_calls'] += 1
        counter['forward_images'] += int(x.shape[0])
        counter['forward_pixels'] += int(x.shape[0] * x.shape[-2] * x.shape[-1])
        size = f'{x.shape[-2]}x{x.shape[-1]}'
        counter['resolution_counts'][size] = counter['resolution_counts'].get(size, 0) + int(x.shape[0])
        counter['all_no_grad'] = counter['all_no_grad'] and not torch.is_grad_enabled() and not x.requires_grad
    hook = model.register_forward_pre_hook(count)
    ce, margin, details = np.empty((4, 2, 30)), np.empty((4, 2, 30)), []
    model.eval()
    ordinary = {}
    try:
        with torch.no_grad():
            for state_name, image in [('clean', clean), ('png', png)]:
                logits = model(image)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('Nonfinite ordinary-view source logits')
                ordinary[state_name] = logit_metrics(logits, label)
                ordinary[state_name]['prediction'] = int(logits.argmax(1).item())
            for pool_index, noises in enumerate(pools):
                for state_index, image in enumerate((clean, png)):
                    for noise_index, noise in enumerate(noises):
                        endpoints = []
                        for operators in operator_plans[noise_index]:
                            transformed = apply_ops(image, OPSViewPlan(noise, operators, noise_index))
                            logits = model(transformed)
                            if not torch.isfinite(logits).all():
                                raise FloatingPointError('Nonfinite transformed-view source logits')
                            endpoints.append(logit_metrics(logits, label))
                        ce[pool_index, state_index, noise_index] = np.mean([e['ce'] for e in endpoints])
                        margin[pool_index, state_index, noise_index] = np.mean([e['margin'] for e in endpoints])
                        details.append({'pool': 'fit' if pool_index == 0 else f'fresh{pool_index}',
                            'state': 'clean' if state_index == 0 else 'png', 'pool_index': noise_index,
                            'radius_index': noise_index // 5, 'radius_ratio': RATIOS[noise_index // 5],
                            'ce_K_mean': float(ce[pool_index, state_index, noise_index]),
                            'margin_K_mean': float(margin[pool_index, state_index, noise_index])})
    finally:
        hook.remove()
    if counter['forward_images'] != FORWARDS_PER_IMAGE or not counter['all_no_grad']:
        raise RuntimeError('Actual hook count/no_grad check failed')
    residual = (ce[0, 1] - ce[0, 0]) - (ce[1:, 1] - ce[1:, 0]).mean(axis=0)
    return {'ce_endpoints': endpoint_values(ce.mean(axis=2)), 'margin_descriptive': endpoint_values(margin.mean(axis=2)),
            'ordinary': ordinary, 'noise_state_details': details, 'noise_residual_d': residual.tolist(),
            'actual_counter': counter}


def _hash_map(mapping):
    return {key.replace('\\', '/'): value.lower() for key, value in mapping.items()}


def validate_metadata(related_run, protocol, freeze_path, engineering=False):
    """Metadata-only validation, deliberately without opening images or a model."""
    related_run, protocol, freeze_path = Path(related_run).resolve(), Path(protocol).resolve(), Path(freeze_path).resolve()
    related_plan = read_json(related_run / 'plan.json')
    freeze = read_json(freeze_path)
    if sha256(protocol) != freeze['protocol_sha256'] or sha256(related_run / 'plan.json') != freeze['related_plan_sha256']:
        raise ValueError('Frozen pool protocol/related plan SHA mismatch')
    previous = _hash_map(related_plan['code_sha256'])
    current = {}
    for relative in REQUIRED_KERNELS:
        current[relative] = sha256(ROOT / relative)
        if current[relative] != previous.get(relative):
            raise ValueError(f'Original kernel changed: {relative}')
    if current['baseline/ops.py'] != freeze['original_ops_file_sha256']:
        raise ValueError('OPS kernel differs from frozen reconstruction source')
    manifest = Path(related_plan['manifest'])
    if sha256(manifest) != related_plan['manifest_sha256']:
        raise ValueError('Manifest SHA mismatch')
    records = select_records(manifest, 'dev')
    ids = [r['id'] for r in records]
    if len(ids) != 276 or ids != related_plan['selected_ids'] or related_plan['blocks'] != [ids[i:i + 69] for i in range(0, 276, 69)]:
        raise ValueError('Frozen dev276 ordering/block definition differs')
    reference = related_plan['clean_reference']
    clean_root = Path(reference['path'])
    if sha256(clean_root / 'config.json') != reference['config_sha256'] or sha256(clean_root / 'progress.jsonl') != reference['progress_sha256']:
        raise ValueError('Frozen clean reference changed')
    clean_config = read_json(clean_root / 'config.json')
    clean_rows = [r for r in read_jsonl(clean_root / 'progress.jsonl') if r['model'] == 'rn50']
    clean = {r['id']: r for r in clean_rows}
    if set(clean) != set(ids) or sum(clean[identity]['correct'] for identity in ids) != 267:
        raise ValueError('Source-clean fixed267/276 population mismatch')
    if any(clean[record['id']]['label'] != record['label'] for record in records):
        raise ValueError('Source clean labels differ from frozen manifest')
    selected = records[:2] if engineering else records
    result = {'protocol': str(protocol), 'protocol_sha256': sha256(protocol), 'freeze': str(freeze_path),
              'freeze_sha256': sha256(freeze_path), 'related_run': str(related_run),
              'related_plan_sha256': sha256(related_run / 'plan.json'), 'kernel_sha256': current,
              'manifest': str(manifest), 'manifest_sha256': sha256(manifest), 'clean_reference': reference,
              'source_identity': clean_config['model_identities']['rn50'],
              'mode': 'engineering_first2' if engineering else 'formal276', 'selected_ids': [r['id'] for r in selected],
              'formal_all_ids': ids, 'source_clean_ids': [identity for identity in ids if clean[identity]['correct']],
              'blocks': [[r['id'] for r in selected]] if engineering else related_plan['blocks'],
              'planned_forward_images': len(selected) * FORWARDS_PER_IMAGE,
              'planned_forward_images_per_image': FORWARDS_PER_IMAGE, 'planned_input_gradient_images': 0,
              'planned_target_forward_images': 0, 'fresh_namespace_seeds': list(FRESH_SEEDS),
              'operator_namespace_seed': OPERATOR_SEED, 'K': K, 'bootstrap_seed': BOOTSTRAP_SEED,
              'last_gpu_start_utc': freeze['last_gpu_start_utc'], 'window_deadline_utc': freeze['window_deadline_utc'],
              'block_boundary_runtime_seconds': 1800}
    return result, related_plan, selected, clean


def validate_completed_results(plan, related_plan, records):
    """Gate execution on all20 completed tasks, then verify only OPS inputs."""
    related = Path(plan['related_run'])
    state = read_json(related / 'state.json')
    if state['plan_sha256'] != plan['related_plan_sha256']:
        raise ValueError('Related completion state belongs to another plan')
    tasks = related_plan['tasks']
    if len(tasks) != 20 or any(state['tasks'].get(t['id'], {}).get('status') != 'complete' for t in tasks):
        raise ValueError('Pool observation requires all20 related tasks complete, not just process exit0')
    # Completion is checked against per-image persisted records, not status alone.
    for task in tasks:
        expected = set(task['selected_ids'])
        attack_rows = read_jsonl(related / 'attacks' / task['id'] / 'progress.jsonl')
        evaluations = read_jsonl(related / 'evaluations' / task['id'] / 'progress.jsonl')
        if len(attack_rows) != 69 or {r['id'] for r in attack_rows} != expected:
            raise ValueError('Completed task attack rows are not a full predefined69-image block')
        for target in related_plan['targets']:
            target_rows = [r for r in evaluations if r['model'] == target]
            if len(target_rows) != 69 or {r['id'] for r in target_rows} != expected:
                raise ValueError('Completed task evaluation rows are incomplete')
    inputs = {}
    for task in tasks:
        if task['configuration'] != 'ops260':
            continue
        root = related / 'attacks' / task['id']
        config = read_json(root / 'config.json')
        arguments = config['arguments']
        wanted = {'method': 'ops', 'source': 'rn50', 'steps': 10, 'views': 1, 'ne': 5, 'np': 5,
                  'beta': 2., 'epsilon_bytes': 8., 'alpha_mult': 1., 'seed': 20261027, 'view_batch': 1,
                  'profile': 'ops-author-31ops-93pool'}
        if any(arguments.get(key) != value for key, value in wanted.items()) or config['selected_ids'] != task['selected_ids']:
            raise ValueError('Existing OPS260 parameters differ from frozen probe protocol')
        if config['manifest_sha256'] != plan['manifest_sha256'] or config['source_identity'] != plan['source_identity']:
            raise ValueError('Existing OPS260 manifest/source weights differ')
        if any(_hash_map(config['code_sha256']).get(key) != value for key, value in plan['kernel_sha256'].items()):
            raise ValueError('Existing OPS260 core kernel versions differ')
        for row in read_jsonl(root / 'progress.jsonl'):
            if row['image_seed'] != image_seed(20261027, row['id']):
                raise ValueError('Original OPS per-image seed mismatch')
            if row['forward_images'] != 260 or row['input_gradient_images'] != 260 or row['png_max_byte_difference'] > 8:
                raise ValueError('Original OPS actual260-image cost/EPS8 record mismatch')
            inputs[row['id']] = {'png': str(root / 'images' / f'{row["id"]}.png'),
                                 'png_sha256': row['png_sha256'], 'image_seed': row['image_seed'],
                                 'label': row['label'], 'canonical_sha256': row['canonical_sha256'],
                                 'attack_config_sha256': sha256(root / 'config.json')}
    for record in records:
        original = inputs[record['id']]
        if original['label'] != record['label'] or original['canonical_sha256'] != record['canonical_sha256']:
            raise ValueError('Original OPS ID/label/canonical identity mismatch')
    return inputs


def pool_diagnostic_command(args):
    from .runner import _code_identity, _device, _output, numeric_settings
    command_start, command_started_utc = time.perf_counter(), _utc()
    if args.limit not in (None, 2):
        raise ValueError('--limit is restricted to2 for the independent engineering run; omit for formal276')
    output = _output(args.output)
    plan, related_plan, records, clean_reference = validate_metadata(args.related_run, args.protocol, args.freeze, args.limit == 2)
    plan.update({'device': args.device, 'observation_code_sha256': _code_identity()})
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError('Pool observation plan differs; use an independent output directory')
    else:
        write_json(plan_path, plan)
    if not args.execute:
        return {'plan': str(plan_path), 'mode': plan['mode'], 'forward_images': plan['planned_forward_images'], 'executed': False}
    now = datetime.now(timezone.utc)
    if now > _time(plan['last_gpu_start_utc']):
        raise ValueError('Latest permitted GPU observation start time has passed')
    original_inputs = validate_completed_results(plan, related_plan, records)
    checkpoint = Path(plan['source_identity']['checkpoint'])
    if sha256(checkpoint) != plan['source_identity']['checkpoint_sha256']:
        raise ValueError('Source checkpoint SHA mismatch')
    device = _device(args.device)
    model, identity = load_model('rn50', device)
    if identity != plan['source_identity']:
        raise ValueError('Loaded source model identity differs from original OPS source')
    config = {'plan_sha256': sha256(plan_path), 'numeric_settings': numeric_settings(), 'source_identity': identity,
              'observation': 'existing PNG only; no_grad; no attack regeneration; no target models'}
    config_path = output / 'config.json'
    if config_path.exists() and read_json(config_path) != config:
        raise ValueError('Observation runtime configuration differs')
    write_json(config_path, config)
    completed = read_jsonl(output / 'progress.jsonl')
    if len({r['id'] for r in completed}) != len(completed) or any(r['id'] not in plan['selected_ids'] for r in completed):
        raise ValueError('Invalid observation progress IDs')
    done = {r['id']: r for r in completed}
    for row in completed:
        if sha256(output / row['detail_path']) != row['detail_sha256']:
            raise ValueError('Committed observation detail hash mismatch')
    runtime_path = output / 'runtime.json'
    runtime = read_json(runtime_path) if runtime_path.exists() else {'attempts': [], 'blocks': {}}
    attempt = {'started_utc': command_started_utc, 'status': 'running'}
    runtime['attempts'].append(attempt)
    write_json(runtime_path, runtime)
    start = command_start
    active_block, block_checkpoint = None, None
    try:
        record_map = {r['id']: r for r in records}
        for block_index, ids in enumerate(plan['blocks'], 1):
            if all(identity in done for identity in ids):
                continue
            total_elapsed = max(sum(a.get('wall_seconds', 0) for a in runtime['attempts']) + time.perf_counter() - start,
                                sum(b.get('attempt_seconds', 0) for b in runtime['blocks'].values()))
            if (total_elapsed >= 1800 or datetime.now(timezone.utc) >= _time(plan['window_deadline_utc']) or
                    runtime.get('first_block_projected_over_limit')):
                attempt['stop_reason'] = 'predeclared block-boundary time/projection limit'
                break
            block_start = time.perf_counter()
            block_entry = runtime['blocks'].setdefault(str(block_index), {'attempt_seconds': 0., 'started_utc': _utc()})
            active_block, block_checkpoint = block_entry, block_start
            for identity_id in ids:
                if identity_id in done:
                    continue
                image_start = time.perf_counter()
                record, original = record_map[identity_id], original_inputs[identity_id]
                if sha256(record['canonical_path']) != record['canonical_sha256'] or sha256(original['png']) != original['png_sha256']:
                    raise ValueError('Canonical/OPS PNG content SHA mismatch')
                clean, png = image_tensor(record['canonical_path'], device), image_tensor(original['png'], device)
                if clean.shape != (1, 3, 224, 224) or png.shape != clean.shape or (png - clean).abs().max().item() > 8 / 255 + 1e-6:
                    raise ValueError('Canonical224/OPS EPS8 PNG boundary mismatch')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                pools, pool_metadata = reconstruct_pools(identity_id, original['image_seed'], clean.shape)
                operators, operator_metadata = evaluation_operators(identity_id, clean.shape)
                measurement = measure_image(model, clean, png, record['label'], pools, operators)
                if measurement['ordinary']['clean']['prediction'] != clean_reference[identity_id]['prediction']:
                    raise ValueError('Recomputed ordinary clean prediction differs from frozen source reference')
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
                detail = {'id': identity_id, 'label': record['label'], 'block': block_index,
                          'source_clean_correct': clean_reference[identity_id]['correct'],
                          'canonical_sha256': record['canonical_sha256'], **original,
                          'pool_metadata': pool_metadata, 'operator_metadata': operator_metadata, **measurement,
                          'elapsed_seconds': time.perf_counter() - image_start,
                          'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                          'committed_utc': _utc()}
                detail_path = Path('details') / f'{identity_id}.json'
                write_json(output / detail_path, detail)
                row = {'id': identity_id, 'block': block_index, 'source_clean_correct': clean_reference[identity_id]['correct'],
                       **{key: measurement['ce_endpoints'][key] for key in ('D', 'G', 'C', 'A')},
                       'forward_images': measurement['actual_counter']['forward_images'], 'input_gradient_images': 0,
                       'target_forward_images': 0, 'all_no_grad': measurement['actual_counter']['all_no_grad'],
                       'elapsed_seconds': detail['elapsed_seconds'], 'committed_utc': detail['committed_utc'],
                       'detail_path': str(detail_path), 'detail_sha256': sha256(output / detail_path)}
                append_jsonl(output / 'progress.jsonl', row)
                done[identity_id] = row
                block_entry['attempt_seconds'] += time.perf_counter() - block_checkpoint
                block_checkpoint = time.perf_counter()
                write_json(runtime_path, runtime)
                print(f'pool-diagnostic {identity_id}:482 source forwards,0 gradients', flush=True)
                del pools, clean, png
            block_entry['attempt_seconds'] += time.perf_counter() - block_checkpoint
            active_block, block_checkpoint = None, None
            block_entry['complete'] = all(identity_id in done for identity_id in ids)
            block_entry['ended_utc'] = _utc()
            if block_index == 1 and plan['mode'] == 'formal276' and block_entry['complete']:
                projected = block_entry['attempt_seconds'] * 4
                runtime['first_block_projected_seconds'] = projected
                runtime['first_block_projected_over_limit'] = projected > 1800
            write_json(runtime_path, runtime)
        attempt['status'] = 'complete' if len(done) == len(records) else 'partial'
    except BaseException as error:
        attempt['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
        attempt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if active_block is not None:
            active_block['attempt_seconds'] += time.perf_counter() - block_checkpoint
            active_block['complete'] = False
            active_block['ended_utc'] = _utc()
        attempt['ended_utc'] = _utc()
        attempt['wall_seconds'] = time.perf_counter() - start
        write_json(runtime_path, runtime)
    return {'status': attempt['status'], 'mode': plan['mode'], 'completed_images': len(done),
            'planned_images': len(records), 'forward_images': sum(r['forward_images'] for r in done.values()),
            'input_gradient_images': 0, 'output': str(output)}
