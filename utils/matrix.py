"""Frozen v1 diagnostic matrix: planning and serial resumable execution."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from .data import select_records
from .io import append_jsonl, read_json, sha256, write_json
from .models import ROOT

SEEDS = [20261027, 20261028, 20261029]
TARGETS = ['dn121', 'vit_b_16']
STAGES = ['b20-a1', 'classic', 'b200', 'b20-a2-s27', 'b20-a2-rest']


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def build_jobs(image_count=276):
    jobs = []

    def add(stage, budget, method, steps, views, alpha, seed):
        identity = f'{budget}_{method}_t{steps}_n{views}_a{alpha}_s{seed}'
        jobs.append({'id': identity, 'stage': stage, 'budget': budget, 'method': method,
                     'steps': steps, 'views': views, 'alpha_mult': alpha, 'seed': seed,
                     'epsilon_bytes': 8, 'alpha_bytes': 8 * alpha / steps,
                     'image_count': image_count, 'gradient_images_per_image': steps * views,
                     'planned_gradient_images': image_count * steps * views})

    for seed in SEEDS:
        for steps, views in [(20, 1), (10, 2), (5, 4)]:
            for method in ('di-n', 'sia'):
                add('b20-a1', 'B20', method, steps, views, 1, seed)
    add('classic', 'classic', 'mi', 10, 1, 1, SEEDS[0])
    for seed in SEEDS:
        add('classic', 'classic', 'di', 10, 1, 1, seed)
    for steps, views in [(10, 20), (40, 5)]:
        for method in ('di-n', 'sia'):
            add('b200', 'B200', method, steps, views, 1, SEEDS[0])
    for seed in SEEDS:
        stage = 'b20-a2-s27' if seed == SEEDS[0] else 'b20-a2-rest'
        for steps, views in [(20, 1), (10, 2), (5, 4)]:
            for method in ('di-n', 'sia'):
                add(stage, 'B20', method, steps, views, 2, seed)
    for index, job in enumerate(jobs, 1):
        job['order'] = index
    return jobs


def _namespace(**kwargs):
    return argparse.Namespace(**kwargs)


def matrix_command(args):
    from . import runner
    output = runner._output(args.output)
    manifest = Path(args.manifest).resolve()
    records = select_records(manifest, 'dev')
    if len(records) != 276:
        raise ValueError('Frozen diagnostic matrix requires exactly dev276 (pilot excluded)')
    protocol = Path(args.protocol).resolve()
    plan = {'schema_version': 1, 'protocol': str(protocol), 'protocol_sha256': sha256(protocol),
            'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
            'selected_ids': [r['id'] for r in records], 'split': 'dev', 'image_count': len(records),
            'source': 'rn50', 'targets': TARGETS, 'view_batch': 1, 'device': args.device,
            'save_float': args.save_float, 'code_sha256': runner._code_identity(),
            'stage_order': STAGES, 'seeds': SEEDS, 'bootstrap_seed': 20261030,
            'bootstrap_resamples': 10000, 'jobs': build_jobs(),
            'planned_total_gradient_images': sum(j['planned_gradient_images'] for j in build_jobs()),
            'statistics': 'image-level seed mean; paired IDs; CI conditional on fixed seeds; targets independent'}
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError('Existing frozen matrix plan differs; use a new output directory')
    else:
        write_json(plan_path, plan)
    if not args.execute:
        return {'plan': str(plan_path), 'jobs': len(plan['jobs']),
                'gradient_images': plan['planned_total_gradient_images'], 'executed': False}
    requested_stages = args.stages.split(',') if args.stages else STAGES
    if any(stage not in STAGES for stage in requested_stages):
        raise ValueError(f'--stages must be drawn from {STAGES}')
    if args.max_jobs is not None and args.max_jobs < 1:
        raise ValueError('--max-jobs must be positive')
    deadline = None
    if args.stop_starting_utc:
        deadline = datetime.fromisoformat(args.stop_starting_utc.replace('Z', '+00:00'))
        if deadline.tzinfo is None:
            raise ValueError('--stop-starting-utc requires an explicit UTC timezone')
        deadline = deadline.astimezone(timezone.utc)
    state_path = output / 'state.json'
    state = read_json(state_path) if state_path.exists() else {'plan_sha256': sha256(plan_path), 'jobs': {}, 'clean': {}}
    if state['plan_sha256'] != sha256(plan_path):
        raise ValueError('State does not belong to this plan')
    selection = dict(manifest=str(manifest), split='dev', allow_holdout=False, limit=None,
                     ids=None, device=args.device)

    def invoke(kind, identity, namespace, callback):
        container = state['clean'] if kind == 'clean' else state['jobs']
        entry = container.setdefault(identity, {'attempts': [], 'status': 'pending'})
        attempt = {'kind': kind, 'started_utc': utc_now(), 'status': 'running'}
        entry['attempts'].append(attempt)
        entry['status'] = 'running'
        write_json(state_path, state)
        append_jsonl(output / 'events.jsonl', {'id': identity, **attempt})
        start = time.perf_counter()
        try:
            result = callback(namespace)
            attempt['status'] = 'complete'
            entry['status'] = 'complete'
            return result
        except BaseException as error:
            attempt['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
            attempt['error'] = f'{type(error).__name__}: {error}'
            entry['status'] = attempt['status']
            raise
        finally:
            attempt['ended_utc'] = utc_now()
            attempt['wall_seconds'] = time.perf_counter() - start
            entry['total_attempt_wall_seconds'] = sum(a.get('wall_seconds', 0) for a in entry['attempts'])
            write_json(state_path, state)
            append_jsonl(output / 'events.jsonl', {'id': identity, **attempt})

    clean_output = output / 'clean'
    if state['clean'].get('dev', {}).get('status') != 'complete':
        invoke('clean', 'dev', _namespace(command='clean', **selection, models='rn50,dn121,vit_b_16',
                                        output=str(clean_output)), runner.clean_command)
    executed = 0
    for job in plan['jobs']:
        if job['stage'] not in requested_stages or state['jobs'].get(job['id'], {}).get('status') == 'complete':
            continue
        if (args.max_jobs is not None and executed >= args.max_jobs) or (deadline and datetime.now(timezone.utc) >= deadline):
            break
        attack_output = output / 'attacks' / job['id']
        evaluation_output = output / 'evaluations' / job['id']
        attack_args = _namespace(command='attack', **selection, output=str(attack_output),
                                 method=job['method'], source='rn50', steps=job['steps'], views=job['views'],
                                 epsilon_bytes=8., alpha_mult=float(job['alpha_mult']), seed=job['seed'],
                                 view_batch=1, di_probability=.5, resize_rate=1.1, save_float=args.save_float)
        evaluate_args = _namespace(command='evaluate', **selection, output=str(evaluation_output),
                                   attack_run=str(attack_output), targets='dn121,vit_b_16',
                                   population='target-clean', source_clean=str(clean_output))

        def group(_unused):
            # Each call resumes existing per-image records before finishing evaluation.
            runner.attack_command(attack_args)
            runner.evaluate_command(evaluate_args)

        invoke('job', job['id'], None, group)
        executed += 1
    return {'plan': str(plan_path), 'state': str(state_path), 'jobs_started_this_call': executed,
            'complete_jobs': sum(j.get('status') == 'complete' for j in state['jobs'].values()),
            'total_jobs': len(plan['jobs']), 'stop_starting_utc': args.stop_starting_utc}
