"""Predeclared 69-image blocks for recent baseline anchors; no holdout access."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time

from .data import select_records
from .io import append_jsonl, read_json, read_jsonl, sha256, write_json
from .matrix import utc_now

CONFIGURATIONS = {
    'ops260': {'method': 'ops', 'steps': 10, 'views': 1, 'ne': 5, 'np': 5, 'beta': 2.,
               'profile': 'ops-author-31ops-93pool', 'gradient_images_per_image': 260},
    'bsr200': {'method': 'bsr', 'steps': 10, 'views': 20, 'profile': 'bsr-transferattack-3blocks', 'gradient_images_per_image': 200},
    'din260': {'method': 'di-n', 'steps': 10, 'views': 26, 'profile': None, 'gradient_images_per_image': 260},
    'sia260': {'method': 'sia', 'steps': 10, 'views': 26, 'profile': None, 'gradient_images_per_image': 260},
    'bsr260': {'method': 'bsr', 'steps': 10, 'views': 26, 'profile': 'bsr-transferattack-3blocks', 'gradient_images_per_image': 260},
}


def build_related_tasks(ids):
    if len(ids) != 276 or len(set(ids)) != 276:
        raise ValueError('Related matrix requires exactly 276 unique development IDs')
    blocks = [list(ids[start:start + 69]) for start in range(0, 276, 69)]
    tasks = []
    for stage, names in [('s2', ['ops260', 'bsr200']), ('s3', ['din260', 'sia260', 'bsr260'])]:
        for block, selected in enumerate(blocks, 1):
            for name in names:
                tasks.append({'id': f'{stage}_b{block}_{name}', 'stage': stage, 'block': block,
                              'configuration': name, 'selected_ids': selected, **CONFIGURATIONS[name],
                              'epsilon_bytes': 8, 'alpha_mult': 1., 'seed': 20261027,
                              'planned_gradient_images': 69 * CONFIGURATIONS[name]['gradient_images_per_image']})
    for order, task in enumerate(tasks, 1):
        task['order'] = order
    return blocks, tasks


def validate_clean_run(path, manifest_sha, ids):
    path = Path(path).resolve()
    config = read_json(path / 'config.json')
    if config['manifest_sha256'] != manifest_sha:
        raise ValueError('Existing clean run uses another manifest')
    predictions = {(r['model'], r['id']) for r in read_jsonl(path / 'progress.jsonl')}
    if any((model, identity) not in predictions for model in ('rn50', 'dn121', 'vit_b_16') for identity in ids):
        raise ValueError('Clean run must include source and both targets for every development image')
    return {'path': str(path), 'config_sha256': sha256(path / 'config.json'),
            'progress_sha256': sha256(path / 'progress.jsonl')}


def related_matrix_command(args):
    from . import runner
    output = runner._output(args.output)
    manifest = Path(args.manifest).resolve()
    records = select_records(manifest, 'dev')
    ids = [r['id'] for r in records]
    blocks, tasks = build_related_tasks(ids)
    clean_identity = validate_clean_run(args.clean_run, sha256(manifest), ids)
    protocol = Path(args.protocol).resolve()
    versions = {'bsr_commit': 'ae960f546a907a193931a2df12afd0f3fef2e0a5',
                'ops_commit': '7914115d86d1add0f09c3a0720d12961e14a0c25'}
    plan = {'schema_version': 1, 'manifest': str(manifest), 'manifest_sha256': sha256(manifest),
            'protocol': str(protocol), 'protocol_sha256': sha256(protocol), 'selected_ids': ids,
            'blocks': blocks, 'tasks': tasks, 'configurations': CONFIGURATIONS,
            'source': 'rn50', 'targets': ['dn121', 'vit_b_16'], 'split': 'dev', 'seed': 20261027,
            'view_batch': 1, 'device': args.device, 'clean_reference': clean_identity,
            'save_float': args.save_float, 'code_sha256': runner._code_identity(), 'upstream_versions': versions,
            'planned_gradient_images': sum(task['planned_gradient_images'] for task in tasks),
            'analysis': 'Only complete predefined69-image blocks; S3 requires OPS plus all three controls in a block; '
                        'single-seed image-paired bootstrap10000 seed20261030'}
    plan_path = output / 'plan.json'
    if plan_path.exists():
        if read_json(plan_path) != plan:
            raise ValueError('Frozen related plan differs; use another output directory')
    else:
        write_json(plan_path, plan)
    if not args.execute:
        return {'plan': str(plan_path), 'tasks': len(tasks), 'gradient_images': plan['planned_gradient_images'], 'executed': False}
    stages = args.stages.split(',') if args.stages else ['s2', 's3']
    if any(stage not in {'s2', 's3'} for stage in stages):
        raise ValueError('Related stages must be s2,s3')
    if args.max_blocks is not None and args.max_blocks < 1:
        raise ValueError('--max-blocks must be positive')
    deadline = None
    if args.stop_starting_utc:
        deadline = datetime.fromisoformat(args.stop_starting_utc.replace('Z', '+00:00'))
        if deadline.tzinfo is None:
            raise ValueError('UTC deadline must include timezone')
        deadline = deadline.astimezone(timezone.utc)
    state_path = output / 'state.json'
    state = read_json(state_path) if state_path.exists() else {'plan_sha256': sha256(plan_path), 'tasks': {}}
    if state['plan_sha256'] != sha256(plan_path):
        raise ValueError('Related state plan mismatch')
    started_blocks = 0
    for stage in ('s2', 's3'):
        if stage not in stages:
            continue
        if stage == 's3' and any(state['tasks'].get(t['id'], {}).get('status') != 'complete'
                                 for t in tasks if t['stage'] == 's2'):
            raise ValueError('S3 starts only after all four predefined S2 blocks are complete')
        for block in range(1, 5):
            block_tasks = [t for t in tasks if t['stage'] == stage and t['block'] == block]
            if all(state['tasks'].get(t['id'], {}).get('status') == 'complete' for t in block_tasks):
                continue
            already_started = any(state['tasks'].get(t['id'], {}).get('attempts') for t in block_tasks)
            if ((args.max_blocks is not None and started_blocks >= args.max_blocks) or
                    (deadline and datetime.now(timezone.utc) >= deadline and not already_started)):
                return {'stopped_between_blocks': True, 'started_blocks': started_blocks, 'state': str(state_path)}
            if stage == 's3':
                ops_task = next(t for t in tasks if t['block'] == block and t['configuration'] == 'ops260')
                if state['tasks'].get(ops_task['id'], {}).get('status') != 'complete':
                    raise ValueError(f'S3 block{block} requires its completed S2 OPS260 block')
            started_blocks += 1
            for task in block_tasks:
                if state['tasks'].get(task['id'], {}).get('status') == 'complete':
                    continue
                selection = dict(manifest=str(manifest), split='dev', allow_holdout=False, limit=None,
                                 ids=','.join(task['selected_ids']), device=args.device)
                attack_output = output / 'attacks' / task['id']
                evaluate_output = output / 'evaluations' / task['id']
                attack_args = argparse.Namespace(command='attack', **selection, output=str(attack_output),
                    method=task['method'], source='rn50', steps=10, views=task['views'], epsilon_bytes=8.,
                    alpha_mult=1., seed=20261027, view_batch=1, di_probability=.5, resize_rate=1.1,
                    save_float=args.save_float, ne=task.get('ne', 5), np=task.get('np', 5), beta=task.get('beta', 2.),
                    profile=task['profile'])
                eval_args = argparse.Namespace(command='evaluate', **selection, output=str(evaluate_output),
                    attack_run=str(attack_output), targets='dn121,vit_b_16', population='target-clean',
                    source_clean=clean_identity['path'])
                entry = state['tasks'].setdefault(task['id'], {'attempts': []})
                attempt = {'started_utc': utc_now(), 'status': 'running'}
                entry['attempts'].append(attempt)
                entry['status'] = 'running'
                write_json(state_path, state)
                start = time.perf_counter()
                try:
                    runner.attack_command(attack_args)
                    runner.evaluate_command(eval_args)
                    entry['status'] = attempt['status'] = 'complete'
                except BaseException as error:
                    entry['status'] = attempt['status'] = 'interrupted' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'failed'
                    attempt['error'] = f'{type(error).__name__}: {error}'
                    raise
                finally:
                    attempt['ended_utc'] = utc_now()
                    attempt['wall_seconds'] = time.perf_counter() - start
                    entry['total_attempt_wall_seconds'] = sum(a.get('wall_seconds', 0) for a in entry['attempts'])
                    write_json(state_path, state)
                    append_jsonl(output / 'events.jsonl', {'task': task['id'], **attempt})
    return {'state': str(state_path), 'started_blocks': started_blocks,
            'complete_tasks': sum(e.get('status') == 'complete' for e in state['tasks'].values()), 'planned_tasks': len(tasks)}
