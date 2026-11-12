"""Original299/V1 batch1 base, FFT/AaF, and independent PNG evaluation.

No resume: each command requires a fresh output. This driver does not change any
historical targeted/FTM path. All complete source forwards are counted by hook.
"""
import hashlib
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline.feature_finetuning import Layer3Source, finetune, PROTOCOL_DEFAULT, SOURCE_LAYERS
from baseline.targeted import ForwardCounter, optimize, target_metrics
from . import runner
from .data import select_records
from .io import (append_jsonl, image_tensor, read_json, read_jsonl, read_pixels,
                 save_adversarial, sha256, write_json)
from .metrics import image_seed
from .models import load_model
from .original299 import ORIGINAL_PREPROCESSING, verify_original
from .targeted_data import target_labels
from .targeted_runner import checkpoints


def utc():
    return datetime.now(timezone.utc).isoformat()


def pixel_sha(path):
    return hashlib.sha256(read_pixels(path).tobytes()).hexdigest()


def fresh_output(path):
    output = runner._output(path)
    if any(output.iterdir()):
        raise FileExistsError('Nonempty output refused; no fine-tuning resume/overwrite')
    return output


# SHA256 of results/holdout_protocol_v1.md, registered 2026-11-04 before the final evaluation.
HOLDOUT_PROTOCOL_SHA256 = 'a891bf83d543ec56ebbce7d96cd9054c366d7961ce480bf80328121e6f3cb371'


def holdout_gate(args):
    """Holdout only for the single pre-registered final evaluation: exact split 'holdout',
    --allow-holdout, and a protocol file whose SHA256 equals the registered one."""
    splits = args.split.split(',')
    wants = 'holdout' in splits or getattr(args, 'allow_holdout', False)
    if not wants:
        return False
    protocol = getattr(args, 'holdout_protocol', None)
    if (splits != ['holdout'] or not getattr(args, 'allow_holdout', False) or not protocol
            or sha256(protocol) != HOLDOUT_PROTOCOL_SHA256):
        raise ValueError('Holdout needs --split holdout --allow-holdout and the registered protocol file')
    return True


def records299(args):
    allow = holdout_gate(args)
    manifest = read_json(args.manifest)
    if manifest.get('input_profile') != 'original299-v1' or manifest.get('preprocessing') != ORIGINAL_PREPROCESSING:
        raise ValueError('Verified original299-v1 manifest required')
    origin = read_json(manifest['source_manifest_path'])
    if sha256(manifest['source_manifest_path']) != manifest['source_manifest_sha256']:
        raise ValueError('Original source manifest SHA differs')
    old = {r['id']: r for r in origin['records']}
    if (len(old) != len(origin['records']) or len(manifest['records']) != len(old)
            or {r['id'] for r in manifest['records']} != set(old) or manifest['counts'] != origin['counts']):
        raise ValueError('Original299 cohort/counts differ')
    for row in manifest['records']:
        prior = old[row['id']]
        if any(row.get(k) != prior.get(k) for k in ('id','label','split','original_path','original_sha256','original_label')):
            raise ValueError('Original299 record binding differs')
        if Path(row['canonical_path']).resolve() != Path(row['original_path']).resolve() or row['canonical_sha256'] != row['original_sha256']:
            raise ValueError('Original299 canonical reference differs')
    records = select_records(args.manifest, args.split, allow, args.limit, args.ids)
    for record in records:
        verify_original(record)
    return records


def identity299(identity, name):
    vit = name == 'vit_b_16'
    return {**identity, 'input_profile': 'original299-v1',
            'artifact_input_size': [299, 299], 'input_size': [224,224] if vit else [299,299],
            'preprocessing': ('decoded RGB299 -> bilinear224 align_corners=False antialias=False -> weight normalization; no crop'
                              if vit else 'decoded original RGB299 uint8/255 -> weight normalization only; no resize')}


def state_sha(model):
    result = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        result.update(name.encode())
        result.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def pixels(count):
    return sum(int(s.split('x')[0])*int(s.split('x')[1])*n for s,n in count.resolutions.items())


def guidance_scalars(values, method):
    """Read only the already computed guidance; no model calls or RNG draws."""
    meanings = {'original':'clean original-class aggregate, normalized by its correction1 std',
                'target':'fixed initial-PNG target-class aggregate, normalized by its correction1 std',
                'combined':'target minus0.2 times original; fixed layer3 inner-product guidance'}
    result = {}
    for name in ('original','target','combined'):
        tensor = values[name].detach()
        std = float(tensor.std(correction=1).item())
        l2 = float(torch.linalg.vector_norm(tensor.to(torch.float64)).item())
        result[name] = {'method':method,'meaning':meanings[name],
                        'std_correction1':std if math.isfinite(std) else None,
                        'std_dtype':str(tensor.dtype),
                        'l2_norm':l2 if math.isfinite(l2) else None,'l2_accumulation_dtype':'torch.float64',
                        'nonfinite_count':int((~torch.isfinite(tensor)).sum().item()),
                        'zero_count':int((tensor==0).sum().item()),'numel':tensor.numel()}
    return result


def png299(path, expected_sha=None):
    if expected_sha is not None and sha256(path) != expected_sha:
        raise ValueError('Artifact PNG file SHA changed')
    with Image.open(path) as image:
        if image.format != 'PNG' or image.mode != 'RGB' or image.size != (299,299):
            raise ValueError('Expected RGB299 PNG')
    return read_pixels(path)


def assert_budget(path, clean, epsilon):
    decoded = png299(path)
    if int(np.abs(decoded.astype(np.int16)-clean.astype(np.int16)).max()) > epsilon:
        raise ValueError('Decoded initial/final PNG exceeds clean-centered integer budget')
    return decoded


def initial_rows(args, records, labels, identity):
    directory = Path(args.base_run).resolve()
    config_path = directory/'config.json'
    config = read_json(config_path)
    if (config.get('kind') != 'feature-base299' or config.get('profile') != 'local-original299-v1-logit-rtm-prefix'
            or config.get('source') != getattr(args,'source','rn50_v1') or config.get('batch_size') != 1
            or config['manifest_sha256'] != sha256(args.manifest)):
        raise ValueError('Initial run kind/manifest mismatch')
    if config['source_identity'] != identity or config['labels_csv_sha256'] != sha256(args.labels_csv):
        raise ValueError('Initial run source/label identity mismatch')
    if config['code_sha256'] != runner._code_identity():
        raise ValueError('Initial base source code differs from current frozen code')
    if config['arguments']['epsilon_bytes'] != args.epsilon_bytes:
        raise ValueError('Initial run budget mismatch')
    rows = read_jsonl(directory/'progress.jsonl')
    mapped = {r['id']: r for r in rows}
    if len(mapped) != len(rows):
        raise ValueError('Duplicate initial progress IDs')
    selected = {}
    for record in records:
        row = mapped.get(record['id'])
        if row is None or row['target_label'] != labels[record['id']] or row['label'] != record['label'] or row['canonical_sha256'] != record['canonical_sha256']:
            raise ValueError('Initial image/target/clean binding mismatch')
        if args.base_checkpoint not in config['checkpoint_steps'] or str(args.base_checkpoint) not in row['checkpoints']:
            raise ValueError('Requested initial checkpoint missing')
        entry = row['checkpoints'][str(args.base_checkpoint)]
        assert_budget(entry['png_path'], read_pixels(record['canonical_path']), args.epsilon_bytes)
        png299(entry['png_path'], entry['png_sha256'])
        if pixel_sha(entry['png_path']) != entry['png_pixel_sha256']:
            raise ValueError('Initial decoded pixel SHA differs')
        selected[record['id']] = entry
    return selected, {'path':str(directory), 'config_sha256':sha256(config_path),
                      'progress_sha256':sha256(directory/'progress.jsonl'),
                      'checkpoint':args.base_checkpoint, 'source_code_sha256':config['code_sha256']}


def fine_protocol(args):
    return {'alpha_bytes':getattr(args,'fine_alpha_bytes',None),'di':getattr(args,'fine_di','author'),
            'momentum_norm':getattr(args,'fine_momentum_norm','none'),
            'init_momentum':'inherit' if getattr(args,'fine_continue',False) else None,
            'conflict_reset':getattr(args,'fine_conflict_reset',False),
            'conflict_samples':getattr(args,'fine_conflict_samples',1),
            'joint_weight':getattr(args,'fine_joint_weight',None),
            'fine_steps':getattr(args,'fine_steps',None)}


def continue_start(entry,clean,args,device):
    """Author/ablation starts from the decoded PNG; --fine-continue starts from the
    saved float image and inherited base momentum (both SHA-verified)."""
    protocol=fine_protocol(args)
    keep=getattr(args,'fine_continue',False)
    if not keep and not getattr(args,'fine_float_start',False):
        return image_tensor(entry['png_path'],device),protocol
    if 'float_path' not in entry or (keep and 'momentum_path' not in entry):
        raise ValueError('Float start needs --save-float (and --save-momentum for --fine-continue) in the base run')
    if sha256(entry['float_path'])!=entry['float_sha256'] or (keep and sha256(entry['momentum_path'])!=entry['momentum_sha256']):
        raise ValueError('Initial float/momentum SHA differs')
    start=torch.from_numpy(np.load(entry['float_path'])['image']).to(device)
    if start.shape!=clean.shape or float((start-clean).abs().max())>args.epsilon_bytes/255+1e-6:
        raise ValueError('Initial float image shape/budget mismatch')
    if keep:
        decay=getattr(args,'fine_momentum_decay',1.0)
        if not 0<=decay<=1:
            raise ValueError('--fine-momentum-decay must be in [0,1]')
        momentum=torch.from_numpy(np.load(entry['momentum_path'])['momentum']).to(device)
        protocol['init_momentum']=momentum if decay==1 else momentum*decay
    return start,protocol


def attack_command(args):
    base = args.command == 'feature-base299'
    if not math.isfinite(args.epsilon_bytes) or args.epsilon_bytes < 0:
        raise ValueError('Finite nonnegative epsilon required')
    if base and (args.steps < 0 or not math.isfinite(args.alpha_bytes) or args.alpha_bytes < 0):
        raise ValueError('Invalid base budget/steps')
    records = records299(args)
    labels = target_labels(args.labels_csv, records)
    output = fresh_output(args.output)
    device = runner._device(args.device)
    source = getattr(args,'source','rn50_v1')
    if source not in SOURCE_LAYERS:
        raise ValueError(f'Source must be one of {sorted(SOURCE_LAYERS)}')
    wrapped, identity = load_model(source, device)
    identity = identity299(identity, source)
    initial, binding = ({},None) if base else initial_rows(args,records,labels,identity)
    points = checkpoints(args.checkpoints,args.steps) if base else [fine_protocol(args)['fine_steps'] or (15 if args.method.endswith('aaf') else 10)]
    config = runner._config(args,records,kind='feature-base299' if base else 'feature-finetune299',
        source_identity=identity, target_labels=labels, labels_csv_sha256=sha256(args.labels_csv),
        initial_run=binding, checkpoint_steps=points, source=source, batch_size=1,
        profile=('local-original299-v1-logit-rtm-prefix' if base else 'author-code-batch1-original299-v1-'+args.method
                 +('' if fine_protocol(args)==PROTOCOL_DEFAULT else '-protocol-ablation')),
        fine_start=None if base else ('float+inherited-momentum' if getattr(args,'fine_continue',False)
                                      else 'float+zero-momentum' if getattr(args,'fine_float_start',False) else 'png'),
        fine_protocol=None if base else fine_protocol(args),
        layer=None if base else SOURCE_LAYERS[source], guidance_samples_per_class=30 if not base and args.method in ('fft','aaf') else 0,
        adaptation='Batch1 local transfer of fixed author code, not author batch4/global RNG/software/main-table reproduction. Base uses existing local torch RDI, not author NumPy DI. All complete source forwards; only guidance layer3 retained, author initial shape-only forward omitted. Frozen parameters/autograd.grad instead of author parameter backward.',
        gradient_cost_definition='feature_gradient_images counts logit-to-layer3 guidance backward; input_gradient_images counts objective-to-input backward. These are not equivalent costs.',
        elapsed_seconds_profile='Synchronized end-to-end image wall: guidance/optimization, PNG/float IO, endpoints. Excludes model load/clean read; parameter+BN digest verified separately for whole run.',
        rng='Base dedicated CPU torch.Generator SHA(seed:imageID); fine dedicated NumPy RandomState(MT19937), SHA(seed:imageID) modulo2**32; no global experimental RNG draws')
    config['matching_rng_advance'] = {'enabled':not base and args.method in ('logit-fft','logit-aaf'),
        'policy':'For logit controls, discard corresponding original30+target30 author mask draws before DI; no guidance values, no model forward/backward. Same single RandomState as feature method.',
        'forward_images':0,'feature_gradient_images':0,'input_gradient_images':0}
    write_json(output/'config.json',config)
    before = state_sha(wrapped)
    rows = []
    run_start = time.perf_counter()
    model = Layer3Source(wrapped, SOURCE_LAYERS[source])
    try:
        for record in records:
            clean_pixels = read_pixels(record['canonical_path'])
            clean = image_tensor(record['canonical_path'],device)
            target = labels[record['id']]
            seed = image_seed(args.seed,record['id'])
            grad_counts = {'feature':0,'input':0}
            phase_seconds = {'guidance':0.,'matching_rng_advance':0.,'optimization':0.,'endpoints':0.}
            guidance_summary = {}
            conflict_info = {}
            saved = {}
            sync(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            started_utc = utc()
            with ForwardCounter(model) as count:
                try:
                    count.phase = 'clean'
                    with torch.no_grad():
                        source_clean = target_metrics(model(clean),target,record['label'])
                    def endpoint(step,image):
                        sync(device)
                        endpoint_start=time.perf_counter()
                        path = output/'images'/f'{record["id"]}.png' if step == points[-1] else output/'checkpoints'/str(step)/'images'/f'{record["id"]}.png'
                        count.phase = 'endpoint_float'
                        with torch.no_grad():
                            floating = target_metrics(model(image),target,record['label'])
                        metrics = save_adversarial(path,image,clean_pixels,args.epsilon_bytes)
                        count.phase = 'endpoint_png'
                        with torch.no_grad():
                            decoded = target_metrics(model(image_tensor(path,device)),target,record['label'])
                        item = {'step':step,'png_path':str(path),'png_pixel_sha256':pixel_sha(path),
                                'float_source':floating,'png_source':decoded,**metrics}
                        if args.save_float:
                            float_path = path.with_suffix('.npz')
                            np.savez_compressed(float_path,image=image.detach().cpu().numpy())
                            item.update(float_path=str(float_path),float_sha256=sha256(float_path))
                        saved[str(step)] = item
                        sync(device)
                        phase_seconds['endpoints'] += time.perf_counter()-endpoint_start
                    if base:
                        def callback(step,image):
                            grad_counts['input'] = step
                            endpoint(step,image)
                        def state_callback(step,momentum):
                            if getattr(args,'save_momentum',False):
                                mpath=(output/'images' if step==points[-1] else output/'checkpoints'/str(step)/'images')/f'{record["id"]}.momentum.npz'
                                mpath.parent.mkdir(parents=True,exist_ok=True)
                                np.savez_compressed(mpath,momentum=momentum.detach().cpu().numpy())
                                pending_momentum[str(step)]={'momentum_path':str(mpath),'momentum_sha256':sha256(mpath)}
                        pending_momentum={}
                        # Count completed backwards even if an endpoint callback fails.
                        # optimize has no per-gradient callback; on optimization failure
                        # the exact completed B is unknown and recorded as null below.
                        optimize_start=time.perf_counter()
                        optimize(model,clean,target,'logit-rtm',args.steps,args.epsilon_bytes,args.alpha_bytes,
                                 seed,checkpoints=points,callback=callback,counter=count,state_callback=state_callback)
                        for key,value in pending_momentum.items():saved[key].update(value)
                        sync(device)
                        phase_seconds['optimization']=time.perf_counter()-optimize_start-phase_seconds['endpoints']
                        grad_counts['input'] = args.steps
                        rng_state = None
                    else:
                        rng = np.random.RandomState(seed % (2**32))
                        def after(kind):
                            grad_counts[kind] += 1
                        optimize_start=time.perf_counter()
                        def observed(kind,_values):
                            sync(device)
                            if kind=='guidance':
                                guidance_summary.update(guidance_scalars(_values,args.method))
                                sync(device)
                                phase_seconds['guidance']=time.perf_counter()-optimize_start
                            if kind=='matching_rng_advance':phase_seconds['matching_rng_advance']=_values['seconds']
                            if kind=='conflict_reset':conflict_info.update(_values)
                            count.phase = 'optimization'
                        count.phase = 'guidance' if not args.method.startswith('logit-') else 'optimization'
                        start_image,protocol=continue_start(initial[record['id']],clean,args,device)
                        image = finetune(model,clean,start_image,target,
                                         record['label'],args.method,args.epsilon_bytes,rng,after,observed,
                                         protocol=protocol)
                        sync(device)
                        phase_seconds['optimization']=time.perf_counter()-optimize_start-phase_seconds['guidance']-phase_seconds['matching_rng_advance']
                        rng_state = hashlib.sha256(rng.get_state()[1].tobytes()+str(rng.get_state()[2:]).encode()).hexdigest()
                        endpoint(points[-1],image)
                    sync(device)
                    final = saved[str(points[-1])]
                    extra = 0 if base or fine_protocol(args)['conflict_samples'] == 1 else fine_protocol(args)['conflict_samples']
                    expected_b = args.steps if base else points[-1] + extra
                    expected_feature_b = 60 if not base and args.method in ('fft','aaf') else 0
                    expected_f = 1 + 2*len(points) + expected_b + expected_feature_b + (2 if not base and args.method=='ila' else 0)
                    expected_input_b = expected_b * (2 if not base and fine_protocol(args)['joint_weight'] is not None else 1)
                    if count.total != expected_f or grad_counts != {'feature':expected_feature_b,'input':expected_input_b}:
                        raise ValueError('Actual forward/backward accounting mismatch')
                    row = {'id':record['id'],'label':record['label'],'target_label':target,'split':record['split'],
                           'canonical_sha256':record['canonical_sha256'],'clean_pixel_sha256':pixel_sha(record['canonical_path']),
                           'image_seed':seed,'numpy_seed':None if base else seed % (2**32),'final_numpy_state_sha256':rng_state,
                           'started_utc':started_utc,'completed_utc':utc(),
                           'source_clean':source_clean,'source_float':final['float_source'],'source_png':final['png_source'],
                           'checkpoints':saved,'png_path':final['png_path'],'png_sha256':final['png_sha256'],
                           'png_pixel_sha256':final['png_pixel_sha256'],'initial':initial.get(record['id']),
                           'forward_images':count.total,'forward_pixels':pixels(count),'forward_by_phase':dict(count.counts),
                           'resolution_counts':dict(count.resolutions),'feature_gradient_images':grad_counts['feature'],
                           'phase_seconds':phase_seconds,'matching_rng_advance_mask_draws':60 if not base and args.method in ('logit-fft','logit-aaf') else 0,
                           'guidance_summary':guidance_summary,
                           'guidance_summary_applicable':not base and not args.method.startswith('logit-'),
                           'conflict_reset':conflict_info,
                           'input_gradient_images':grad_counts['input'],'input_gradient_pixels':grad_counts['input']*299**2,
                           'elapsed_seconds':time.perf_counter()-start,
                           'peak_cuda_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
                    append_jsonl(output/'progress.jsonl',row)
                    rows.append(row)
                except BaseException as error:
                    sync(device)
                    append_jsonl(output/'attempts.jsonl',{'id':record['id'],'status':'interrupted' if isinstance(error,KeyboardInterrupt) else 'failed',
                        'utc':utc(),'error':str(error),'forward_images':count.total,'forward_by_phase':dict(count.counts),
                        'forward_pixels':pixels(count),'feature_gradient_images':grad_counts['feature'],
                        'input_gradient_images':None if base else grad_counts['input'],
                        'completed_base_checkpoint_backward_lower_bound':grad_counts['input'] if base else None,
                        'phase_seconds':phase_seconds,'started_utc':started_utc,
                        'guidance_summary':guidance_summary,
                        'guidance_summary_applicable':not base and not args.method.startswith('logit-'),
                        'peak_cuda_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                        'elapsed_seconds':time.perf_counter()-start})
                    raise
    finally:
        model.close()
        after = state_sha(wrapped)
        summary = {'status':'complete' if len(rows)==len(records) else 'partial','completed':len(rows),'planned':len(records),
                   'source_state_sha256_before':before,'source_state_sha256_after':after,'parameters_and_bn_unchanged':before==after,
                   'parameters_grad_none':all(p.grad is None for p in wrapped.parameters()),
                   'forward_images':sum(r['forward_images'] for r in rows),
                   'feature_gradient_images':sum(r['feature_gradient_images'] for r in rows),
                   'input_gradient_images':sum(r['input_gradient_images'] for r in rows),
                   'wall_seconds':time.perf_counter()-run_start,'utc':utc()}
        write_json(output/'summary.json',summary)
    if before != after or not summary['parameters_grad_none']:
        raise ValueError('Source parameter/BN state changed or parameter gradients accumulated')
    return summary


def evaluate_command(args):
    records = records299(args)
    labels = target_labels(args.labels_csv,records)
    directory = Path(args.attack_run).resolve()
    attack_config = read_json(directory/'config.json')
    if attack_config.get('kind') not in ('feature-base299','feature-finetune299') or attack_config['manifest_sha256'] != sha256(args.manifest) or attack_config['labels_csv_sha256'] != sha256(args.labels_csv):
        raise ValueError('Evaluation attack kind/data identity mismatch')
    source = attack_config.get('source')
    if source not in SOURCE_LAYERS or attack_config.get('batch_size')!=1 or attack_config['source_identity']['input_size'] != [299,299]:
        raise ValueError('Evaluation requires original299 batch1 artifacts from a known source')
    rows = read_jsonl(directory/'progress.jsonl')
    mapping = {r['id']:r for r in rows}
    if len(mapping)!=len(rows):
        raise ValueError('Duplicate attack rows')
    names = args.targets.split(',')
    if len(names)!=len(set(names)) or any(n not in ('rn50_v1','dn121','vgg16_bn','inc_v3','vit_b_16') for n in names):
        raise ValueError('Targets must be unique rn50_v1,dn121,vgg16_bn,inc_v3,vit_b_16')
    output = fresh_output(args.output)
    device = runner._device(args.device)
    config = runner._config(args,records,kind='feature-evaluation299',
        attack_config_sha256=sha256(directory/'config.json'),attack_progress_sha256=sha256(directory/'progress.jsonl'),
        source_identity=attack_config['source_identity'],target_labels=labels,labels_csv_sha256=sha256(args.labels_csv),
        target_identities={},cost_definition='Independent target-stage clean+decodedPNG forwards; never recycled as source attack forwards')
    write_json(output/'config.json',config)
    write_json(output/'summary.json',{'status':'partial','models':{},'planned_models':names})
    complete = []
    summaries = {}
    for name in names:
        model, identity = load_model(name,device)
        config['target_identities'][name] = identity299(identity,name)
        if name==source and config['target_identities'][name]!=attack_config['source_identity']:
            raise ValueError('Independent source recheck weight/preprocessing identity differs')
        write_json(output/'config.json',config)
        if device.type=='cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model_rows=[]
        sync(device)
        start=time.perf_counter()
        with ForwardCounter(model) as count:
            try:
                for record in records:
                    row=mapping.get(record['id'])
                    if row is None:
                        continue
                    if row['label']!=record['label'] or row['target_label']!=labels[record['id']] or row['canonical_sha256']!=record['canonical_sha256']:
                        raise ValueError('Attack row labels/clean mismatch')
                    chosen=str(args.checkpoint) if args.checkpoint is not None else str(max(map(int,row['checkpoints'])))
                    if chosen not in row['checkpoints']:
                        raise ValueError('Requested evaluation checkpoint missing')
                    entry=row['checkpoints'][chosen]
                    png299(entry['png_path'],entry['png_sha256'])
                    if pixel_sha(entry['png_path'])!=entry['png_pixel_sha256']:
                        raise ValueError('Evaluation decoded pixel SHA differs')
                    assert_budget(entry['png_path'],read_pixels(record['canonical_path']),attack_config['arguments']['epsilon_bytes'])
                    def forward(path,phase):
                        image=image_tensor(path,device)
                        if name=='vit_b_16':
                            image=F.interpolate(image,size=(224,224),mode='bilinear',align_corners=False,antialias=False)
                        count.phase=phase
                        with torch.no_grad():
                            return target_metrics(model(image),labels[record['id']],record['label'])
                    clean=forward(record['canonical_path'],'clean')
                    adv=forward(entry['png_path'],'adv_png')
                    result={'id':record['id'],'model':name,'label':record['label'],'target_label':labels[record['id']],
                            'step':entry['step'],'clean':clean,'adv':adv,'source_clean':row['source_clean'],
                            'png_sha256':entry['png_sha256'],'png_pixel_sha256':entry['png_pixel_sha256']}
                    append_jsonl(output/'progress.jsonl',result)
                    model_rows.append(result);complete.append(result)
            except BaseException as error:
                sync(device)
                append_jsonl(output/'attempts.jsonl',{'model':name,'status':'failed','utc':utc(),'error':str(error),
                    'forward_images':count.total,'forward_pixels':pixels(count),'elapsed_seconds':time.perf_counter()-start})
                summaries[name]={'status':'failed','completed':len(model_rows),'planned':len(records),
                                 'forward_images':count.total,'forward_pixels':pixels(count),
                                 'forward_by_phase':dict(count.counts),'elapsed_seconds':time.perf_counter()-start}
                write_json(output/'summary.json',{'status':'partial','models':summaries,'planned_models':names})
                raise
        sync(device)
        populations={}
        for population in ('all','target-clean','source-target-clean'):
            eligible=[r for r in model_rows if population=='all' or r['clean']['true_correct'] and (population=='target-clean' or r['source_clean']['true_correct'])]
            success=sum(r['adv']['targeted_success'] for r in eligible)
            populations[population]={'ids':[r['id'] for r in eligible],'numerator':success,'denominator':len(eligible),
                                     'targeted_success_rate':success/len(eligible) if eligible else None}
        summaries[name]={'status':'complete' if len(model_rows)==len(records) else 'partial','completed':len(model_rows),'planned':len(records),
                         'query_role':'independent-source-recheck' if name==source else 'independent-target',
                         'populations':populations,'forward_images':count.total,'forward_pixels':pixels(count),
                         'forward_by_phase':dict(count.counts),'resolution_counts':dict(count.resolutions),
                         'elapsed_seconds':time.perf_counter()-start,
                         'peak_cuda_allocated':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None}
        write_json(output/'summary.json',{'status':'complete' if len(summaries)==len(names) and all(s['status']=='complete' for s in summaries.values()) else 'partial','models':summaries})
        del model,count
    return read_json(output/'summary.json')
