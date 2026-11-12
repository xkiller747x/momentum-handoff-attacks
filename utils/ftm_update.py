"""Read-only FTM raw-update audit and same-RNG known freezing-control report."""
import json
from pathlib import Path
import time
import numpy as np
import torch

from .io import read_json,read_jsonl,read_pixels,sha256,write_json
from .targeted_runner import verify_saved


class UpdateTrace:
    """FP64 CPU observations only; first directions never enter optimization."""
    def __init__(self,path,sample_ids,layer_types,update_mode,beta=.01,state_control='raw'):
        self.path=Path(path);self.path.parent.mkdir(parents=True,exist_ok=True)
        self.temporary=self.path.with_suffix('.jsonl.tmp');self.stream=self.temporary.open('w',encoding='utf-8')
        self.ids=sample_ids;self.types=layer_types;self.mode=update_mode;self.beta=beta
        self.state_control=state_control
        self.first={};self.rows=0;self.seconds=0.;self.closed=False
    def close(self):
        if not self.closed:self.stream.flush();self.stream.close();self.closed=True
    def commit(self):
        self.close();self.temporary.replace(self.path)
        return {'path':str(self.path),'sha256':sha256(self.path),'rows':self.rows,'observer_seconds':self.seconds,
            'observer_seconds_scope':'FP64 CPU callback/copy/JSON writing; close and SHA additionally included in batch wall time',
            'effective_v_definition':'FP64 beta*captured_current_z_norm*u/(norm64(u)+1e-7), holding z fixed; before channel mixing, not a GPU bitwise v measurement'}
    def __call__(self,step,name,old,gradient,after,context):
        start=time.perf_counter()
        a=old.detach().to(device='cpu',dtype=torch.float64).reshape(len(self.ids),-1)
        g=gradient.detach().to(device='cpu',dtype=torch.float64).reshape_as(a)
        b=after.detach().to(device='cpu',dtype=torch.float64).reshape_as(a)
        z=context.output_norms[name].detach().to(device='cpu',dtype=torch.float64)
        if not all(torch.isfinite(v).all() for v in (a,g,b,z)):raise ValueError('Nonfinite FTM observation')
        state=self.first.setdefault(name,{'steps':[None]*len(self.ids),'directions':[None]*len(self.ids)})
        for i,identity in enumerate(self.ids):
            an=float(a[i].norm());gn=float(g[i].norm());bn=float(b[i].norm());change=b[i]-a[i];change_norm=float(change.norm())
            if state['steps'][i] is None and change_norm>0 and bn>0:
                state['steps'][i]=step;state['directions'][i]=(b[i]/bn).clone()
            direction=state['directions'][i]
            drift=float((b[i]/bn-direction).norm()) if bn>0 and direction is not None else None
            chord=float((b[i]/bn-a[i]/an).norm()) if an>0 and bn>0 else None
            negative_dot=bool(torch.dot(a[i]/an,b[i]/bn)<0) if an>0 and bn>0 else None
            va=self.beta*z[i]*a[i]/(an+1e-7);vb=self.beta*z[i]*b[i]/(bn+1e-7)
            vnorm=float(va.norm());vdiff=float((vb-va).norm())
            row={'step':step,'layer':name,'layer_type':self.types[name],'sample_id':identity,'update_mode':self.mode,
                'state_control':self.state_control,
                'old_u_norm':an,'grad_u_norm':gn,'new_u_norm':bn,'actual_delta_u_norm':change_norm,
                'theoretical_relative_step':gn/an if an>0 else None,'actual_relative_step':change_norm/an if an>0 else None,
                'theoretical_relative_step_definition':'Raw gradient norm / old u norm before state transform; not accepted direction step.',
                'old_to_new_unit_direction_chord':chord,'old_to_new_unit_dot_negative':negative_dot,
                'changed_element_fraction':float((b[i]!=a[i]).double().mean()),
                'first_nonzero_update_step':state['steps'][i],'unit_direction_l2_drift_from_first':drift,
                'current_output_norm':float(z[i]),'effective_v_old_norm_fixed_z':vnorm,
                'effective_v_absolute_change_fixed_z':vdiff,'effective_v_relative_change_fixed_z':vdiff/vnorm if vnorm>0 else None,
                'old_u_zero':an==0,'gradient_zero':gn==0,'is_subsequent_selected_update':state['steps'][i] is not None and step>state['steps'][i],
                'effective_v_observation_precision':'FP64 CPU parameterization, fixed current z, before blending'}
            self.stream.write(json.dumps(row,allow_nan=False)+'\n');self.rows+=1
        self.seconds+=time.perf_counter()-start


def compatible_configs(full,control):
    if full.get('kind')!='targeted-attack' or control.get('kind')!='targeted-attack':raise ValueError('Expected targeted attack runs')
    for key in ('manifest_sha256','labels_csv_sha256','target_labels','selected_ids','source_identity','pairs','pair_table_sha256',
                'layer_table_sha256','qualified_layer_names','settings','checkpoint_steps','code_sha256','numeric_settings','state_control','state_control_spec'):
        if full.get(key)!=control.get(key):raise ValueError(f'FTM diagnostic identity differs: {key}')
    for key in ('source','steps','epsilon_bytes','alpha_bytes','seed','batch_size','ftm_layer_profile'):
        if full['arguments'].get(key)!=control['arguments'].get(key):raise ValueError(f'FTM diagnostic budget differs: {key}')
    if full.get('update_mode',full['arguments'].get('ftm_update_mode','full'))!='full' or control.get('update_mode',control['arguments'].get('ftm_update_mode'))!='first-nonzero-only':
        raise ValueError('Expected full versus first-nonzero-only known control')
    for config,mode in ((full,'full'),(control,'first-nonzero-only')):
        if config['arguments'].get('ftm_update_mode',mode)!=mode:raise ValueError('FTM update mode metadata inconsistent')


def compare_images(a,b):
    pa,pb=read_pixels(a['png_path']),read_pixels(b['png_path'])
    if pa.shape!=pb.shape:raise ValueError('Diagnostic PNG shapes differ')
    result={'png_exact_equal':bool(np.array_equal(pa,pb)),'png_max_abs_bytes':int(np.abs(pa.astype(np.int16)-pb.astype(np.int16)).max()),
            'float_available':bool(a.get('float_path') and b.get('float_path')),'float_exact_equal':None,'float_max_abs':None,'float_l2':None}
    if result['float_available']:
        with np.load(a['float_path'],allow_pickle=False) as archive:x=archive['image']
        with np.load(b['float_path'],allow_pickle=False) as archive:y=archive['image']
        if x.shape!=y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Diagnostic float shape/nonfinite mismatch')
        delta=x.astype(np.float64)-y.astype(np.float64)
        result.update(float_exact_equal=bool(np.array_equal(x,y)),float_max_abs=float(np.abs(delta).max()),float_l2=float(np.linalg.norm(delta.reshape(-1))))
    return result


def trace_summary(rows):
    result=[]
    for kind in ('Conv2d','Linear'):
        group=[r for r in rows if r['layer_type']==kind];later=[r for r in group if r['is_subsequent_selected_update']]
        item={'layer_type':kind,'selected_sample_updates':len(group),'subsequent_selected_sample_updates':len(later),
            'layers':sorted({r['layer'] for r in group}),'samples':sorted({r['sample_id'] for r in group})}
        for key in ('theoretical_relative_step','actual_relative_step','changed_element_fraction','unit_direction_l2_drift_from_first','effective_v_relative_change_fixed_z','old_to_new_unit_direction_chord','old_to_new_unit_dot_negative'):
            values=[r[key] for r in later if r.get(key) is not None]
            item[key]={'non_null_count':len(values),'missing_count':len(later)-len(values),
                'min':min(values) if values else None,'median':float(np.median(values)) if values else None,'max':max(values) if values else None,
                'mean':float(np.mean(values)) if values else None}
        result.append(item)
    return result


def report_command(args):
    from .runner import _output
    from .matrix_report import _csv
    roots=[Path(args.full_run).resolve(),Path(args.control_run).resolve()];configs=[read_json(root/'config.json') for root in roots]
    compatible_configs(*configs);output=_output(args.output);all_rows=[];batches=[];costs=[];maps=[];completed=[]
    expected=set(configs[0]['selected_ids']);pairs={p['batch_id']:p for p in configs[0]['pairs']}
    for name,root,config in zip(('full','first-nonzero-only'),roots,configs):
        records=None
        if config.get('manifest_path'):
            if sha256(config['manifest_path'])!=config['manifest_sha256']:raise ValueError('Diagnostic manifest changed')
            records={r['id']:r for r in read_json(config['manifest_path'])['records']}
        if config['arguments'].get('labels_csv') and sha256(config['arguments']['labels_csv'])!=config['labels_csv_sha256']:raise ValueError('Diagnostic label CSV changed')
        rows=read_jsonl(root/'progress.jsonl');mapping={r['id']:r for r in rows}
        if len(mapping)!=len(rows) or not set(mapping).issubset(expected):raise ValueError('Unexpected diagnostic image rows')
        done=set();traces=[]
        for row in rows:
            pair=pairs.get(row['batch_id'])
            if pair is None or row['id'] not in pair['image_ids'] or row['batch_rng_seed']!=pair['rng_seed'] or row['target_label']!=config['target_labels'][row['id']] or row['layer_table_sha256']!=config['layer_table_sha256']:raise ValueError('Diagnostic pair/target/layer identity differs')
            if records and (row['label']!=records[row['id']]['label'] or row['canonical_sha256']!=records[row['id']]['canonical_sha256']):raise ValueError('Diagnostic true-label/canonical identity differs')
            verify_saved(row)
            path=Path(row['shared_batch_total_path'])
            if sha256(path)!=row['shared_batch_total_sha256']:raise ValueError('Diagnostic batch metadata changed')
            if row['batch_id'] in done:continue
            stats=read_json(path)
            if not set(pair['image_ids']).issubset(mapping):raise ValueError('Half committed diagnostic pair')
            trace=stats.get('update_trace')
            if trace:
                if sha256(trace['path'])!=trace['sha256']:raise ValueError('Diagnostic trace SHA changed')
                trace_rows=read_jsonl(trace['path'])
                if len(trace_rows)!=trace['rows']:raise ValueError('Diagnostic trace row count differs')
                if any(r['sample_id'] not in pair['image_ids'] or r['layer'] not in config['qualified_layer_names'] or r['update_mode']!=name for r in trace_rows):raise ValueError('Diagnostic trace pair/layer/mode differs')
                traces.extend({'run':name,'batch_id':row['batch_id'],**r} for r in trace_rows)
            batches.append({'run':name,'batch_id':row['batch_id'],'generator_state_sha256':stats['final_generator_state_sha256'],
                'trace_rows':trace['rows'] if trace else 0,'observer_seconds':trace['observer_seconds'] if trace else 0,
                'forward_images':stats['forward_images'],'input_gradient_images':stats['input_gradient_images'],'elapsed_seconds':stats['elapsed_seconds'],
                'peak_cuda_allocated_bytes':stats['peak_cuda_allocated_bytes']})
            done.add(row['batch_id'])
        all_rows.extend(traces);maps.append(mapping);completed.append(set(mapping)==expected)
        costs.append({'run':name,'preparation':read_jsonl(root/'preparation.jsonl'),'attempts':read_jsonl(root/'attempts.jsonl'),
            'summary':read_json(root/'summary.json') if (root/'summary.json').exists() else None,
            'trace_summary':trace_summary(traces)})
    for pair in pairs:
        values=[r['generator_state_sha256'] for r in batches if r['batch_id']==pair]
        if len(values)==2 and values[0]!=values[1]:raise ValueError('Full/control final pair RNG differs')
    comparisons=[]
    final=str(configs[0]['arguments']['steps'])
    for identity in configs[0]['selected_ids']:
        if identity not in maps[0] or identity not in maps[1]:continue
        a,b=maps[0][identity],maps[1][identity]
        comparisons.append({'id':identity,'batch_id':a['batch_id'],**compare_images(a['checkpoints'][final],b['checkpoints'][final])})
    report={'status':'complete' if all(completed) else 'partial','expected_images':len(expected),'observed_images':[len(m) for m in maps],
        'paired_images':len(comparisons),'image_comparisons':comparisons,'batches':batches,'costs':costs,
        'interpretation':'Low-priority replication audit and known freezing control. Raw u and fixed-current-z effective v are distinct. Conv and FC summaries are separate; selected sample-updates are descriptive, not IID evidence. Changes are retained; no target-transfer conclusion, CI, new attack or repair claim.'}
    write_json(output/'report.json',report);_csv(output/'images.csv',comparisons);_csv(output/'batches.csv',batches);_csv(output/'updates.csv',all_rows)
    summaries=[{'run':value['run'],**r} for value in costs for r in value['trace_summary']];_csv(output/'trace_summary.csv',summaries)
    _csv(output/'attempt_costs.csv',[{'run':value['run'],**row} for value in costs for row in value['attempts']])
    lines=['# FTM update replication audit','',f'Status: {report["status"]}; paired images {len(comparisons)}/{len(expected)}.','',report['interpretation'],'',
        '| Image | Float exact | Float maxabs | Float L2 | PNG exact | PNG maxbyte |','|---|---|---:|---:|---|---:|']
    for row in comparisons:lines.append(f'| {row["id"]} | {row["float_exact_equal"]} | {row["float_max_abs"]} | {row["float_l2"]} | {row["png_exact_equal"]} | {row["png_max_abs_bytes"]} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return {'output':str(output),'status':report['status'],'paired_images':len(comparisons),'target_forward_images':0}
