"""CPU20 grouping/ABI checks and pinned pure-author small-model differences."""
import argparse
import ast
import copy
import ctypes
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Callable
import unittest
from unittest.mock import MagicMock,patch

import torch
from torch import nn,Tensor
import torch.nn.functional as F

from baseline.ftm import FeatureTuning,author_layers,optimize_ftm
from baseline.targeted import rdi,ti_kernel
from utils.ftm_cpu_audit import validate,configure,ProcessMemoryCounters,process_peak_memory
from utils.targeted_ftm import pair_table
from utils.metrics import image_seed


def author_class(probability):
    folder=Path(__file__).parent/'reference';source=folder/'ftm_attacks.py.txt';config=folder/'ftm_config.py.txt'
    if hashlib.sha256(source.read_bytes()).hexdigest()!='4673e39590d3ad362563e4b90232b34bc8df3e5bc9df66b50d0819c2ade6e8ad':raise ValueError('Pinned author source changed')
    if hashlib.sha256(config.read_bytes()).hexdigest()!='84863e7c1104e5c7f266be42068dd2361e4820432ba1a82e8b7b1ee222493665':raise ValueError('Pinned author config changed')
    settings=dict(ast.literal_eval(next(n.value for n in ast.parse(config.read_text()).body if isinstance(n,ast.Assign)))[1]);settings['mix_prob']=probability
    tree=ast.Module(body=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.ClassDef) and n.name=='FeatureTuning'],type_ignores=[])
    namespace={'torch':torch,'nn':nn,'Tensor':Tensor,'Callable':Callable,'F':F}
    exec(compile(tree,str(source),'exec'),namespace)
    return namespace['FeatureTuning'],settings


class Small(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,3,1,stride=16,bias=False)
        self.pool=nn.AdaptiveAvgPool2d(1);self.fc=nn.Linear(3,3,bias=False)
        with torch.no_grad():
            for i,p in enumerate(self.parameters()):p.copy_(torch.linspace(.03+i*.01,.3+i*.01,p.numel()).reshape_as(p))
        for p in self.parameters():p.requires_grad_(False)
        self.eval()
    def forward(self,x):return self.fc(self.pool(self.conv(x)).flatten(1))


class CPUAuditTests(unittest.TestCase):
    def test_legacy_tables_exact_and_single20_group_identity(self):
        records=[{'id':str(i).zfill(2)} for i in range(20)]
        for size in (1,2):
            expected=[]
            for start in range(0,20,size):
                members=[r['id'] for r in records[start:start+size]]
                identity='pair-'+hashlib.sha256(json.dumps(members,separators=(',',':')).encode()).hexdigest()
                expected.append({'batch_id':identity,'cluster_id':identity,'image_ids':members,'batch_size':size,'odd_singleton':False,'rng_seed':image_seed(28,identity)})
            self.assertEqual(pair_table(records,size,28),expected)
        group=pair_table(records[::-1],20,28,cpu_audit=True)
        self.assertEqual(len(group),1);self.assertTrue(group[0]['batch_id'].startswith('batch20-'))
        self.assertEqual(group[0]['image_ids'],[r['id'] for r in records]);self.assertEqual(group[0]['rng_seed'],image_seed(28,group[0]['batch_id']))
        self.assertEqual(group,pair_table(records,20,28,cpu_audit=True))
        for size,enabled,selected in [(20,False,records),(2,True,records),(20,True,records[:19]),(20,True,[*records[:-1],records[0]])]:
            with self.subTest(size=size,enabled=enabled),self.assertRaises(ValueError):pair_table(selected,size,28,cpu_audit=enabled)

    def test_explicit_audit_guard_and_contradictions(self):
        args=argparse.Namespace(ftm_batch20_cpu_audit=True,batch_size=20,device='cpu',split='pilot',ftm_input_profile='original299-v1',source='rn50_v1',ftm_layer_profile='mapped299')
        records=[{'id':str(i),'split':'pilot'} for i in range(20)]
        self.assertTrue(validate(args,records))
        for key,value in [('batch_size',2),('device','cuda:0'),('split','dev'),('split','pilot,holdout'),('source','rn50'),('ftm_input_profile','canonical224-v2'),('ftm_layer_profile','native224')]:
            bad=copy.copy(args);setattr(bad,key,value)
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):validate(bad,records)
        with self.assertRaises(ValueError):validate(args,records[:19])
        bad=copy.copy(args);bad.ftm_batch20_cpu_audit=False
        with self.assertRaises(ValueError):validate(bad,records)
        bad.batch_size=2;self.assertFalse(validate(bad,records[:2]))

    def test_thread_metadata_only_explicit_configuration_no_environment_mutation(self):
        with patch('torch.set_num_threads') as intra,patch('torch.set_num_interop_threads') as inter,patch('torch.get_num_threads',return_value=4),patch('torch.get_num_interop_threads',side_effect=[2,4,4]),patch.dict('os.environ',{'MKL_CBWR':'UNCHANGED','OMP_NUM_THREADS':'4'}):
            result=configure();intra.assert_called_once_with(4);inter.assert_called_once_with(4)
            self.assertEqual(result['environment']['MKL_CBWR'],'UNCHANGED')
            self.assertEqual((result['intraop_threads'],result['interop_threads']),(4,4))
            self.assertIn('torch_build_configuration',result)

    def test_windows_memory_abi_actual_return_and_failure(self):
        self.assertEqual(ProcessMemoryCounters.PageFaultCount.offset,4)
        self.assertEqual(ProcessMemoryCounters.PeakWorkingSetSize.offset,8)
        self.assertEqual(ctypes.sizeof(ProcessMemoryCounters),8+8*ctypes.sizeof(ctypes.c_size_t))
        current=MagicMock(return_value=ctypes.c_void_p(-1));measure=MagicMock()
        def measured(_handle,pointer,size):
            data=ctypes.cast(pointer,ctypes.POINTER(ProcessMemoryCounters)).contents
            self.assertEqual(size,ctypes.sizeof(ProcessMemoryCounters));self.assertEqual(data.cb,size)
            data.PeakWorkingSetSize=2**33+123;return 1
        measure.side_effect=measured
        loader=lambda name,**_kw:SimpleNamespace(GetCurrentProcess=current) if name=='kernel32' else SimpleNamespace(GetProcessMemoryInfo=measure)
        value=process_peak_memory('Windows',loader)
        self.assertEqual(value['PeakWorkingSet64'],2**33+123)
        self.assertEqual(current.restype,ctypes.c_void_p)
        self.assertEqual(measure.argtypes,[ctypes.c_void_p,ctypes.POINTER(ProcessMemoryCounters),ctypes.c_uint32])
        measure.side_effect=None;measure.return_value=0
        with self.assertRaises(OSError):process_peak_memory('Windows',loader)
        self.assertFalse(process_peak_memory('Linux')['available'])

    def test_batch20_author_exact_forward_gradient_u_rng_and_trace_toggle(self):
        # Native224 small-model differential; actual299/RN50V1 is root's
        # separate engineering audit. No weights/data/GPU in this test.
        with torch.random.fork_rng(devices=[]):
            clean=torch.linspace(.1,.7,20*3*224*224).reshape(20,3,224,224);targets=torch.arange(20)%3
            for probability in (.1,1.):
                with self.subTest(probability=probability):
                    cls,settings=author_class(probability);base=Small();author=cls(base,224,settings,'cpu')
                    torch.manual_seed(1234);reference=clean.clone();momentum=torch.zeros_like(clean)
                    outputs=[];gradients=[];states=[];selected=[]
                    try:
                        author.start_feature_record()
                        with torch.no_grad():author(clean)
                        author.end_feature_record()
                        for step in range(3):
                            x=reference.detach().requires_grad_(True);logits=author(rdi(x,None));outputs.append(logits.detach().clone())
                            names=[name for name,flag in author.mixing_triggered.items() if flag]
                            selected.append(dict(author.mixing_triggered))
                            grads=torch.autograd.grad(logits.gather(1,targets[:,None]).sum(),[x,*[author.outputs_tuning[name] for name in names]])
                            gradients.append(grads[0].detach().clone())
                            for name,g in zip(names,grads[1:]):author.outputs_tuning[name]=(author.outputs_tuning[name]-g).detach().requires_grad_(True)
                            states.append({name:v.detach().clone() for name,v in author.outputs_tuning.items()})
                            gradient=F.conv2d(grads[0],ti_kernel(clean),padding=2,groups=3);norm=gradient.abs().sum((1,2,3),keepdim=True)
                            momentum=momentum+gradient/torch.where(norm>0,norm,torch.ones_like(norm))
                            reference=torch.clamp(x.detach()+(.2/255)*momentum.sign(),(clean-16/255).clamp(0,1),(clean+16/255).clamp(0,1)).detach()
                        expected_rng=torch.get_rng_state().clone()
                    finally:author.remove_hooks()
                    for tracing in (False,True):
                        model=copy.deepcopy(base);generator=torch.Generator().manual_seed(1234);logits_seen=[];raw_norms={};hooks=[];index=[0];raw_grad=torch.autograd.grad
                        # Register before FTM so the independent norm observer
                        # sees each layer's original output, before mixing.
                        for name,module in author_layers(model):hooks.append(module.register_forward_hook(lambda _m,_i,o,key=name:raw_norms.__setitem__(key,o.detach().reshape(20,-1).norm(dim=1))))
                        try:
                            with FeatureTuning(model,generator,prob=probability,beta=.01,trace_updates=tracing) as context:
                                context.cache(clean);names=[name for name,_ in context.layers]
                                hooks.append(model.register_forward_hook(lambda _m,_i,o:logits_seen.append(o.detach().clone())))
                                def checked_grad(*a,**kw):
                                    result=raw_grad(*a,**kw);self.assertTrue(torch.equal(result[0],gradients[index[0]]));return result
                                def observed(step,image,ctx):
                                    self.assertTrue(torch.equal(logits_seen[-1],outputs[step-1]))
                                    self.assertEqual({names[k]:v for k,v in selected[step-1].items()},ctx.mixing_triggered)
                                    self.assertTrue(all(isinstance(flag,bool) for flag in ctx.mixing_triggered.values()))
                                    for key,value in states[step-1].items():
                                        self.assertTrue(torch.equal(value,ctx.outputs_tuning[names[key]]));self.assertTrue(torch.isfinite(value).all())
                                    if tracing:
                                        for key,norm in ctx.output_norms.items():self.assertTrue(torch.equal(norm,raw_norms[key]))
                                        self.assertTrue(any(torch.unique(v).numel()>1 for v in ctx.output_norms.values()))
                                    index[0]+=1
                                with patch('torch.autograd.grad',side_effect=checked_grad):
                                    actual,stats=optimize_ftm(context,clean,targets,3,16,.2,observer=observed)
                                self.assertEqual(stats['input_gradient_images'],60)
                                for name,count in stats['selected_layer_counts'].items():self.assertEqual(count,sum(row.get(names.index(name),False) for row in selected))
                            self.assertTrue(torch.equal(actual,reference));self.assertTrue(torch.equal(generator.get_state(),expected_rng))
                        finally:
                            for hook in hooks:hook.remove()
