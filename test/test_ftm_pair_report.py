"""CPU fixtures for fixed-pair inference and source-only repeat identities."""
import argparse
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from baseline.ftm import layer_digest
from utils.determinism import configure_strict
from utils.ftm_pair_report import (check_configs,cluster_estimate,load_attack,
    load_evaluations,repeat_report,pair_report)
from utils.io import sha256,write_json
from utils.targeted_ftm import pair_table


def config(ids=('a','b','c','d'),seed=20261030,mode='full'):
    pairs=pair_table([{'id':i} for i in ids],2,seed)
    return {'kind':'targeted-attack','arguments':{'method':'ftm','source':'rn50','seed':seed,
        'batch_size':2,'steps':299,'epsilon_bytes':16.,'alpha_bytes':2.,'ids':None,'limit':None,
        'ftm_update_mode':mode,'deterministic':True},'selected_ids':list(ids),
        'target_labels':{i:7 for i in ids},'pairs':pairs,'pair_table_sha256':layer_digest(pairs),
        'update_mode':mode,'profile':'ftm-author-config-mapped299-local224-batch2-update-'+mode,
        'layer_table_sha256':'layers','code_sha256':{'baseline/ftm.py':'frozen'},
        'manifest_sha256':'manifest','labels_csv_sha256':'labels',
        'numeric_settings':{'deterministic_algorithms':True,'deterministic_warn_only':False,
            'cudnn_deterministic':True},'source_identity':{'weight_sha256':'weight'}}


def artifact(root,cfg,changed=False):
    root.mkdir(parents=True);write_json(root/'config.json',cfg);rows=[]
    for pair in cfg['pairs']:
        batch=root/(pair['batch_id']+'.json')
        write_json(batch,{**pair,'layer_table_sha256':'layers','final_generator_state_sha256':'rng',
            'elapsed_seconds':1.,'peak_cuda_allocated_bytes':123,'forward_images':606,'input_gradient_images':598})
        for i in pair['image_ids']:
            png=root/(i+'.png');floating=root/(i+'.npz')
            value=1 if changed and i=='a' else 0
            Image.fromarray(np.full((2,2,3),value,dtype=np.uint8)).save(png)
            np.savez_compressed(floating,image=np.full((1,3,2,2),value/255,dtype=np.float32))
            entry={'png_path':str(png),'png_sha256':sha256(png),'float_path':str(floating),'float_sha256':sha256(floating)}
            rows.append({'id':i,'label':0,'target_label':7,'batch_id':pair['batch_id'],
                'batch_rng_seed':pair['rng_seed'],'layer_table_sha256':'layers',
                'shared_batch_total_path':str(batch),'shared_batch_total_sha256':sha256(batch),
                'source_clean':{'true_correct':True},'checkpoints':{'299':entry},
                'forward_images':303,'input_gradient_images':299})
    (root/'progress.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')


class PairStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.pairs=[{'batch_id':'p1','image_ids':['a','b']},{'batch_id':'p2','image_ids':['c','d']}]
        self.indices=np.array([[0,0],[0,1],[1,0],[1,1]])
        values=[(20261030,[(1,0),(0,1),(1,0),(1,0)]),(20261031,[(1,0),(0,0),(0,1),(0,0)])]
        self.rows=[{'seed':s,'id':i,'full':f,'control':c,'eligible':True}
            for s,items in values for i,(f,c) in zip('abcd',items)]

    def test_seed_mean_before_pair_bootstrap_and_opposite_flips(self):
        result=cluster_estimate(self.rows,self.pairs,[20261030,20261031],self.indices,True)
        self.assertEqual(result['difference'],.25)
        self.assertEqual(result['bootstrap']['low'],.25)
        self.assertTrue(result['bootstrap']['degenerate'])
        self.assertEqual(result['discordant_pair_count'],2)
        self.assertIsNone(result['zero_discordant_pair_binomial_reference_upper'])
        single=cluster_estimate(self.rows,self.pairs,[20261030],self.indices,True)
        self.assertEqual(single['difference'],.5)

    def test_conditional_ratio_of_sums_and_empty_draws(self):
        rows=[{**r,'eligible':r['id'] in 'acd'} for r in self.rows]
        result=cluster_estimate(rows,self.pairs,[20261030,20261031],self.indices,True)
        self.assertEqual(result['difference'],.5)
        expected=np.percentile([1.,.5,.5,.25],[2.5,97.5])
        np.testing.assert_allclose([result['bootstrap']['low'],result['bootstrap']['high']],expected)
        rows=[{**r,'eligible':r['id']=='a'} for r in rows]
        result=cluster_estimate(rows,self.pairs,[20261030,20261031],self.indices,True)
        self.assertEqual(result['bootstrap']['valid_draws'],3)
        self.assertEqual(result['bootstrap']['empty_denominator_draws'],1)

    def test_partial_no_ci_and_zero_discordance_reference(self):
        rows=[{**r,'control':r['full']} for r in self.rows]
        partial=cluster_estimate(rows,self.pairs,[20261030,20261031],self.indices,False)
        self.assertIsNone(partial['bootstrap'])
        self.assertIsNone(partial['zero_discordant_pair_binomial_reference_upper'])
        full=cluster_estimate(rows,self.pairs,[20261030,20261031],self.indices,True)
        self.assertAlmostEqual(full['zero_discordant_pair_binomial_reference_upper']['upper'],1-.025**.5)
        missing=[r for r in self.rows if not(r['seed']==20261031 and r['id']=='a')]
        partial=cluster_estimate(missing,self.pairs,[20261030,20261031],self.indices,False)
        self.assertNotIn('a',partial['matched_image_ids_across_selected_seeds'])


class PairIdentityTests(unittest.TestCase):
    def test_only_declared_mode_seed_and_complete_pair_subset_differences(self):
        base=config();check_configs(base,config(mode='first-nonzero-only'),allow_mode=True)
        check_configs(base,config(seed=20261031),allow_seed=True)
        subset=config(('a','b'));subset['arguments']['ids']='a,b'
        check_configs(base,subset,subset=True)
        for key in ('numeric_settings','code_sha256','source_identity','layer_table_sha256'):
            bad=copy.deepcopy(base);bad[key]='drift'
            with self.assertRaises(ValueError):check_configs(base,bad,allow_mode=True,allow_seed=True)
        with self.assertRaises(ValueError):check_configs(base,config(('b','c')),subset=True)
        bad=copy.deepcopy(base);bad['arguments']['deterministic']=False
        with self.assertRaises(ValueError):check_configs(base,bad,subset=True)

    def test_repeat_retains_float_png_differences_and_metadata_sha(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);a=root/'a';b=root/'b';artifact(a,config());artifact(b,config(('a','b')),True)
            with patch('utils.runner._output',side_effect=lambda p:Path(p).mkdir(parents=True,exist_ok=True) or Path(p)):
                repeat_report(argparse.Namespace(run_a=str(a),run_b=str(b),output=str(root/'report')))
            report=json.loads((root/'report/report.json').read_text())
            self.assertEqual(report['status'],'complete');self.assertEqual(len(report['image_comparisons']),2)
            changed=next(r for r in report['image_comparisons'] if r['id']=='a')
            self.assertFalse(changed['png_exact_equal']);self.assertFalse(changed['float_exact_equal'])
            row=load_attack(a)['rows']['a'];Path(row['shared_batch_total_path']).write_text('{}')
            with self.assertRaisesRegex(ValueError,'SHA'):load_attack(a)

    def test_evaluation_binding_and_empty_report_partial(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);a=root/'a';artifact(a,config());attack=load_attack(a);evaluation=root/'eval';evaluation.mkdir()
            write_json(evaluation/'config.json',{'kind':'targeted-evaluate','attack_config_sha256':'wrong','attack_progress_sha256':attack['progress_sha256']})
            with self.assertRaisesRegex(ValueError,'binding'):load_evaluations([evaluation],attack,['dn121'])
            plan=root/'plan.json';write_json(plan,{'cohort_ids':list('abcd'),'targets':['dn121'],'cells':[
                {'seed':s,'mode':m,'attack_run':str(root/f'{s}-{m}')} for s in (20261030,20261031) for m in ('full','first-nonzero-only')]})
            with patch('utils.runner._output',side_effect=lambda p:Path(p).mkdir(parents=True,exist_ok=True) or Path(p)):
                pair_report(argparse.Namespace(plan=str(plan),output=str(root/'report')))
            report=json.loads((root/'report/report.json').read_text())
            self.assertEqual(report['status'],'partial')
            self.assertTrue(all(r['bootstrap'] is None for r in report['statistics']))


class StrictDeterminismTests(unittest.TestCase):
    def test_default_noop_and_cuda_environment_rejection_without_cuda_init(self):
        with patch('torch.use_deterministic_algorithms') as use:
            configure_strict(argparse.Namespace(deterministic=False,device='cuda:0'));use.assert_not_called()
            with patch.dict('os.environ',{'CUBLAS_WORKSPACE_CONFIG':''}),patch('torch.cuda.is_initialized') as initialized:
                with self.assertRaises(ValueError):configure_strict(argparse.Namespace(deterministic=True,device='cuda:0'))
                initialized.assert_not_called();use.assert_not_called()

    def test_explicit_strict_cpu_calls_error_mode_and_preserves_environment(self):
        flags=(torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
            torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        try:
            with patch('torch.use_deterministic_algorithms') as use,patch('torch.cuda.is_initialized') as initialized:
                configure_strict(argparse.Namespace(deterministic=True,device='cpu'))
                use.assert_called_once_with(True,warn_only=False);initialized.assert_not_called()
                self.assertTrue(torch.backends.cudnn.deterministic);self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            (torch.backends.cudnn.deterministic,torch.backends.cudnn.benchmark,
                torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)=flags
