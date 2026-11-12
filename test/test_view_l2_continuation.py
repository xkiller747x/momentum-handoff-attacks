"""CPU-only scheduler/statistics tests; no model or CUDA initialization."""
import tempfile
import unittest
from pathlib import Path
from utils.io import sha256
from utils.view_l2_continuation import tasks,verify_hashes,may_start,parsed,contrast,SEEDS


class ContinuationTests(unittest.TestCase):
    def test_three_seed_order_paths_and_cost(self):
        ids=[str(i) for i in range(276)];rows=tasks(ids,'profile','seed')
        self.assertEqual([(r['seed'],r['block']) for r in rows],[(s,b) for s in SEEDS for b in range(1,5)])
        self.assertEqual(Path(rows[0]['old_attack']),Path('profile/attacks/b1'))
        self.assertEqual(Path(rows[4]['old_attack']),Path('seed/attacks/s20261028_b1/refresh'))
        self.assertEqual(sum(r['source_forward_images'] for r in rows),215280)
        self.assertEqual(sum(r['target_forward_images'] for r in rows),3312)
        self.assertEqual([i for r in rows[:4] for i in r['selected_ids']],ids)

    def test_sha_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'proof';path.write_bytes(b'original');pins={str(path):sha256(path)}
            verify_hashes(pins);path.write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_hashes(pins)

    def test_partial_has_no_main_estimate_even_with_two_seeds(self):
        rows=[{'seed':s,'id':'a','difference':1.} for s in SEEDS[:2]]
        result=contrast(rows,['a'],SEEDS,False)
        self.assertIsNone(result['difference']);self.assertIsNone(result['ci'])

    def test_image_seed_averaging(self):
        rows=[{'seed':s,'id':i,'difference':value} for s,values in zip(SEEDS,[(1.,0.),(0.,1.),(1.,1.)]) for i,value in zip(['a','b'],values)]
        result=contrast(rows,['a','b'],SEEDS,True)
        self.assertAlmostEqual(result['difference'],2/3)
        self.assertEqual(result['ci']['image_count'],2)

    def test_clock_boundaries(self):
        plan={'freeze':{'boundary_cap_seconds':6000,'initial_projection_seconds':5549.581374482827,
            'last_new_block_utc':'2026-10-27T06:30:11Z','last_estimated_block_end_utc':'2026-10-27T06:45:11Z'}}
        state={'blocks':{},'attempts':[]}
        self.assertTrue(may_start(plan,state,parsed('2026-10-27T05:12:41Z'),0)[0])
        self.assertFalse(may_start(plan,state,parsed('2026-10-27T05:12:42Z'),0)[0])
        state['attempts']=[{}]
        self.assertTrue(may_start(plan,state,parsed('2026-10-27T06:30:11Z'),5999)[0])
        self.assertFalse(may_start(plan,state,parsed('2026-10-27T06:30:11.001Z'),0)[0])
        self.assertFalse(may_start(plan,state,parsed('2026-10-27T06:00:11Z'),6000)[0])
        state['blocks']={'x':{'status':'complete','total_attempt_wall_seconds':901}}
        self.assertFalse(may_start(plan,state,parsed('2026-10-27T06:30:11Z'),0)[0])
        state['first_block_projected_over_limit']=True
        self.assertFalse(may_start(plan,state,parsed('2026-10-27T05:30:11Z'),0)[0])
