"""Paired-bootstrap helper used by the paper figures."""
import unittest

from utils.paper_figures import flips_by_label, paired_difference


class PaperFigureTests(unittest.TestCase):
    def test_paired_difference_mean_and_interval(self):
        ids = [str(i) for i in range(10)]
        a = {('m', i): {'targeted_success': k < 6} for k, i in enumerate(ids)}
        b = {('m', i): {'targeted_success': k < 3} for k, i in enumerate(ids)}
        mean, lo, hi = paired_difference(a, b, 'm', ids, draws=500, seed=1)
        self.assertAlmostEqual(mean, 30.0)
        self.assertLessEqual(lo, mean); self.assertGreaterEqual(hi, mean)
        self.assertGreaterEqual(lo, 0.0)

    def test_multi_target_difference_averages_per_image(self):
        ids = ['a', 'b']
        a = {('m', 'a'): {'targeted_success': True}, ('n', 'a'): {'targeted_success': False},
             ('m', 'b'): {'targeted_success': True}, ('n', 'b'): {'targeted_success': True}}
        b = {key: {'targeted_success': False} for key in a}
        mean, lo, hi = paired_difference(a, b, ('m', 'n'), ids, draws=200, seed=1)
        self.assertAlmostEqual(mean, 75.0)
        self.assertGreaterEqual(lo, 50.0); self.assertLessEqual(hi, 100.0)

    def test_flips_by_label_groups_sorted_ids_by_winning_hand_off(self):
        # id: (reset success, inherit success, target label)
        rows = {'c': (False, True, 7), 'a': (True, False, 7), 'b': (False, True, 7), 'd': (True, False, 3),
                'e': (True, True, 3)}
        reset = {('t', i): {'targeted_success': r} for i, (r, _, _) in rows.items()}
        inherit = {('t', i): {'targeted_success': k} for i, (_, k, _) in rows.items()}
        labels = {i: lab for i, (_, _, lab) in rows.items()}
        self.assertEqual(flips_by_label(reset, inherit, 't', labels), {'inherit': {7: ['b', 'c']},
                                                                       'reset': {7: ['a'], 3: ['d']}})
        self.assertEqual(flips_by_label(reset, reset, 't', labels), {'inherit': {}, 'reset': {}})

    def test_identical_runs_give_zero(self):
        ids = ['a', 'b']
        a = {('m', i): {'targeted_success': True} for i in ids}
        self.assertEqual(paired_difference(a, a, 'm', ids, draws=50), (0.0, 0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
