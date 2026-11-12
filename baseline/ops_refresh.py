"""Paper-loop noise lifecycle adaptation of the existing author OPS profile.

Only pool lifecycle differs. Existing OPSContext.step_plans is inherited intact;
its Python index/composite/parameter stream remains paired to the fixed profile.
"""
import hashlib
import random
import time

import torch

from .ops import OPSContext, RATIOS

PROFILE = 'ops-author-noiserefresh'


class OPSRefreshContext(OPSContext):
    def __init__(self, shape, epsilon, ne, np_, beta, seed):
        if ne < 0 or np_ < 0 or np_ > 93 or beta < 0:
            raise ValueError('OPS requires Ne>=0, 0<=Np<=93 and beta>=0')
        self.shape, self.ne, self.np = tuple(shape), ne, np_
        self.rng = random.Random(seed)
        self.noise_generator = torch.Generator(device='cpu').manual_seed(seed)
        self.epsilon, self.beta = epsilon, beta
        self.steps_used, self.pool_generation_count = 0, 0
        self.pool_generation_seconds, self.pool_sha256 = [], []
        self.pool = []
        if ne * np_:
            self._generate_pool()  # First pool is generated ONCE, and used by step1.

    def _generate_pool(self):
        start = time.perf_counter()
        pool = []
        digest = hashlib.sha256()
        for ratio in RATIOS:
            radius = self.beta * self.epsilon * ratio
            noise = torch.empty((self.ne, *self.shape[1:]), dtype=torch.float32).uniform_(
                -radius, radius, generator=self.noise_generator)
            pool.extend(noise[index:index + 1] for index in range(self.ne))
            digest.update(noise.contiguous().numpy().tobytes())
        self.pool = pool
        self.pool_generation_count += 1
        self.pool_sha256.append(digest.hexdigest())
        self.pool_generation_seconds.append(time.perf_counter() - start)

    def step_plans(self):
        if self.steps_used and self.ne * self.np:
            self._generate_pool()
        plans = super().step_plans()
        self.steps_used += 1
        return plans

    def lifecycle_record(self):
        return {'noise_pool_generation_count': self.pool_generation_count,
                'noise_pool_generation_seconds': sum(self.pool_generation_seconds),
                'noise_pool_generation_timing_includes_sha256': True,
                'noise_pool_generation_seconds_by_pool': self.pool_generation_seconds.copy(),
                'noise_pool_sha256_by_pool': self.pool_sha256.copy(),
                'noise_tensors_generated': self.pool_generation_count * 6 * self.ne,
                'noise_pool_steps_used': self.steps_used,
                'noise_generator': 'dedicated CPU float32 continuous stream; initial seed=image_seed',
                'profile': PROFILE}
