
import random
import logging

import numpy as np

from ...base_optimizer import BaseOptimizer
from ...base_positioner import BasePositioner

# Set up the logger
logger = logging.getLogger('particle_logger')
logger.setLevel(logging.DEBUG)

# Create a file handler to save logs to a file
file_handler = logging.FileHandler('particle_movement.log')
file_handler.setLevel(logging.DEBUG)

# Create a formatter and set it for the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the logger
logger.addHandler(file_handler)

class HPSO(BaseOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_best_gscore = None

    def _init_particles(self, _cand_):
        _p_list_ = [Particle() for _ in range(self._arg_.n_part)]
        for i, _p_ in enumerate(_p_list_):
            _p_.nr = i
            _p_.pos_current = _cand_._space_.get_random_pos()
            _p_.pos_best = _p_.pos_current
            _p_.velo = np.zeros(len(_cand_._space_.para_space))

        return _p_list_

    def _move_particles(self, _cand_, _p_list_, iter_count):
        if self.prev_best_gscore is not None and self.prev_best_gscore == _cand_.score_best:
            w_max, w_min = 0.9, 0.4
            D = (10 - iter_count) / 10  # 最大反復数は手動
            E = w_max - w_min
            w = D * E + w_min
            print("weight :{}".format(w))
        else:
            w = 0.7
        logger.info("Weight (w) calculated: {}".format(w)) 

        for _p_ in _p_list_:
            r1, r2 = random.random(), random.random()

            A = w * _p_.velo
            B = self._arg_.c_k * r1 * np.subtract(_p_.pos_best, _p_.pos_current)
            C = self._arg_.c_s * r2 * np.subtract(_cand_.pos_best, _p_.pos_current)

            new_velocity = A + B + C

            _p_.velo = new_velocity
            _p_.pos_new = _p_.move_part(_cand_, _p_.pos_current)

    def _eval_particles(self, _cand_, _p_list_, X, y):
        for _p_ in _p_list_:
            _p_.score_new = _cand_.eval_pos(_p_.pos_new, X, y)

            if _p_.score_new > _cand_.score_best:
                _cand_, _p_ = self._update_pos(_cand_, _p_)

    def _iterate(self, i, _cand_, _p_list_, X, y):
        self._move_particles(_cand_, _p_list_, i)
        self._eval_particles(_cand_, _p_list_, X, y)

        self.prev_best_gscore = _cand_.score_best
        logger.info("Score calculated: {}".format(_cand_.score_best)) 
         

        return _cand_

    def _init_opt_positioner(self, _cand_, X, y):
        _p_list_ = self._init_particles(_cand_)

        return _p_list_


class Particle(BasePositioner):
    def __init__(self):
        self.nr = None
        self.velo = None

    def move_part(self, _cand_, pos):
        pos_new = (pos + self.velo).astype(int)
        # limit movement
        n_zeros = [0] * len(_cand_._space_.dim)
        return np.clip(pos_new, n_zeros, _cand_._space_.dim)
