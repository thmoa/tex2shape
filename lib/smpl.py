#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import sys

import chumpy as ch
import scipy.sparse as sp

from chumpy.ch import Ch

from vendor.smpl.posemapper import posemap, Rodrigues
from vendor.smpl.serialization import backwards_compatibility_replacements

from ch_ext import sp_dot

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl


class Smpl(Ch):
    """
    Class to store SMPL object with slightly improved code and access to more matrices
    """
    terms = 'model',
    dterms = 'trans', 'betas', 'pose', 'v_personal'

    def __init__(self, *args, **kwargs):
        self.on_changed(self._dirty_vars)

    def on_changed(self, which):
        if 'model' in which:
            if not isinstance(self.model, dict):
                dd = pkl.load(open(self.model))
            else:
                dd = self.model

            backwards_compatibility_replacements(dd)

            # for s in ['v_template', 'weights', 'posedirs', 'pose', 'trans', 'shapedirs', 'betas', 'J']:
            for s in ['posedirs', 'shapedirs']:
                if (s in dd) and not hasattr(dd[s], 'dterms'):
                    dd[s] = ch.array(dd[s])

            self.f = dd['f']
            self.v_template = dd['v_template']
            if not hasattr(self, 'v_personal'):
                self.v_personal = ch.zeros_like(self.v_template)
            self.shapedirs = dd['shapedirs']
            self.J_regressor = dd['J_regressor']
            if 'J_regressor_prior' in dd:
                self.J_regressor_prior = dd['J_regressor_prior']
            self.bs_type = dd['bs_type']
            self.weights = dd['weights']
            if 'vert_sym_idxs' in dd:
                self.vert_sym_idxs = dd['vert_sym_idxs']
            if 'weights_prior' in dd:
                self.weights_prior = dd['weights_prior']
            self.kintree_table = dd['kintree_table']
            self.posedirs = dd['posedirs']

            if not hasattr(self, 'betas'):
                self.betas = ch.zeros(self.shapedirs.shape[-1])

            if not hasattr(self, 'trans'):
                self.trans = ch.zeros(3)

            if not hasattr(self, 'pose'):
                self.pose = ch.zeros(72)

            self._set_up()

    def _set_up(self):
        self.v_shaped = self.shapedirs.dot(self.betas) + self.v_template

        body_height = (self.v_shaped[2802, 1] + self.v_shaped[6262, 1]) - (
                self.v_shaped[2237, 1] + self.v_shaped[6728, 1])
        self.scale = 1.66 / body_height

        self.v_shaped_personal = self.scale * self.v_shaped + self.v_personal

        if sp.issparse(self.J_regressor):
            self.J = self.scale * sp_dot(self.J_regressor, self.v_shaped)
        else:
            self.J = self.scale * ch.sum(self.J_regressor.T.reshape(-1, 1, 24) * self.v_shaped.reshape(-1, 3, 1), axis=0).T
        self.v_posevariation = self.posedirs.dot(posemap(self.bs_type)(self.pose))
        self.v_poseshaped = self.v_shaped_personal + self.v_posevariation

        self.A, A_global = self._global_rigid_transformation()
        self.Jtr = ch.vstack([g[:3, 3] for g in A_global])
        self.J_transformed = self.Jtr + self.trans.reshape((1, 3))

        self.V = self.A.dot(self.weights.T)

        rest_shape_h = ch.hstack((self.v_poseshaped, ch.ones((self.v_poseshaped.shape[0], 1))))
        self.v_posed = ch.sum(self.V.T * rest_shape_h.reshape(-1, 4, 1), axis=1)[:, :3]
        self.v = self.v_posed + self.trans

    def _global_rigid_transformation(self):
        results = {}
        pose = self.pose.reshape((-1, 3))
        parent = {i: self.kintree_table[0, i] for i in range(1, self.kintree_table.shape[1])}

        with_zeros = lambda x: ch.vstack((x, ch.array([[0.0, 0.0, 0.0, 1.0]])))
        pack = lambda x: ch.hstack([ch.zeros((4, 3)), x.reshape((4, 1))])

        results[0] = with_zeros(ch.hstack((Rodrigues(pose[0, :]), self.J[0, :].reshape((3, 1)))))

        for i in range(1, self.kintree_table.shape[1]):
            results[i] = results[parent[i]].dot(with_zeros(ch.hstack((
                Rodrigues(pose[i, :]),      # rotation around bone endpoint
                (self.J[i, :] - self.J[parent[i], :]).reshape((3, 1))     # bone
            ))))

        results = [results[i] for i in sorted(results.keys())]
        results_global = results

        # subtract rotated J position
        results2 = [results[i] - (pack(
            results[i].dot(ch.concatenate((self.J[i, :], [0]))))
        ) for i in range(len(results))]
        result = ch.dstack(results2)

        return result, results_global

    def compute_r(self):
        return self.v.r

    def compute_dr_wrt(self, wrt):
        if wrt is not self.trans and wrt is not self.betas and wrt is not self.pose and wrt is not self.v_personal:
            return None

        return self.v.dr_wrt(wrt)
