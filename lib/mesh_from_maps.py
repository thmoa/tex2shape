import sys
import os

import numpy as np
import scipy.sparse as sp

from lib.smpl import Smpl
from lib.maps import normalize

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl


class MeshFromMaps:

    def __init__(self):

        with open(os.path.join(os.path.dirname(__file__), '../assets/hres.pkl'), 'r') as f:
            self.hres = pkl.load(f)

        with open(os.path.join(os.path.dirname(__file__), '../assets/neutral_smpl.pkl'), 'r') as f:
            model_data = pkl.load(f)

        model_data = self._make_hres(model_data)

        self.vt = self.hres['vt']
        self.ft = self.hres['ft']
        self.smpl = Smpl(model_data)

        self._prepare()

    def _make_hres(self, dd):
        hv = self.hres['v']
        hf = self.hres['f']
        mapping = self.hres['mapping']
        num_betas = dd['shapedirs'].shape[-1]
        J_reg = dd['J_regressor'].asformat('csr')

        model = {
            'v_template': hv,
            'weights': np.hstack([
                np.expand_dims(
                    np.mean(
                        mapping.dot(np.repeat(np.expand_dims(dd['weights'][:, i], -1), 3)).reshape(-1, 3)
                        , axis=1),
                    axis=-1)
                for i in range(24)
            ]),
            'posedirs': mapping.dot(dd['posedirs'].reshape((-1, 207))).reshape(-1, 3, 207),
            'shapedirs': mapping.dot(dd['shapedirs'].reshape((-1, num_betas))).reshape(-1, 3, num_betas),
            'J_regressor': sp.csr_matrix((J_reg.data, J_reg.indices, J_reg.indptr), shape=(24, hv.shape[0])),
            'kintree_table': dd['kintree_table'],
            'bs_type': dd['bs_type'],
            'J': dd['J'],
            'f': hf,
        }

        return model

    def _prepare(self):
        vt_per_v = {i: [] for i in range(len(self.smpl))}

        for ff, fft in zip(self.smpl.f, self.ft):
            for vt0, v0 in zip(fft, ff):
                vt_per_v[v0].append(vt0)

        self.single_vt_keys = []
        self.single_vt_val = []
        self.dual_vt_keys = []
        self.dual_vt_val = []
        self.triple_vt_keys = []
        self.triple_vt_val = []
        self.quadruple_vt_keys = []
        self.quadruple_vt_val = []

        self.multi_vt = {}

        for v in vt_per_v.keys():
            vt_list = np.unique(vt_per_v[v])

            if len(vt_list) == 1:
                self.single_vt_keys.append(v)
                self.single_vt_val.append(vt_list[0])
            elif len(vt_list) == 2:
                self.dual_vt_keys.append(v)
                self.dual_vt_val.append(vt_list)
            elif len(vt_list) == 3:
                self.triple_vt_keys.append(v)
                self.triple_vt_val.append(vt_list)
            elif len(vt_list) == 4:
                self.quadruple_vt_keys.append(v)
                self.quadruple_vt_val.append(vt_list)
            else:
                self.multi_vt[v] = vt_list

    def _lookup(self, uv, map):
        ui = np.round(uv[:, 0] * (map.shape[1] - 1)).astype(np.int32)
        vi = np.round((1 - uv[:, 1]) * (map.shape[0] - 1)).astype(np.int32)

        return map[vi, ui]

    def get_mesh(self, n_map, d_map, betas=None, pose=None):
        n_map = normalize(n_map)

        normals = np.zeros_like(self.smpl)
        displacements = np.zeros_like(self.smpl)

        normals[self.single_vt_keys] = self._lookup(self.vt[self.single_vt_val], n_map)
        normals[self.dual_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.dual_vt_val).ravel()], n_map).reshape((-1, 2, 3)),
            axis=1)
        normals[self.triple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.triple_vt_val).ravel()], n_map).reshape((-1, 3, 3)),
            axis=1)
        normals[self.quadruple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.quadruple_vt_val).ravel()], n_map).reshape((-1, 4, 3)),
            axis=1)

        for v in self.multi_vt:
            normals[v] = np.mean(self._lookup(self.vt[self.multi_vt[v]], n_map), axis=0)

        displacements[self.single_vt_keys] = self._lookup(self.vt[self.single_vt_val], d_map)
        displacements[self.dual_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.dual_vt_val).ravel()], d_map).reshape((-1, 2, 3)),
            axis=1)
        displacements[self.triple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.triple_vt_val).ravel()], d_map).reshape((-1, 3, 3)),
            axis=1)
        displacements[self.quadruple_vt_keys] = np.mean(
            self._lookup(self.vt[np.array(self.quadruple_vt_val).ravel()], d_map).reshape((-1, 4, 3)), axis=1)

        for v in self.multi_vt:
            displacements[v] = np.mean(self._lookup(self.vt[self.multi_vt[v]], d_map), axis=0)

        if betas is not None:
            self.smpl.betas[:] = betas
        else:
            self.smpl.betas[:] = 0

        if pose is not None:
            self.smpl.pose[:] = pose
            normals_H = np.hstack((normals, np.zeros((normals.shape[0], 1))))
            normals = np.sum(self.smpl.V.T.r * normals_H.reshape((-1, 4, 1)), axis=1)[:, :3]
        else:
            self.smpl.pose[:] = 0

        self.smpl.v_personal[:] = displacements

        return {
            'v': self.smpl.r,
            'f': self.smpl.f,
            'vn': normals,
            'vt': self.vt,
            'ft': self.ft,
        }
