import cv2
import os
import numpy as np


def map_densepose_to_tex(img, iuv_img, tex_res):
    if map_densepose_to_tex.lut is None:
        map_densepose_to_tex.lut = np.load(os.path.join(os.path.dirname(__file__), '../assets/dp_uv_lookup_256.npy'))

    iuv_raw = iuv_img[iuv_img[:, :, 0] > 0]
    data = img[iuv_img[:, :, 0] > 0]
    i = iuv_raw[:, 0] - 1

    if iuv_raw.dtype == np.uint8:
        u = iuv_raw[:, 1] / 255.
        v = iuv_raw[:, 2] / 255.
    else:
        u = iuv_raw[:, 1]
        v = iuv_raw[:, 2]

    u[u > 1] = 1.
    v[v > 1] = 1.

    uv_smpl = map_densepose_to_tex.lut[
        i.astype(np.int),
        np.round(v * 255.).astype(np.int),
        np.round(u * 255.).astype(np.int)
    ]

    tex = np.ones((tex_res, tex_res, img.shape[2])) * 0.5

    u_I = np.round(uv_smpl[:, 0] * (tex.shape[1] - 1)).astype(np.int32)
    v_I = np.round((1 - uv_smpl[:, 1]) * (tex.shape[0] - 1)).astype(np.int32)

    tex[v_I, u_I] = data

    return tex


map_densepose_to_tex.lut = None


def normalize(map):
    norm = np.linalg.norm(map, axis=-1)
    norm[norm == 0] = 1.

    return map / np.expand_dims(norm, -1)


def to_image_range(map, m_min=None, m_max=None):
    if m_min is None:
        m_min = np.min(map).astype(np.float32)
    if m_max is None:
        m_max = np.max(map).astype(np.float32)

    return np.clip((map - m_min) / (m_max - m_min), 0, 1)


def as_blender_normalmap(n_map):
    n_map = to_image_range(normalize(n_map), -1, 1)
    n_map[:, :, 1:] = 1 - n_map[:, :, 1:]
    return n_map


def save_as_blender_normalmap(filename, n_map):
    cv2.imwrite(filename, np.uint8(as_blender_normalmap(n_map)[:, :, ::-1] * 255))
