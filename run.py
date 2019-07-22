from __future__ import print_function

import cv2
import sys
import os
import argparse

import numpy as np

from glob import glob

from models.tex2shape_model import Tex2ShapeModel
from models.betas_model import BetasModel
from lib.mesh_from_maps import MeshFromMaps

from lib import mesh
from lib.maps import map_densepose_to_tex, normalize

if sys.version_info[0] == 3:
    import _pickle as pkl
else:
    import cPickle as pkl


def main(img_files, iuv_files, out_dir, weights_tex2shape, weights_betas):

    if os.path.isfile(img_files) != os.path.isfile(iuv_files):
        print('Inconsistent input.')
        exit(1)

    tex2shape_model = Tex2ShapeModel()
    betas_model = BetasModel()

    tex2shape_model.load(weights_tex2shape)
    betas_model.load(weights_betas)

    mfm = MeshFromMaps()

    if os.path.isfile(img_files):
        img_files = [img_files]
        iuv_files = [iuv_files]
    else:
        img_files = sorted(glob(os.path.join(img_files, '*.png')) + glob(os.path.join(img_files, '*.jpg')))
        iuv_files = sorted(glob(os.path.join(iuv_files, '*.png')))

    for img_file, iuv_file in zip(img_files, iuv_files):

        img = cv2.imread(img_file) / 255.
        iuv_img = cv2.imread(iuv_file)
        unwrap = np.expand_dims(map_densepose_to_tex(img, iuv_img, 512), 0)

        name = os.path.splitext(os.path.basename(img_file))[0]

        print('Processing {}...'.format(name))

        iuv_img = iuv_img * 1.
        iuv_img[:, :, 1:] /= 255.
        iuv_img = np.expand_dims(iuv_img, 0)

        print('> Estimating normal and displacement maps...')
        pred = tex2shape_model.predict(unwrap)

        print('> Estimating betas...')
        betas = betas_model.predict(iuv_img)

        print('> Saving maps and betas...')
        pkl.dump({
            'normal_map': normalize(pred[0, :, :, :3]),
            'displacement_map': pred[0, :, :, 3:] / 10.,
            'betas': betas[0],
        }, open('{}/{}.pkl'.format(out_dir, name), 'wb'), protocol=2)

        print('> Baking obj file for easy inspection...')
        m = mfm.get_mesh(pred[0, :, :, :3], pred[0, :, :, 3:] / 10, betas=betas[0])
        mesh.write('{}/{}.obj'.format(out_dir, name), v=m['v'], f=m['f'], vn=m['vn'], vt=m['vt'], ft=m['ft'])

        print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'image',
        type=str,
        help="Input image or folder")

    parser.add_argument(
        'iuv',
        type=str,
        help="Densepose IUV image or folder")

    parser.add_argument(
        '--out_dir', '-od',
        default='out',
        help='Output directory')

    parser.add_argument(
        '--weights_tex2shape', '-wt',
        default='weights/tex2shape_weights.hdf5',
        help='Tex2shape model weights file (*.hdf5)')

    parser.add_argument(
        '--weights_betas', '-wb',
        default='weights/betas_weights.hdf5',
        help='Betas model weights file (*.hdf5)')

    args = parser.parse_args()
    main(args.image, args.iuv, args.out_dir, args.weights_tex2shape, args.weights_betas)
