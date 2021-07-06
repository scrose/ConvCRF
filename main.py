"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
Source: https://github.com/MarvinTeichmann/ConvCRF
"""

import sys
import os
import logging
import numpy as np
from fullcrf import fullcrf
from preprocess.data import Dataset
from utils import utils
from convcrf import convcrf

from tqdm import trange
from pydensecrf import densecrf as dcrf, utils as crf_utils
# from pydensecrf import densecrf as dcrf, utils as crf_utils
# import pyximport
# pyximport.install()

# Stdout log messages
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


# # Optimize CRF parameters through gradient training
# def crf_learn(args):
#     # initialize input data
#     data = Dataset(args)
#
#     # Calculate initial mIoU with ground-truth
#     unary_data = np.moveaxis(data.unary, 0, -1)
#     print("Unary mIoU loss = {}".format(data.loss(unary_data)))
#
#     # Get the labeling
#     # 	VectorXs labeling = getLabeling( anno, mask.rows*mask.cols, M );
#     # 	print(labeling, mask.cols, mask.rows);
#
#     # -------------------------------------------------------
#     # Initialize fully-connected CRF model instance
#     logging.info("Build FullCRF.")
#     crf = fullcrf.FullCRF(data.shape, data.n_classes)
#
#     # Initialize label-compatibilites `µ(xi, xj)`
#     compat_g = 1
#     compat_b = np.identity(data.n_classes)
#
#     # Compute indices for the lattice approximation.
#     # input image format: [HWCh]
#     logging.info("Starting Computation.")
#     crf.compute_lattice(data.img_resized, compat_g, compat_b)
#
#     # Initialize loss function
#     # Choose your loss function
#     # LogLikelihood objective( labeling, 0.01 ); // Log likelihood loss
#     # Hamming objective( labeling, 0.0 ); // Global accuracy
#     # Hamming objective( labeling, 1.0 ); // Class average accuracy
#     # Hamming objective( labeling, 0.2 ); // Hamming loss close to intersection over union
#
#     # Intersection over union accuracy
#     labeling = np.array([[1, 0, 0, 1, 1, 0, 1, 1, 1]])
#     iou = None
#
#     NIT = 5
#     verbose = True
#     learning_params = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]])
#
#     # Optimize the CRF in 3 phases:
#     # - Unary only
#     # - Unary and pairwise
#     # - Full CRF
#
#     for i in range(learning_params.shape[0]):
#         logging.info("Learning Step: {}".format(i))
#         # Return the parameters
#         logging.info("Unary parameters: {}".format(crf.unaryParameters()))
#         logging.info("Pairwise parameters: {}".format(crf.labelCompatibilityParameters()))
#         logging.info("Kernel parameters: {}".format(crf.kernelParameters()))
#         #
#         # Setup the energy
#         e = energy.EnergyCRF(
#             crf, iou, NIT, learning_params[i][0], learning_params[i][1], learning_params[i][2]
#         )
#         # e.setL2Norm(1e-3)
#         #
#         # # Minimize the energy
#         # p = minimizeLBFGS(energy, 2, true)
#         #
#         # # Save the values
#         # id = 0
#         # if learning_params[i][0]:
#         #     crf.setUnaryParameters(p.segment(id, crf.unaryParameters().rows()));
#         #     id += crf.unaryParameters().rows();
#         #
#         # if learning_params[i][1]:
#         #     crf.setLabelCompatibilityParameters(p.segment(id, crf.labelCompatibilityParameters().rows()));
#         #     id += crf.labelCompatibilityParameters().rows();
#         #
#         # if learning_params[i][2]:
#         #     crf.setKernelParameters(p.segment(id, crf.kernelParameters().rows()));
#         #
#         # # Return the parameters
#         # logging.info("Unary parameters: {}".format(crf.unaryParameters().transpose()))
#         # logging.info("Pairwise parameters: {}".format(crf.labelCompatibilityParameters().transpose()))
#         # logging.info("Kernel parameters: {}".format(crf.kernelParameters().transpose()))
#         #
#         # # Do map inference
#         # map = crf.map(NIT)
#
#         return


# ----------------------------------------
# Apply model for inference
# ----------------------------------------
def crf_inference(args):
    # Initialize input image/mask
    data = Dataset(args)

    # # -------------------------------------------------------
    # # Initialize fully-connected CRF model instance
    # logging.info("Build FullCRF.")
    # crf = fullcrf.FullCRF(data.shape, data.n_classes)
    #
    # # Compute indices for the lattice approximation.
    # # input image format: [HWCh]
    # logging.info("Starting Computation.")
    # crf.compute_lattice(data.img_resized)
    #
    # # Compute CRF inference
    # crf_prob = crf.compute(data.unary, data.img_resized)

    logging.info("Build ConvCRF.")
    ##
    # Create CRF module
    gausscrf = convcrf.GaussCRF(conf=config, shape=data.shape, nclasses=data.n_classes)
    # Cuda computation is required.
    # A CPU implementation of our message passing is not provided.
    gausscrf.cuda()

    logging.info("Start Computation.")
    # Perform CRF inference
    prediction = gausscrf.forward(unary=unary_var, img=img_var)

    # Combine unary + crf predictions
    merged_data = data.unary * crf_prob
    merged_data = np.expand_dims(merged_data, axis=0)
    unary_data = np.expand_dims(data.unary, axis=0)
    crf_prob = np.expand_dims(crf_prob, axis=0)

    # initialize model/output files for prediction results
    fname = os.path.basename(args.img_path).replace('.', '_')

    # output mask image
    outfile = os.path.join(args.out_path, fname + '_' + '_merged.png')
    utils.output_img(merged_data, data.n_classes, data.w_resized, data.h_resized, outfile)

    outfile = os.path.join(args.out_path, fname + '_' + '_unary.png')
    utils.output_img(unary_data, data.n_classes, data.w_resized, data.h_resized, outfile)

    outfile = os.path.join(args.out_path, fname + '_' + '_crf.png')
    utils.output_img(crf_prob, data.n_classes, data.w_resized, data.h_resized, outfile)

    # Calculate mIoU with ground-truth
    if args.mask_path:
        crf_prob = np.moveaxis(crf_prob, 0, -1)
        print("mIoU loss = {}".format(data.loss(crf_prob)))

    exit(0)


def get_parser():
    from argparse import ArgumentParser

    arg_parser = ArgumentParser()

    arg_parser.add_argument("--img_path", type=str,
                            help="Input image")

    arg_parser.add_argument("--mask_path", type=str,
                            help="Ground-truth mask.")

    arg_parser.add_argument("--unary_path", type=str,
                            help="Unary data file.")

    arg_parser.add_argument("--out_path", type=str,
                            help="output image")

    arg_parser.add_argument("--in_channels", type=int, default=1,
                            help="Image channels.")

    arg_parser.add_argument("--n_classes", type=int, default=9,
                            help="Number of classes.")

    arg_parser.add_argument("--isolate", type=int, default=None,
                            help="Isolate class for segmentation.")

    arg_parser.add_argument("--merge", type=bool, default=False,
                            help="Merge dataset classes.")

    arg_parser.add_argument('--output', type=str,
                            help="Optionally save output as img.")

    arg_parser.add_argument('--normalize', type=bool, default=False,
                            help="Normalize input image before inference.")

    arg_parser.add_argument('--resample', type=int, default=None,
                            help="Resample factor for input image (multiplier of tile size).")

    arg_parser.add_argument('--mode', type=str, default='inference',
                            choices=["inference", "learn"],
                            help="Operation mode of the CRF.")

    arg_parser.add_argument("--iterations", type=int, default=5,
                            help="Number of training or inference iterations.")

    return arg_parser


if __name__ == '__main__':

    # input parameters
    parser = get_parser()
    input_args = parser.parse_args()

    # Compute CRF inference
    if input_args.mode == 'learn':
        logging.info("===== Starting parameter training.")
        #crf_learn(input_args)
    elif input_args.mode == 'inference':
        logging.info("===== Starting inference.")
        crf_inference(input_args)
    else:
        print("Run mode {} not found.".format(input_args.mode))

    exit(0)
