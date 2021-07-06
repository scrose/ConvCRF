import json
import os
import random
import sys
import numpy as np
import torch


class Parameters:
    """
    - General parameters
    - Model parameters
    - Land Cover Categories (LCC-A, LCC-B)
    - Data Augmentation parameters
    - Network hyperparameters
    - Utility functions
    """

    def __init__(self):
        # ===================================
        # General Parameters
        # ===================================

        # device settings
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Enumerated modes
        self.TRAIN = 'train'
        self.VALID = 'valid'
        self.TEST = 'test'
        self.PREPROCESS = 'preprocess'
        self.META = 'metadata'
        self.AUGMENT = 'augment'
        self.EXTRACT = 'extract'
        self.AUGMENT = 'augment'
        self.PROFILE = 'profile'
        self.RATES = 'rates'
        self.TUNE = 'tune'
        self.NORMAL = 'normal'
        self.SUMMARY = 'summary'
        self.MERGE = 'merge'
        self.OVERFIT = 'overfit'
        self.COMBINED = 'combined'

        # list of available datasets
        self.dsets = ['jean', 'fortin']

        # general data paths
        self.src_db = None
        self.tgt_db = None
        self.files = None

        # ===================================
        # Model Parameters
        # ===================================

        # size of the output feature map
        self.output_size = 324

        # size of the tiles to extract and save in the database, must be >= to input size
        self.input_size = 512
        self.patch_size = 512

        # patch stride: smaller than input_size for overlapping tiles
        self.stride_size = 512

        # number of pixels to pad *after* resize to image with by mirroring (edge's of
        # patches tend not to be analyzed well, so padding allows them to appear more centered
        # in the patch)
        self.pad_size = (self.input_size - self.output_size) // 2

        # Calculate crop sizes
        self.crop_left = self.pad_size
        self.crop_right = self.pad_size + self.output_size
        self.crop_up = self.pad_size
        self.crop_down = self.pad_size + self.output_size

        # what percentage of the dataset should be used as a held out validation/testing set
        self.buf_size = 1000
        self.partition = 0.10
        self.clip = 1.
        self.clip_overfit = 0.003

        # Get random seed so that we can reproducibly do the cross validation setup
        self.seed = random.randrange(sys.maxsize)
        random.seed(self.seed)  # set the seed
        # print(f"random seed (note down for reproducibility): {seed}")

        # ===================================
        # Land Cover Categories (LCC-A, LCC-B, LCC-merged)
        # ===================================

        # -----------------------------------
        # DST-A Land Cover Categories (LCC-A)
        # -----------------------------------
        # 1. '#000000' [0,0,0] (black - solid): Not categorized
        # 2. '#ffa500' [255, 165, 0] (orange) Broadleaf/Mixedwood forest
        # 3. '#228b22' [34, 139, 34] (dark green - approx): Coniferous forest
        # 4. '#7cfc00' [124, 252, 0] (light green - approx): Herbaceous/Shrub
        # 5. '#873434' [139, 69, 19] (sanguine brown - approx): Sand/gravel/rock
        # 6. '#5f9ea0' [95, 158, 160] (turquoise) Wetland
        # 7. '#0000ff' [0,0,255] (blue - solid): Water
        # 8. '#2dbdff' [45, 189, 255] (light blue - approx): Snow/Ice
        # 9. '#ff0004' [255, 0, 4] (red - solid): Regenerating area

        self.mask_categories_alt = {
            '#000000': 'Not categorized',
            '#ffa500': 'Broadleaf/Mixedwood',
            '#228b22': 'Coniferous',
            '#7cfc00': 'Herbaceous/Shrub',
            '#8b4513': 'Sand/Gravel/Rock',
            '#5f9ea0': 'Wetland',
            '#0000ff': 'Water',
            '#2dbdff': 'Snow/Ice',
            '#ff0004': 'Regenerating Area',
        }

        self.category_labels_alt = [
            'Not categorized',
            'Broadleaf/Mixedwood',
            'Coniferous',
            'Herbaceous/Shrub',
            'Sand/Gravel/Rock',
            'Wetland',
            'Water',
            'Snow/Ice',
            'Regenerating Area',
        ]

        self.palette_alt = np.array(
            [[0, 0, 0],
             [255, 165, 0],
             [34, 139, 34],
             [124, 252, 0],
             [139, 69, 19],
             [95, 158, 160],
             [0, 0, 255],
             [45, 189, 255],
             [255, 0, 4],
             ])

        # merged classes
        self.categories_merged_alt = [
            np.array([0]),
            np.array([1, 2]),
            np.array([3]),
            np.array([4, 5]),
            np.array([6]),
            np.array([7]),
            np.array([8]),
            np.array([9]),
            np.array([10]),
        ]

        # ------------------------------------
        # DST-B Land Cover Categories (LCC-B)
        # ------------------------------------
        # 1. '#000000' [0,0,0] (black - solid): Not categorized
        # 2. '#ffaa00' [] Broadleaf forest
        # 3. '#d5d500' [] Mixedwood forest
        # 4. '#005500' [0,85,0] (camarone - approx): Coniferous forest
        # 5. '#41dc66' [65,220,102] (emerald - approx): Shrub
        # 6. '#ffff7f' [255,255,127] (dolly - approx): Herbaceous
        # 7. '#873434' [135,52,52] (sanguine brown - approx): Rock
        # 8. '#aaaaff' [] Wetland
        # 9. '#0000ff' [0,0,255] (blue - solid): Water
        # 10. '#b0fffd' [176,255,253] (French pass - approx): Snow/Ice
        # 11. '#ff00ff' [255,0,255] (magenta - solid): Regenerating area

        self.mask_categories = {
            '#000000': 'Not categorized',
            '#ffaa00': 'Broadleaf forest',
            '#d5d500': 'Mixedwood forest',
            '#005500': 'Coniferous forest',
            '#41dc66': 'Shrub',
            '#ffff7f': 'Herbaceous',
            '#873434': 'Rock',
            '#aaaaff': 'Wetland',
            '#0000ff': 'Water',
            '#b0fffd': 'Snow/Ice',
            '#ff00ff': 'Regenerating Area',
        }

        self.category_labels = [
            'Not categorized',
            'Broadleaf forest',
            'Mixedwood forest',
            'Coniferous forest',
            'Shrub',
            'Herbaceous',
            'Rock',
            'Wetland',
            'Water',
            'Snow/Ice',
            'Regenerating Area'
        ]

        self.palette = np.array(
            [[0, 0, 0],
             [255, 170, 0],
             [213, 213, 0],
             [0, 85, 0],
             [65, 220, 102],
             [255, 255, 127],
             [135, 52, 52],
             [170, 170, 255],
             [0, 0, 255],
             [176, 255, 253],
             [255, 0, 255],
             ])

        # ------------------------------------
        # Merged Land Cover Categories (LCC-Merged)
        # ------------------------------------

        # merged classes
        self.categories_merged = [
            np.array([0]),
            np.array([1, 2, 3, 4, 5]),
            np.array([6, 7, 8, 9]),
            np.array([10])
        ]

        # Merged classes
        self.palette_merged = np.array(
            [[0, 0, 0],
             [65, 220, 102],
             [135, 52, 52],
             [255, 0, 255],
             ])

        self.mask_categories_merged = {
            '#000000': 'Not categorized',
            '#41dc66': 'Vegetation',
            '#873434': 'Non-Vegetation',
            '#ff00ff': 'Regenerating Area',
        }

        self.category_labels_merged = [
            'Not categorized',
            'Vegetation',
            'Non-Vegetation',
            'Regenerating Area']

        # Merged classes
        self.palette_merged = np.array(
            [[0, 0, 0],
             [65, 220, 102],
             [135, 52, 52],
             [255, 0, 255],
             ])

        # ===================================
        # Data Augmentation Parameters
        # ===================================

        self.aug_n_samples_max = 4000
        self.min_sample_rate = 0
        self.max_sample_rate = 5
        self.sample_rate_coef = np.arange(1, 21, 1)
        self.sample_threshold = np.arange(0, 3., 0.1)

        # Affine coefficient (elastic deformation)
        self.alpha = 0.19

        # ===================================
        # Network Hyperparameters
        # ===================================

        self.dropout = 0.5
        self.lr_min = 1e-6
        self.lr_max = 0.1
        self.gamma = 0.9
        self.l2_reg = 1e-4
        self.in_channels = 3
        self.momentum = 0.9
        self.dice_weight = 0.5
        self.ce_weight = 0.5
        self.dice_smooth = 1.
        self.weight_decay = 5e-5
        self.grad_steps = 16
        self.test_intv = 70

        # ===================================
        # CRF Parameters
        # ===================================
        self.sxy_g = (3, 3)
        self.sxy_b = (80, 80)
        self.srgb = (13, 13, 13)
        self.schan = (0.1, 0.1, 0.1)
        self.crf_iter = 5

        # The compat argument can be any of the following:

        # Scalar: PottsCompatibility is being used (Implements potts \mu(a,b) = -w[a==b]).
        # 1D array: DiagonalCompatibility is being used (Implements diagonal \mu(a,b) = -[a==b]v(a)).
        # 2D array: MatrixCompatibility is being used (Implements matrix \mu(a,b)).
        #
        # These are label-compatibilites µ(xi, xj) whose parameters could possibly
        # be learned. For example, they could indicate that mistaking bird pixels
        # for sky is not as bad as mistaking cat for sky. The arrays should have
        # nlabels or (nlabels,nlabels) as shape and a float32 datatype.

        self.compat_g = np.array([8, 8, 8, 1, 1, 1, 1, 1, 1]).astype(np.float32)
        # self.compat_g = 8
        self.compat_b = 100*np.array([
            0.080374,
            0.708806,
            0.121845,
            0.259942,
            0.222811,
            0.416504,
            1.000000,
            0.346472,
            0.283262]).astype(np.float32)
        self.compat_b = 10

# -----------------------------------
# Create parameters instance
# -----------------------------------
params: Parameters = Parameters()
