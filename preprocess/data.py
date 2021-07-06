import sys
from utils import utils
import torch
import numpy as np
import cv2
import logging
from params import params

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


class Dataset:
    """
    Dataset wrapper
    """

    def __init__(self, args):
        self.crf = None
        self.shape = None
        self.image_data = None

        # global parameters
        self.in_channels = args.in_channels
        self.n_classes = args.n_classes
        self.isolate = args.isolate
        self.patch_size = params.patch_size  # Set stride to half tile size
        self.stride = self.patch_size // 2
        self.offset = None

        # Load image data and parameters
        # -------------------------------------------------------
        self.img = utils.get_image(args.img_path, args.in_channels)
        self.w = self.img.shape[1]
        self.h = self.img.shape[0]

        # Log initialization
        logging.info(
            'Test Image loaded: {}\n\tWidth: {}px\n\tHeight: {}px\n\tChannels: {}'.format(
                args.img_path, self.w, self.h, self.in_channels))

        # Resize image
        # Resample image (resample_factor x patch_size)
        self.scale_factor = args.resample

        self.img_resized, self.w_resized, self.h_resized, self.offset = \
            utils.adjust_to_tile(self.img, self.patch_size, self.stride, self.in_channels, self.scale_factor)

        # Compute tiling parameters
        self.shape, n_tiles, n_ch = self.tile(self.img_resized)

        # Log resizing
        logging.info('Image resized: \n\tChannels: {}\n\tWidth: {}px\n\tHeight: {}px'.format(
            n_ch, self.w_resized, self.h_resized))

        # Convert single channel (grayscale) image to 3-channel (RGB) image
        if n_ch == 1:
            logging.info('Converting single channel image to 3-channel.')
            self.img_resized = np.stack((self.img_resized,) * 3, axis=2)

        # reformat and convert image data to uint8 [WHC] -> [HWC]
        self.img_resized = self.img_resized.astype(np.uint8)

        # Log tiling
        logging.info(
            'Image tiled: \n\tNumber: {}\n\tSize: {}px by {}px\n\tOffset: {}\n\tChannels: {}\n\tStride: {}'.format(
                n_tiles, self.shape[0], self.shape[1], self.offset, n_ch, self.stride))

        # Load and normalize ground-truth mask to format: [NWHC]
        # -------------------------------------------------------
        if args.mask_path:
            logging.info('Mask Image: {}'.format(args.mask_path))
            mask_data = utils.get_image(args.mask_path, 3)
            mask, w_mask, h_mask, offset_mask = utils.adjust_to_tile(
                mask_data, self.patch_size, self.stride, 3, self.scale_factor, interpolate=cv2.INTER_NEAREST)

            # Reformat mask data
            mask = torch.tensor(np.moveaxis(np.expand_dims(mask, axis=0), 3, 1))

            # Convert palettes, if needed
            if args.merge:
                print('\tMerging LCC-B palette to LCC-A palette ... ', end='')
                mask = utils.class_encode(mask, params.palette).long()
                mask = utils.merge_classes(mask, params.categories_merged_alt)
            else:
                mask = utils.class_encode(mask, params.palette_alt).long()

            # convert ground-truth to one-hot encoding
            if self.isolate:
                n_isolated_classes = 2
                mask = torch.nn.functional.one_hot(mask, num_classes=n_isolated_classes).permute(0, 3, 1, 2)
                print('\tIsolating mask class: {}'.format(self.isolate))
                mask = torch.tensor(utils.isolate_class(mask, self.isolate))
            else:
                mask = torch.nn.functional.one_hot(mask, num_classes=self.n_classes).permute(0, 3, 1, 2)

            # Log mask resize
            logging.info(
                'Mask Resized: \n\tWidth: {}px\n\tHeight: {}px\n\tOffset: {}\n\tClasses: {}\n\tIsolated: {}'.format(
                    w_mask, h_mask, offset_mask, self.n_classes, self.isolate))

            # Reformat ground-truth data: [1CHW] -> [HWC]
            self.mask = np.moveaxis(mask.numpy().squeeze(0), 0, -1)

        # Load and normalize unary data to format: [NWHC]
        # -------------------------------------------------------
        self.unary = None
        if args.unary_path:
            unary = utils.load_output(args.unary_path)
            unary = np.concatenate(unary['results'])
            unary = torch.nn.functional.softmax(torch.tensor(unary), dim=1)
            self.unary = unary.numpy()

            # Isolate class in unary data
            if self.isolate:
                print('\tIsolating unary class: {}'.format(self.isolate))
                self.unary = utils.isolate_class(self.unary, self.isolate)

            # restore unary to full-size [NCWH -> NHW]
            self.unary = utils.restore(self.unary, self.w_resized, self.h_resized, self.n_classes, self.stride)

            print(self.unary.shape)

            # Log unary data [NHW]
            logging.info(
                'Unary data: {}\n\tClasses: {}\n\tWidth: {}px\n\tHeight: {}px'.format(
                    args.unary_path, self.unary.shape[0], self.unary.shape[2], self.unary.shape[1]))

    def tile(self, img_data):
        """
        Tile image
        """
        # Convert image to tensor
        img_data = torch.as_tensor(img_data, dtype=torch.float32)
        # Tile image data
        img_data = img_data.unfold(0, self.patch_size, self.stride).unfold(1, self.patch_size, self.stride)
        img_data = torch.reshape(img_data,
                                 (img_data.shape[0] * img_data.shape[1], self.in_channels, self.patch_size,
                                  self.patch_size))
        img_data = img_data.permute(0, 2, 3, 1)

        # Return shape, n_tiles, n_ch
        return img_data.shape[1:3], img_data.shape[0], img_data.shape[3]

    def loss(self, pred):
        # Calculate mIoU with ground-truth
        return utils.dice_loss(pred, self.mask)
