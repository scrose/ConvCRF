"""
The MIT License (MIT)

Copyright (c) 2017 Marvin Teichmann
"""

import torch
import cv2
import os
import sys
import numpy as np
import logging
from params import params

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)


if __name__ == '__main__':
    logging.info("Hello World.")


# -----------------------------------
# Loads image data into array
# Read image and reverse channel order
# Loads image as 8 bit (regardless of original depth)
# -----------------------------------
def get_image(image_path, img_ch=3):
    assert img_ch == 3 or img_ch == 1, 'Invalid input channel number.'
    assert os.path.exists(image_path), 'Image path {} does not exist.'.format(image_path)
    image = None
    if img_ch == 1:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    elif img_ch == 3:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# load test output data from model
def load_output(output_path):
    # Load output data
    if not os.path.exists(output_path):
        print('Output file {} not found.'.format(output_path))
    else:
        return torch.load(output_path, map_location=lambda storage, loc: storage)


# load test output data from model
# Input format: []
def output_img(img_data, n_classes, w, h, output_path):

    # Colourize and resize mask to full size
    mask_img = colourize(np.argmax(img_data, axis=1), n_classes, palette=params.palette_alt)
    mask_img = cv2.resize(mask_img[0].astype('float32'), (w, h), interpolation=cv2.INTER_NEAREST)

    # Save output mask image to file (RGB -> BGR conversion)
    # Note that the default color format in OpenCV is often
    # referred to as RGB but it is actually BGR (the bytes are reversed).
    cv2.imwrite(output_path, cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR))

    print('Output mask image saved to {}.'.format(output_path))


# -----------------------------------
# Scales image to n x tile dimensions with stride
#  and crops to match input image aspect ratio
# -----------------------------------
def adjust_to_tile(img, patch_size, stride, img_ch, scale_factor=None, interpolate=cv2.INTER_NEAREST):

    # Get full-sized dimensions
    w = img.shape[1]
    h = img.shape[0]

    assert patch_size == 2*stride, "Tile size must be 2x stride."

    # Get width scaling factor for tiling
    if scale_factor:
        scale = (scale_factor * patch_size)/w
    else:
        scale = (int(w / patch_size) * patch_size) / w

    dim = (int(w * scale), int(h * scale))

    # resize image
    img_resized = cv2.resize(img, dim, interpolation=interpolate)
    h_resized = img_resized.shape[0]
    h_tgt = int(h_resized / patch_size) * patch_size

    # crop top of image to match aspect ratio
    img_cropped = None
    h_crop = h_resized - h_tgt
    if img_ch == 1:
        img_cropped = img_resized[h_crop:h_resized, :]
    elif img_ch == 3:
        img_cropped = img_resized[h_crop:h_resized, :, :]

    return img_cropped, img_cropped.shape[1], img_cropped.shape[0], h_crop


# -----------------------------------
# Merge segmentation classes
# -----------------------------------
def merge_classes(data_tensor, merged_classes):

    data = data_tensor.numpy()

    # merge classes
    for i, cat_grp in enumerate(merged_classes):
        data[np.isin(data, cat_grp)] = i

    return torch.tensor(data)


# -----------------------------------
# Merge segmentation classes
# Input:
#   - Numpy array of shape [NCHW]
# -----------------------------------
def isolate_class(data_tensor, isolated_class):

    # isolate classes
    isolated_class_data = np.expand_dims(data_tensor[:, isolated_class, :, :], axis=1)
    # combine merged classes
    merged_class_data = np.concatenate(
        (data_tensor[:, :isolated_class, :, :], data_tensor[:, isolated_class+1:, :, :]), axis=1)
    merged_class_data = np.expand_dims(np.sum(merged_class_data, axis=1), axis=1)
    merged_class_data = np.concatenate((isolated_class_data, merged_class_data), axis=1)

    return merged_class_data


# -----------------------------------
# Merge segmentation classes
# -----------------------------------
def merge_class(data_tensor, palette=params.palette_alt):

    data = data_tensor.numpy()
    # merge classes
    for i, cat_grp in enumerate(palette):
        data[np.isin(data, cat_grp)] = i

    return torch.tensor(data)


# -----------------------------------
# Convert RBG mask array to class-index encoded values
# Input format:
#  - [NCWH] with RGB-value encoding, where C = RGB (3)
#  - Palette parameters in form [CC'], where C = number of classes, C' = 3 (RGB)
# Output format:
#  - [NCWH] with one-hot encoded classes, where C = number of classes
# -----------------------------------
def class_encode(input_data, palette):

    # Ensure image is RBG format
    assert input_data.shape[1] == 3
    input_data = input_data.to(torch.float32).mean(dim=1)
    palette = torch.from_numpy(palette).to(torch.float32).mean(dim=1)

    # map mask colours to segmentation classes
    for idx, c in enumerate(palette):
        class_bool = input_data == c
        input_data[class_bool] = idx

    return input_data.to(torch.uint8)


# -----------------------------------
# Colourize one-hot encoded image by palette
# Input format: NCWH (one-hot class encoded)
# -----------------------------------
def colourize(img_data, n_classes, palette=None):

    n = img_data.shape[0]
    w = img_data.shape[1]
    h = img_data.shape[2]

    # collapse one-hot encoding to single channel
    # make 3-channel (RGB) image
    img_data = np.moveaxis(np.stack((img_data,) * 3, axis=1), 1, -1).reshape(n * w * h, 3)

    # map categories to palette colours
    for i in range(n_classes):
        class_bool = img_data == np.array([i, i, i])
        class_idx = np.all(class_bool, axis=1)
        img_data[class_idx] = palette[i]

    return img_data.reshape(n, w, h, 3)


# -----------------------------------
# Collate mask tiles of format [NCHW]
# Combines prediction mask tiles into full-sized mask
# -----------------------------------
def reconstruct(tiles, w, h, w_full, h_full, offset, n_classes, stride):

    # Calculate reconstruction dimensions
    patch_size = params.patch_size
    n_strides_in_row = w // stride - 1
    n_strides_in_col = h // stride - 1

    # Calculate overlap
    olap_size = patch_size - stride

    # initialize full image numpy array
    mask_fullsized = np.empty((n_classes, h + offset, w), dtype=np.float32)

    # Create empty rows
    r_olap_prev = None
    r_olap_merged = None

    # row index (set to offset height)
    row_idx = offset

    for i in range(n_strides_in_col):
        # Get initial tile in row
        t_current = tiles[i*n_strides_in_row]
        r_current = np.empty((n_classes, patch_size, w), dtype=np.float32)
        col_idx = 0
        # Step 1: Collate column tiles in row
        for j in range(n_strides_in_row):
            t_current_width = t_current.shape[2]
            if j < n_strides_in_row - 1:
                # Get adjacent tile
                t_next = tiles[i * n_strides_in_row + j + 1]
                # Extract right overlap of current tile
                olap_current = t_current[:, :, t_current_width - olap_size:t_current_width]
                # Extract left overlap of next (adjacent) tile
                olap_next = t_next[:, :, 0:olap_size]
                # Average the overlapping segment logits
                olap_current = torch.nn.functional.softmax(torch.tensor(olap_current), dim=0)
                olap_next = torch.nn.functional.softmax(torch.tensor(olap_next), dim=0)
                olap_merged = (olap_current + olap_next) / 2
                # Insert averaged overlap into current tile
                np.copyto(t_current[:, :, t_current_width - olap_size:t_current_width], olap_merged)
                # Insert updated current tile into row
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)
                col_idx += t_current_width
                # Crop next tile and copy to current tile
                t_current = t_next[:, :, olap_size:t_next.shape[2]]

            else:
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)

        # Step 2: Collate row slices into full mask
        r_current_height = r_current.shape[1]
        # Extract overlaps at top and bottom of current row
        r_olap_top = r_current[:, 0:olap_size, :]
        r_olap_bottom = r_current[:, r_current_height - olap_size:r_current_height, :]

        # Average the overlapping segment logits
        if i > 0:
            # Average the overlapping segment logits
            r_olap_top = torch.nn.functional.softmax(torch.tensor(r_olap_top), dim=0)
            r_olap_prev = torch.nn.functional.softmax(torch.tensor(r_olap_prev), dim=0)
            r_olap_merged = ( r_olap_top + r_olap_prev ) / 2

        # Top row: crop by bottom overlap (to be averaged)
        if i == 0:
            # Crop current row by bottom overlap size
            r_current = r_current[:, 0:r_current_height - olap_size, :]
        # Otherwise: Merge top overlap with previous
        else:
            # Replace top overlap with averaged overlap in current row
            np.copyto(r_current[:, 0:olap_size, :], r_olap_merged)

        # Crop middle rows by bottom overlap
        if 0 < i < n_strides_in_col - 1:
            r_current = r_current[:, 0:r_current_height - olap_size, :]

        # Copy current row to full mask
        np.copyto(mask_fullsized[:, row_idx:row_idx + r_current.shape[1], :], r_current)
        row_idx += r_current.shape[1]
        r_olap_prev = r_olap_bottom

    # Colourize and resize mask to full size
    mask_fullsized = np.expand_dims(mask_fullsized, axis=0)
    _mask_pred = colourize(np.argmax(mask_fullsized, axis=1), n_classes, palette=params.palette_alt)
    mask_resized = cv2.resize(_mask_pred[0].astype('float32'), (w_full, h_full), interpolation=cv2.INTER_AREA)

    return mask_resized


# -----------------------------------
# Collate mask tiles of format [NCHW]
# Combines prediction mask tiles into full-sized mask
# -----------------------------------
def restore(tiles, w, h, n_classes, stride):

    # Calculate reconstruction dimensions
    patch_size = params.patch_size
    n_strides_in_row = w // stride - 1
    n_strides_in_col = h // stride - 1

    # Calculate overlap
    olap_size = patch_size - stride

    # initialize full image numpy array
    mask_fullsized = np.empty((n_classes, h, w), dtype=np.float32)

    # Create empty rows
    r_olap_prev = None
    r_olap_merged = None

    # row index (set to offset height)
    row_idx = 0

    for i in range(n_strides_in_col):
        # Get initial tile in row
        t_current = tiles[i*n_strides_in_row]
        r_current = np.empty((n_classes, patch_size, w), dtype=np.float32)
        col_idx = 0
        # Step 1: Collate column tiles in row
        for j in range(n_strides_in_row):
            t_current_width = t_current.shape[2]
            if j < n_strides_in_row - 1:
                # Get adjacent tile
                t_next = tiles[i * n_strides_in_row + j + 1]
                # Extract right overlap of current tile
                olap_current = t_current[:, :, t_current_width - olap_size:t_current_width]
                # Extract left overlap of next (adjacent) tile
                olap_next = t_next[:, :, 0:olap_size]
                # Average the overlapping segment logits
                olap_current = torch.nn.functional.softmax(torch.tensor(olap_current), dim=0)
                olap_next = torch.nn.functional.softmax(torch.tensor(olap_next), dim=0)
                olap_merged = (olap_current + olap_next) / 2
                # Insert averaged overlap into current tile
                np.copyto(t_current[:, :, t_current_width - olap_size:t_current_width], olap_merged)
                # Insert updated current tile into row
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)
                col_idx += t_current_width
                # Crop next tile and copy to current tile
                t_current = t_next[:, :, olap_size:t_next.shape[2]]

            else:
                np.copyto(r_current[:, :, col_idx:col_idx + t_current_width], t_current)

        # Step 2: Collate row slices into full mask
        r_current_height = r_current.shape[1]
        # Extract overlaps at top and bottom of current row
        r_olap_top = r_current[:, 0:olap_size, :]
        r_olap_bottom = r_current[:, r_current_height - olap_size:r_current_height, :]

        # Average the overlapping segment logits
        if i > 0:
            # Average the overlapping segment logits
            r_olap_top = torch.nn.functional.softmax(torch.tensor(r_olap_top), dim=0)
            r_olap_prev = torch.nn.functional.softmax(torch.tensor(r_olap_prev), dim=0)
            r_olap_merged = ( r_olap_top + r_olap_prev ) / 2

        # Top row: crop by bottom overlap (to be averaged)
        if i == 0:
            # Crop current row by bottom overlap size
            r_current = r_current[:, 0:r_current_height - olap_size, :]
        # Otherwise: Merge top overlap with previous
        else:
            # Replace top overlap with averaged overlap in current row
            np.copyto(r_current[:, 0:olap_size, :], r_olap_merged)

        # Crop middle rows by bottom overlap
        if 0 < i < n_strides_in_col - 1:
            r_current = r_current[:, 0:r_current_height - olap_size, :]

        # Copy current row to full mask
        np.copyto(mask_fullsized[:, row_idx:row_idx + r_current.shape[1], :], r_current)
        row_idx += r_current.shape[1]
        r_olap_prev = r_olap_bottom

    return mask_fullsized
    #return mask_fullsized[:, offset:, :]


def dice_loss(y_pred, y_true):
    """Computes the Sørensen–Dice loss.
    Note that PyTorch optimizers minimize a loss. In this
    case, we would like to maximize the dice loss so we
    return the negated dice loss.
    Args:
        y_true: a tensor of shape [H, W, C].
        y_pred: a tensor of shape [H, W, C]. Corresponds to
            softmax of the classes
        eps: added to the denominator for numerical stability.
    Returns:
        dice_loss: the Sørensen–Dice loss.
    """
    print(y_pred.shape, y_true.shape)
    y_pred = torch.tensor(y_pred)
    y_true = torch.tensor(y_true)
    intersection = torch.sum(y_pred * y_true, dim=(0, 1))
    cardinality = torch.sum(y_pred + y_true, dim=(0, 1))
    dloss = (2. * intersection + params.dice_smooth) / (cardinality + params.dice_smooth)

    return 1. - torch.mean(dloss)
