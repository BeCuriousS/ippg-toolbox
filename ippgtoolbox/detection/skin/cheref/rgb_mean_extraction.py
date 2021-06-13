# ----------------------------------------------------------------------------------------------------------
#
# CREATION DATE: 29.12.2020, 15:43
#
# AUTHOR: Matthieu Scherpf
#
#
# PURPOSE: extract mean red, green and blue values from recordings of the BP4D+ and UBFC dataset using
# precalculated skin masks
# ----------------------------------------------------------------------------------------------------------
# FIXME revise this file
# %%

# ----------------------------------------------------------
# imports
# ----------------------------------------------------------
import cv2
import os
from scipy.io import loadmat, savemat
import numpy as np
import progressbar
from tools import CaffeNetFaceDetector
from utils import UBFCVidReader, BP4DVidReader
from pathlib import Path

# %%

# ----------------------------------------------------------
# definitions
# ----------------------------------------------------------


def extract_mean_rgb_predefined_skin_masks(vid_obj,
                                           mask_path,
                                           face_det_info,
                                           threshold=0.01,
                                           verbose=False):

    vid_obj.readMetaData()

    mask_names = sorted(os.listdir(mask_path))
    mean_rgb = []
    frames_orig = []
    masked_imgs = []

    for i, (frame, mn, face_loc) in enumerate(zip(vid_obj.reader_next(), mask_names, face_det_info['face_locs'])):

        frame = CaffeNetFaceDetector.extract_face_from_rgbImage_with_given_location(
            frame, face_loc)
        frame = cv2.resize(frame, tuple(
            *face_det_info['resize_shape']), cv2.INTER_LINEAR)
        mask = cv2.imread(os.path.join(mask_path, mn), cv2.IMREAD_UNCHANGED)

        if verbose:
            frames_orig.append(frame.copy())

        h, w, _ = frame.shape

        # compute skin mask from deeplab segmentation outputs
        skin_mask = np.ones((*mask.shape, 1), dtype=np.bool)
        skin_mask[mask > 255*threshold] = False
        number_of_skin_pixels = np.sum(skin_mask)
        skin_mask = np.tile(skin_mask, (1, 1, 3))

        # apply skin mask to frame
        frame[np.logical_not(skin_mask)] = 0

        if verbose:
            masked_imgs.append(frame)

        # compute mean
        mean_rgb.append(np.sum(frame, axis=(0, 1))/number_of_skin_pixels)

    if verbose:
        return np.asarray(mean_rgb), frames_orig, masked_imgs
    return mean_rgb


def process_record(vid_obj,
                   full_msk_path,
                   full_dst_path,
                   record_name):
    # necessary data
    face_det_info = loadmat(os.path.join(
        full_msk_path, 'face_detection_info.mat'))
    # compute mean red, green and blue value for each frame
    mean_rgb, frames_orig, masked_imgs = extract_mean_rgb_predefined_skin_masks(
        vid_obj, full_msk_path, face_det_info, verbose=True)
    # save the result
    data_to_save = {
        'mean_rgb': mean_rgb,
    }
    Path(full_dst_path).mkdir(parents=True, exist_ok=True)
    savemat(os.path.join(full_dst_path, 'mean_rgb.mat'), data_to_save)

# %%

# ----------------------------------------------------------
# ----------------------------------------------------------
# BP4D+ dataset
# ----------------------------------------------------------
# ----------------------------------------------------------


# ----------------------------------------------------------
# paths
# ----------------------------------------------------------
src_path = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/measurements'
msk_path = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/sensors_2021_ms/ROI_deepLab'
dst_path = '/media/fast_storage/matthieu_scherpf/2019_06_26_BP4D+_v0.2/processing/sensors_2021_ms/MEAN_RGB_deepLab'

# ----------------------------------------------------------
# loop over records
# ----------------------------------------------------------
record_names = [rn for rn in os.listdir(src_path) if 'F' in rn or 'M' in rn]

bar = progressbar.ProgressBar(
    max_value=len(record_names)-1,
    variables={'subj_name': ''},
    suffix=' >>> Processing record {variables.subj_name:12s}',
    term_width=120)

for i, rn in enumerate(record_names):
    bar.update(i, subj_name=rn)
    # define full absolute paths
    full_src_path = os.path.join(src_path, rn, 'video')
    full_msk_path = os.path.join(msk_path, rn)
    full_dst_path = os.path.join(dst_path, rn)
    # create video object
    vid_obj = BP4DVidReader(full_src_path)
    process_record(vid_obj, full_msk_path, full_dst_path, rn)

# %%

# ----------------------------------------------------------
# ----------------------------------------------------------
# UBFC dataset
# ----------------------------------------------------------
# ----------------------------------------------------------

# ----------------------------------------------------------
# paths
# ----------------------------------------------------------
src_path = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/measurements'
msk_path = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/processing/sensors_2021_ms/ROI_deepLab'
dst_path = '/media/fast_storage/matthieu_scherpf/2018_12_UBFC_Dataset/processing/sensors_2021_ms/MEAN_RGB_deepLab'

# ----------------------------------------------------------
# loop over records
# ----------------------------------------------------------
record_names = [rn for rn in os.listdir(src_path) if 'subject' in rn]

bar = progressbar.ProgressBar(
    max_value=len(record_names)-1,
    variables={'subj_name': ''},
    suffix=' >>> Processing record {variables.subj_name:12s}',
    term_width=120)

for i, rn in enumerate(record_names):
    bar.update(i, subj_name=rn)
    # define full absolute paths
    full_src_path = os.path.join(src_path, rn, 'vid.avi')
    full_msk_path = os.path.join(msk_path, rn)
    full_dst_path = os.path.join(dst_path, rn)
    # create video object
    vid_obj = UBFCVidReader(full_src_path)
    process_record(vid_obj, full_msk_path, full_dst_path, rn)
