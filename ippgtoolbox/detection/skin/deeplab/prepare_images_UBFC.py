# %%
import os
import cv2
from pathlib import Path
from tools import CaffeNetFaceDetector
import sys
from scipy.io import savemat
import pickle
from utils import UBFCVidReader

# %%
# define necessary paths
# base_dir = '/app/shared/data'
print('Definition and creation of directory paths...')
base_dir = '/media/fast_storage/matthieu_scherpf/tmp'
src_dir = os.path.join(base_dir, 'orig/original')
dst_dir = os.path.join(base_dir, 'orig/resized')
loc_dir = os.path.join(base_dir, 'orig/face_detection_info')

# Path(dst_dir).mkdir(parents=False, exist_ok=True)
# Path(loc_dir).mkdir(parents=False, exist_ok=True)

# %%
# read image by image, apply face detection, resize image for processing with deeplab
print('Starting processing of original images: face detection and resizing to required deepLab format...')
resize_shape = (512, 512)
det_inst = CaffeNetFaceDetector.get_instance(
    input_type='uint8', zoom_factor=1, face_loc_resize=(1.2, 1.2))
vid_obj = UBFCVidReader(os.path.join(src_dir, 'vid.avi'))
face_locs = []
for i, frame in enumerate(vid_obj.reader_next()):
    face_roi, face_loc = det_inst.extract_face_from_rgbImage(
        frame, return_loc=True)
    face_locs.append(face_loc)
    face_roi = cv2.resize(face_roi, resize_shape, cv2.INTER_LINEAR)
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
    # NOTE use *.png format when writing images to file with open cv (!!!this way they are saved lossless!!!)
    cv2.imwrite(os.path.join(
        dst_dir, '{}.png'.format(str(i).zfill(5))), face_roi)

# save necessary information to matfile
data_to_save = {
    'face_locs': face_locs,
    'resize_shape': resize_shape,
}
savemat(os.path.join(loc_dir, 'face_detection_info.mat'), data_to_save)
# with open(os.path.join(loc_dir, 'face_detection_info.p'), 'wb') as f:
#     pickle.dump(data_to_save, f)

# %%
