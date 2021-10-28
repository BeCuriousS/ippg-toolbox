# %%
import os
import cv2
from pathlib import Path
from tools import CaffeNetFaceDetector
import sys
from scipy.io import savemat
import pickle

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
    input_type='uint8', zoom_factor=0.8, face_loc_resize=(1.6, 1.6))
img_names = sorted(os.listdir(src_dir))
face_locs = []
for i, name in enumerate(img_names):
    if '.jpg' in name:
        img = cv2.cvtColor(cv2.imread(
            os.path.join(src_dir, name)), cv2.COLOR_BGR2RGB)
        face_roi, face_loc = det_inst.extract_face_from_rgbImage(
            img, return_loc=True)
        face_locs.append(face_loc)
        face_roi = cv2.resize(face_roi, resize_shape, cv2.INTER_LINEAR)
        face_roi = cv2.cvtColor(face_roi, cv2.COLOR_RGB2BGR)
        # NOTE use *.png format when writing images to file with open cv (!!!this way they are saved lossless!!!)
        cv2.imwrite(os.path.join(
            dst_dir, name.replace('.jpg', '.png')), face_roi)

# save necessary information to matfile
data_to_save = {
    'face_locs': face_locs,
    'resize_shape': resize_shape,
}
savemat(os.path.join(loc_dir, 'face_detection_info.mat'), data_to_save)
# with open(os.path.join(loc_dir, 'face_detection_info.p'), 'wb') as f:
#     pickle.dump(data_to_save, f)

# %%
