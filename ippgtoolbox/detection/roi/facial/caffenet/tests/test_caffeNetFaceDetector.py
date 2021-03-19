"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 12:09
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Tests the CaffeNetFaceDetector class based on a visual analysis.
-------------------------------------------------------------------------------
"""
# %%
from detection import CaffeNetFaceDetector
from detection import helpers
import cv2
import matplotlib.pyplot as plt

IMG_PATH = './assets/tony-stark-header.jpg'


class TestCaffeNetFaceDetector:

    def __init__(self):
        self.img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.imshow(self.img)
        kwargs = {
            'conf_th': 0.5,
            'bitdepth': 8,
            'input_type': 'uint8',
            'zoom_factor': 1,
            'face_loc_resize': (1., 1.),
            'rgb2bgr': True,
        }
        self.det = CaffeNetFaceDetector().get_instance(**kwargs)

    def test_extract_face_locations(self):
        self.face_loc = self.det.extract_face_locations(self.img)
        print(self.face_loc)

    def test_extract_face(self):
        self.face = self.det.extract_face(self.img)
        plt.figure()
        plt.imshow(self.face)

    def test_extract_face_mask(self):
        self.face_mask = self.det.extract_face_mask(self.img)
        plt.figure()
        plt.imshow(self.face_mask)

    def test_extract_face_with_given_location(self):
        face = self.det.extract_face_with_given_location(
            self.img, self.face_loc)
        plt.figure()
        plt.imshow(face)

    def test_clipped_zoom(self):
        zoomed_img = helpers.clipped_zoom(self.img, 2.)
        plt.figure()
        plt.imshow(zoomed_img)
        zoomed_img = helpers.clipped_zoom(self.img, 0.5)
        plt.figure()
        plt.imshow(zoomed_img)

    def test_resize_face_loc_borders(self):
        face_loc = helpers.resize_face_loc_borders(self.face_loc, (1.5, 1.5))
        face = self.det.extract_face_with_given_location(self.img, face_loc)
        plt.figure()
        plt.imshow(face)


if __name__ == '__main__':
    testCaffeNetFaceDetector = TestCaffeNetFaceDetector()
    testCaffeNetFaceDetector.test_extract_face_locations()
    testCaffeNetFaceDetector.test_extract_face()
    testCaffeNetFaceDetector.test_extract_face_mask()
    testCaffeNetFaceDetector.test_extract_face_with_given_location()
    testCaffeNetFaceDetector.test_clipped_zoom()
    testCaffeNetFaceDetector.test_resize_face_loc_borders()

# %%
