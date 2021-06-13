"""
-------------------------------------------------------------------------------
Created: 26.03.2021, 10:46
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Tests the cherefSkinDetector implementation by visual inspection.
-------------------------------------------------------------------------------
"""
# %%
from ippgtoolbox.detection import CherefSkinDetector
import matplotlib.pyplot as plt
import cv2

IMG_PATH = './assets/tony-stark-header.jpg'


class TestCherefSkinDetector:

    def __init__(self):
        self.skinDet = CherefSkinDetector(input_type='uint8', rgb2bgr=True)
        self.img = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(self.img)
        plt.title('Original img')

    def test_extract_skin_mask(self):
        self.skin_mask = self.skinDet.extract_skin_mask(self.img)
        plt.figure()
        plt.imshow(self.skin_mask)
        plt.title('Skin mask')

    def test_extract_skin(self):
        self.skin = self.skinDet.extract_skin(self.img)
        plt.figure()
        plt.imshow(self.skin)
        plt.title('Skin')


if __name__ == '__main__':

    testCherefSkinDetector = TestCherefSkinDetector()
    testCherefSkinDetector.test_extract_skin_mask()
    testCherefSkinDetector.test_extract_skin()

# %%
