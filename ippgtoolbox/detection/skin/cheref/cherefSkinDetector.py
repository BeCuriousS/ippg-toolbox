"""
-------------------------------------------------------------------------------
Created: 26.03.2021, 09:46
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Implementation of the threshold based skin detection introduced in:

Dahmani, Djamila; Cheref, Mehdi; Larabi, Slimane (2020): Zero-sum game theory model for segmenting skin regions. In: Image and Vision Computing 99, S. 103925. DOI: 10.1016/j.imavis.2020.103925.

NOTE cite the above publication if you use parts of this code
-------------------------------------------------------------------------------
"""
import cv2
import numpy as np


class CherefSkinDetector:
    """Skin detection based on:

    Dahmani, Djamila; Cheref, Mehdi; Larabi, Slimane (2020): Zero-sum game theory model for segmenting skin regions. In: Image and Vision Computing 99, S. 103925. DOI: 10.1016/j.imavis.2020.103925.

    which is implemented according to:

    https://github.com/CHEREF-Mehdi/SkinDetection
    """

    # ----------------------------------------------------------
    # defined thresholds
    # ----------------------------------------------------------
    HSV_th_lower = (0, 15, 0)
    HSV_th_upper = (17, 170, 255)

    YCrCb_th_lower = (0, 135, 85)
    YCrCb_th_upper = (255, 180, 135)

    def __init__(self, input_type='float', rgb2bgr=True):
        self.input_type = input_type
        self.convert_rgb2bgr = rgb2bgr

    def extract_skin_mask(self, frame):
        """Generate the skin mask for a given frame.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image to extract the skin from

        Returns
        -------
        numpy.ndarray of shape (img_height, img_width) with entries of type bool
            the boolean mask
        """

        frame = self._convert_img(frame)

        # converting from gbr to hsv color space
        frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # skin color range for hsv color space
        HSV_mask = cv2.inRange(
            frame_HSV, self.HSV_th_lower, self.HSV_th_upper)
        HSV_mask = cv2.morphologyEx(
            HSV_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # converting from gbr to YCbCr color space
        frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        # skin color range for hsv color space
        YCrCb_mask = cv2.inRange(
            frame_YCrCb, self.YCrCb_th_lower, self.YCrCb_th_upper)
        YCrCb_mask = cv2.morphologyEx(
            YCrCb_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        # merge skin detection (YCbCr and hsv)
        global_mask = cv2.bitwise_and(YCrCb_mask, HSV_mask)
        global_mask = cv2.medianBlur(global_mask, 3)
        global_mask = cv2.morphologyEx(
            global_mask, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8))

        return global_mask > 0

    def extract_skin(self, frame, mask=None):
        """Extract the skin from a given image using a given mask.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image to extract the skin from
        mask : numpy.ndarray of shape (img_height, img_width) with entries of type bool, optional
            the boolean mask from a previous extraction, by default None

        Returns
        -------
        numpy.ndarray of shape (img_height, img_width, 3)
            the frame where the pixel presumably not containing skin are (0, 0, 0)
        """
        if mask is None:
            mask = self.extract_skin_mask(frame)
        masked_frame = cv2.bitwise_and(
            frame, frame, mask=mask.astype(np.uint8))
        return masked_frame

    def _convert_img(self, frame):
        """Convert the frame regarding data type and color channels.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image to extract the skin from

        Raises
        ------
        NotImplementedError
            if the defined data type is not implemented
        """

        if self.input_type == 'float':
            frame = (frame*(2**8-1)).astype(np.uint8)
        elif self.input_type == 'uint8':
            pass
        else:
            raise NotImplementedError(
                'This input_type <<{}>> is not implemented.'.format(
                    self.input_type))

        if self.convert_rgb2bgr:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        return frame

    @staticmethod
    def get_instance(**kwargs):
        """Get the instance of an object of this class if existing. This is a convenience method to avoid multiple initializations of the same face detector if desired.

        Returns
        -------
        CherefSkinDetector
            an instance of the caffe net face detector
        """
        if CherefSkinDetector.INSTANCE == None:
            CherefSkinDetector.INSTANCE = CherefSkinDetector(**kwargs)
        return CherefSkinDetector.INSTANCE
