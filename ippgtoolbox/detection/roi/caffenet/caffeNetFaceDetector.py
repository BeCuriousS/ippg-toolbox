"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 10:50
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieuscherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Face detection based on a Deep neural network based on caffe.
-------------------------------------------------------------------------------
"""
import cv2
import os
import numpy as np
import pkg_resources
from ..helpers import *


class CaffeNetFaceDetector:
    """Standard face detection residual neural network based on caffe. The following repository was used for this implementation:

    https://github.com/sr6033/face-detection-with-OpenCV-and-DNN

    An average runtime of about 30ms per frame was observed on the CPU. Also the results were quite robust in the experiments.
    """

    MODEL_PATH = pkg_resources.resource_filename(
        'ippgtoolbox', 'detection/roi/caffenet/rsc')
    INSTANCE = None

    def __init__(self,
                 conf_th=0.5,
                 input_type='float',
                 zoom_factor=1,
                 face_loc_resize=(1., 1.),
                 rgb2bgr=True):
        """Initializes a face detector based on a residual neural network. Use the get_instance method if you want to work with a singleton.

        Parameters
        ----------
        conf_th : float, optional
            the confidence threshold to use for the detected bounded boxes, by default 0.5
        input_type : {'float', 'uint8'}, optional
            the data type of the image values. If 'float', then the values must lie in range [0, 1]. If 'uint8' the values are assumed to lie in range [0, 255] , by default 'float'
        zoom_factor : int, optional
            the zoom factor to apply on the input image. This can mitigate the problem of false detection, if the a face is quite near to the image border. The value should lie between 0.5 and 1. to take effect, by default 1
        face_loc_resize : tuple, optional
            the resizing of the bounding box found by the network. You can enlarge or shrink it if needed, by default (1., 1.)
        rgb2bgr : bool, optional
            defines if the color channels of the input image are sorted in r-g-b or b-g-r and therefore if a conversion is necessary as the networks expects the b-g-r format, by default True
        """
        self.net = cv2.dnn.readNetFromCaffe(
            os.path.join(self.MODEL_PATH, 'deploy.prototxt.txt'),
            os.path.join(self.MODEL_PATH,
                         'res10_300x300_ssd_iter_140000.caffemodel')
        )
        self.conf_th = conf_th
        self.input_type = input_type
        self.zoom_factor = zoom_factor
        self.face_loc_resize = face_loc_resize
        self.convert_rgb2bgr = rgb2bgr

    def extract_face_locations(self, frame, verbose=False):
        """Extracts the bounding box from some input image most likely to contain a face.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image from which a face should be detected
        verbose : bool, optional
            if true then all detected bounded boxes are returned as well, by default False

        Returns
        -------
        numpy.ndarray of shape (4,)
            locations of the detected face as specified in the opencv dnn format. If no face could be found the original image size is returned as loc.
        """
        frame = frame.copy()
        (h, w) = frame.shape[0:2]
        loc = np.array([0, frame.shape[1], frame.shape[0], 0])

        frame = clipped_zoom(frame, self.zoom_factor)

        frame = self._convert_img(frame)

        frame = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(frame)
        det = self.net.forward()

        max_conf_idx = np.argmax(det[0, 0, :, 2])
        max_conf = det[0, 0, max_conf_idx, 2]
        if max_conf > self.conf_th:
            loc = det[0, 0, max_conf_idx, 3:7] * np.array([w, h, w, h])
            # orientation from face_recognition package: top, right, bottom, left = face_location
            # orientation from opencv dnn: (startX, startY, endX, endY) = box.astype("int")
            # resort to make it same as for face_recognition package
            loc = np.array([loc[1], loc[2], loc[3], loc[0]])
            # check if there is any negative value in the locations
            loc[loc < 0] = 0

        if self.face_loc_resize != (1., 1.):
            loc = resize_face_loc_borders(loc, self.face_loc_resize)

        if verbose:
            return loc.astype(int), det
        return loc.astype(int)

    def extract_face(self, frame, verbose=False):
        """Extracts the face of an image by applying the bounding box on the input frame.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image to extract the face from            
        verbose : bool, optional
            if true then the applied bounded box is returned as well, by default False

        Returns
        -------
        numpy.ndarray of shape (face_height, face_width, 3)
            the extracted face
        """
        face_loc = self.extract_face_locations(frame)
        frame_roi = frame[face_loc[0]:face_loc[2],
                          face_loc[3]:face_loc[1], :].copy()

        if verbose:
            return frame_roi, face_loc
        return frame_roi

    def extract_face_mask(self, frame, verbose=False):
        """Extracts the binary face mask of an image.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image to extract the face from
        verbose : bool, optional
            if true then the applied bounded box is returned as well, by default False

        Returns
        -------
        numpy.ndarray of shape (face_height, face_width, 3)
            the binary mask for the detected face
        """
        face_loc = self.extract_face_locations(frame)
        # face_mask = np.zeros((*frame.shape[0:2]))
        face_mask = np.zeros(frame.shape)
        face_mask[face_loc[0]:face_loc[2],
                  face_loc[3]:face_loc[1]] = True

        if verbose:
            return face_mask, face_loc
        return face_mask

    def _convert_img(self, frame):
        """Convert the frame regarding data type and color channels.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image to extract the face from

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
        CaffeNetFaceDetector
            an instance of the caffe net face detector
        """
        if CaffeNetFaceDetector.INSTANCE == None:
            CaffeNetFaceDetector.INSTANCE = CaffeNetFaceDetector(**kwargs)
        return CaffeNetFaceDetector.INSTANCE

    @staticmethod
    def extract_face_with_given_location(frame, face_loc):
        """Method that can be applied when a given face location already exists.

        Parameters
        ----------
        frame : numpy.ndarray of shape (img_height, img_width, 3)
            the image to extract the face from
        face_loc : numpy.ndarray of shape (4, )
            the face location to apply for the face extraction

        Returns
        -------
        numpy.ndarray of shape (face_height, face_width, 3)
            the face extracted from the input image
        """
        frame_roi = frame[face_loc[0]:face_loc[2],
                          face_loc[3]:face_loc[1], :].copy()
        return frame_roi
