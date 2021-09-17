"""
-------------------------------------------------------------------------------
Created: 21.02.2021, 11:55
-------------------------------------------------------------------------------
Author: Matthieu Scherpf
Email: Matthieu.Scherpf@tu-dresden.de
Website: https://becuriouss.github.io/matthieu-scherpf/
Project page: tba
-------------------------------------------------------------------------------
Purpose: Some helper functions for the 
-------------------------------------------------------------------------------
"""
import numpy as np
import cv2


def clipped_zoom(frame, zoom_factor):
    """
    Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions
    Args:
        frame : Image array
        zoom_factor : amount of zoom as a ratio (0 to Inf)
    """
    """Center zoom in/out of the given image and returning an enlarged/shrinked view of 
    the image without changing dimensions

    Parameters
    ----------
    frame : numpy.ndarray of shape (img_height, img_width, 3)
        the image on which to apply zoom
    zoom_factor : float
        the zoom factor to apply where zoom_factor > 1 enlarges the image and zoom_factor < 1 shrinks the image

    Returns
    -------
    numpy.ndarray of shape (zoomed_height, zoomed_width, 3)
        the zoomed input image
    """
    if zoom_factor == 1:
        return frame
    height, width = frame.shape[:2]  # It's also the final desired shape
    new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)

    # Crop only the part that will remain in the result (more efficient)
    # Centered bbox of the final desired size in resized (larger/smaller) image coordinates
    y1, x1 = max(0, new_height - height) // 2, max(0, new_width - width) // 2
    y2, x2 = y1 + height, x1 + width
    bbox = np.array([y1, x1, y2, x2])
    # Map back to original image coordinates
    bbox = (bbox / zoom_factor).astype(np.int)
    y1, x1, y2, x2 = bbox
    cropped_frame = frame[y1:y2, x1:x2]

    # Handle padding when downscaling
    resize_height, resize_width = min(
        new_height, height), min(new_width, width)
    pad_height1, pad_width1 = (
        height - resize_height) // 2, (width - resize_width) // 2
    pad_height2, pad_width2 = (height - resize_height) - \
        pad_height1, (width - resize_width) - pad_width1
    pad_spec = [(pad_height1, pad_height2), (pad_width1,
                                             pad_width2)] + [(0, 0)] * (frame.ndim - 2)

    result = cv2.resize(cropped_frame, (resize_width, resize_height))
    result = np.pad(result, pad_spec, mode='constant')
    assert result.shape[0] == height and result.shape[1] == width
    return result


def resize_face_loc_borders(face_loc, face_loc_resize):
    """Resize a bounding box.

    Parameters
    ----------
    face_loc : numpy.ndarray of shape (4,)
        the bounding box to resize
    face_loc_resize : list like object with two floats, e.g. (1.2, 1.2)
        first element represents the resize factor for the height; second element represents the resize factor for the width

    Returns
    -------
    numpy.ndarray of shape (4,)
        the resized face locations specified by the bounding box
    """
    new_face_loc = face_loc.copy()
    row_scale = int((face_loc[2] - face_loc[0]) * (face_loc_resize[0]-1) / 2)
    col_scale = int((face_loc[1] - face_loc[3]) * (face_loc_resize[1]-1) / 2)
    new_face_loc[0] = face_loc[0] - row_scale
    new_face_loc[2] = face_loc[2] + row_scale
    new_face_loc[3] = face_loc[3] - col_scale
    new_face_loc[1] = face_loc[1] + col_scale
    # check if a starting value (x0 or y0) is below zero
    for i in range(4):
        if new_face_loc[i] < 0:
            new_face_loc[i] = 0
    return new_face_loc
