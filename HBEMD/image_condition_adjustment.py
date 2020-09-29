import cv2
import numpy as np
import pylab as plt


def image_condition_adjustment(rgbImage, r_added_value=50, g_added_value=50, b_added_value=50):
    rgbImage[:, :, 0] += r_added_value
    rgbImage[:, :, 1] += g_added_value
    rgbImage[:, :, 2] += b_added_value

    rgbImage[rgbImage > 255] = 255
    rgbOutputImage = rgbImage.astype(np.uint8)
    return rgbOutputImage
