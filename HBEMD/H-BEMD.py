import cv2
import numpy as np
import pylab as plt
import math
from skimage.color import rgb2hsv
import imageio
from HBEMD.image_condition_adjustment import image_condition_adjustment
from HBEMD.BEMD import BEMD
class HBEMD():
    def __init__(self, img_path):
        self.rgb_img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)
        self.rgb_adjusted = image_condition_adjustment(self.rgb_img, r_added_value=0, g_added_value=0, b_added_value=0)
    def hue_value(self):
        hsvInputImage_Full = cv2.cvtColor(self.rgb_adjusted.astype(np.float32), cv2.COLOR_RGB2HSV_FULL)
        hue_full, sat_full, val_full = cv2.split(hsvInputImage_Full)
        hue_16bit = np.array(hue_full, dtype=np.uint16)
        return hue_16bit

    def change_negative_hue_to_positive(hue_channel):
        for j in range(0, hue_channel.shape[0]):
            for k in range(0, hue_channel.shape[1]):
                if (hue_channel[j][k] < 0):
                    hue_channel[j][k] = hue_channel[j][k] + 360
        return hue_channel

    def sifting(self, hue_img, max_imf_value):
        hueradians = np.deg2rad(hue_img)
        cos_hueradians = np.cos(hueradians)
        sin_hueradians = np.sin(hueradians)

        sin_bemd = BEMD()
        imfs_sin_hue = sin_bemd.bemd(sin_hueradians, max_imf=max_imf_value)
        cos_bemd = BEMD()
        imfs_cos_hue = cos_bemd.bemd(cos_hueradians, max_imf=max_imf_value)

        h_bemd_imfs = []
        imfs_no = min(imfs_sin_hue.shape[0], imfs_cos_hue.shape[0])
        for i in range(0, imfs_no):
            imf_cos_hue = imfs_cos_hue[i]
            imf_sin_hue = imfs_sin_hue[i]
            imf_arctan_hue = np.arctan2(imf_sin_hue, imf_cos_hue)
            imf_hue_degree = np.rad2deg(imf_arctan_hue)
            imf_hue_degree_old = np.copy(imf_hue_degree)
            imf_hue_degree = self.change_negative_hue_to_positive(imf_hue_degree_old)
            h_bemd_imfs.append(imf_hue_degree)
        return h_bemd_imfs