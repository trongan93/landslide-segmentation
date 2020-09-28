#!/usr/bin/python
# coding: UTF-8
#
# Implement and optimization for satellite image
# Author: Trong-An Bui (trongan93@gmail.com - http://buitrongan.com)
# Reference from: https://github.com/laszukdawid/PyEMD/blob/master/PyEMD/BEMD.py (Original BEMD algorithm)
#
# Feel free to contact for any information.

from __future__ import division, print_function

import logging
import numpy as np

from scipy.interpolate import Rbf
import pylab as plt

try:
    from skimage.morphology import reconstruction
except ImportError:
    pass

class BEMD:
    logger = logging.getLogger(__name__)

    def __init__(self):
        # ProtoIMF related
        self.mse_thr = 0.01
        self.mean_thr = 0.01

        self.FIXE = 1  # Single iteration by default, otherwise results are terrible
        self.FIXE_H = 0
        self.MAX_ITERATION = 5

    def __call__(self, image, max_imf=-1):
        return self.bemd(image, max_imf=max_imf)

    def extract_max_min_spline(self, image, min_peaks_pos, max_peaks_pos):
        """Calculates top and bottom envelopes for image.

        Parameters
        ----------
        image : numpy 2D array

        Returns
        -------
        min_env : numpy 2D array
            Bottom envelope in form of an image.
        max_env : numpy 2D array
            Top envelope in form of an image.
        """
        xi, yi = np.meshgrid(np.arange(image.shape[0]), np.arange(image.shape[1]))
        min_val = np.array([image[x,y] for x, y in zip(*min_peaks_pos)])
        max_val = np.array([image[x,y] for x, y in zip(*max_peaks_pos)])
        min_env = self.spline_points(min_peaks_pos[0], min_peaks_pos[1], min_val, xi, yi)
        max_env = self.spline_points(max_peaks_pos[0], max_peaks_pos[1], max_val, xi, yi)
        return min_env, max_env

    @classmethod
    def spline_points(cls, X, Y, Z, xi, yi):
        """Creates a spline for given set of points.

        Uses Radial-basis function to extrapolate surfaces. It's not the best but gives something.
        Griddata algorithm didn't work.
        """
        spline = Rbf(X, Y, Z, function='cubic')
        return spline(xi, yi)

    @classmethod
    def find_extrema_positions(cls, image):
        """
        Finds extrema, both mininma and maxima, based on morphological reconstruction.
        Returns extrema where the first and second elements are x and y positions, respectively.

        Parameters
        ----------
        image : numpy 2D array
            Monochromatic image or any 2D array.

        Returns
        -------
        min_peaks_pos : numpy array
            Minima positions.
        max_peaks_pos : numpy array
            Maxima positions.
        """
        max_peaks_pos = BEMD.extract_maxima_positions(image)
        min_peaks_pos = BEMD.extract_minima_positions(image)
        return min_peaks_pos, max_peaks_pos

    @classmethod
    def extract_minima_positions(cls, image):
        return BEMD.extract_maxima_positions(-image)

    @classmethod
    def extract_maxima_positions(cls, image):
        # seed_min = image - 1
        seed_min = image - 0.000001 # case on sin and cos of image
        dilated = reconstruction(seed_min, image, method='dilation')
        cleaned_image = image - dilated
        # imageWithMaxValue = np.where(cleaned_image>0)
        maxima_positions = np.where(cleaned_image>0)[::-1]
        return maxima_positions

    @classmethod
    def end_condition(cls, image, IMFs):
        """Determins whether decomposition should be stopped.

        Parameters
        ----------
        image : numpy 2D array
            Input image which is decomposed.
        IMFs : numpy 3D array
            Array for which first dimensions relates to respective IMF,
            i.e. (numIMFs, imageX, imageY).
        """
        rec = np.sum(IMFs, axis=0)

        # If reconstruction is perfect, no need for more tests
        if np.allclose(image, rec):
            return True

        return False

    def check_proto_imf(self, proto_imf, proto_imf_prev, mean_env):
        """Check whether passed (proto) IMF is actual IMF.
        Current condition is solely based on checking whether the mean is below threshold.

        Parameters
        ----------
        proto_imf : numpy 2D array
            Current iteration of proto IMF.
        proto_imf_prev : numpy 2D array
            Previous iteration of proto IMF.
        mean_env : numpy 2D array
            Local mean computed from top and bottom envelopes.

        Returns
        -------
        boolean
            Whether current proto IMF is actual IMF.
        """
        #TODO: Sifiting is very sensitive and subtracting const val can often flip
        #      maxima with minima in decompoisition and thus repeating above/below
        #      behaviour. For now, mean_env is checked whether close to zero excluding
        #      its offset.
        if np.all(np.abs(mean_env-mean_env.mean())<self.mean_thr):
        #if np.all(np.abs(mean_env)<self.mean_thr):
            return True

        # If very little change with sifting
        if np.allclose(proto_imf, proto_imf_prev, rtol=0.01):
            return True

        # If IMF mean close to zero (below threshold)
        if np.mean(np.abs(proto_imf)) < self.mean_thr:
            return True

        # Everything relatively close to 0
        mse_proto_imf = np.mean(proto_imf*proto_imf)
        if mse_proto_imf > self.mse_thr:
            return False

        return False

    def bemd(self, image, max_imf=-1):
        """Performs bidimensional EMD (BEMD) on grey-scale image with specified parameters.

        Parameters
        ----------
        image : numpy 2D array,
            Grey-scale image.
        max_imf : int, (default: -1)
            IMF number to which decomposition should be performed.
            Negative value means *all*.

        Returns
        -------
        IMFs : numpy 3D array
            Set of IMFs in form of numpy array where the first dimension
            relates to IMF's ordinary number.
        """
        image_s = image.copy()

        imf = np.zeros(image.shape)
        imf_old = imf.copy()

        imfNo = 0
        IMF = np.empty((imfNo,)+image.shape)
        notFinished = True

        while(notFinished):
            self.logger.debug('IMF -- '+str(imfNo))
            # self.logger.debug('Shape IMF: ',IMF.shape)
            # self.logger.debug('IMF: ',IMF[:imfNo])
            res = image_s - np.sum(IMF[:imfNo], axis=0)
            saveLogFile('residue_' + str(imfNo) + '.csv',res)
            imf = res.copy()
            mean_env = np.zeros(image.shape)
            stop_sifting = False

            # Counters
            n = 0   # All iterations for current imf.
            n_h = 0 # counts when mean(proto_imf) < threshold

            while(not stop_sifting and n<self.MAX_ITERATION):
                n += 1
                self.logger.debug("Iteration: %i", n)

                min_peaks_pos, max_peaks_pos = self.find_extrema_positions(imf)
                self.logger.debug("min_peaks_pos = %i  |  max_peaks_pos = %i", len(min_peaks_pos[0]), len(max_peaks_pos[0]))
                if len(min_peaks_pos[0])>1 and len(max_peaks_pos[0])>1:
                    min_env, max_env = self.extract_max_min_spline(imf, min_peaks_pos, max_peaks_pos)
                    mean_env = 0.5*(min_env+max_env)

                    imf_old = imf.copy()
                    imf = imf - mean_env

                    # Fix number of iterations
                    if self.FIXE:
                        if n>=self.FIXE+1:
                            stop_sifting = True

                    # Fix number of iterations after number of zero-crossings
                    # and extrema differ at most by one.
                    elif self.FIXE_H:
                        if n == 1: continue
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            n_h += 1
                        else:
                            n_h = 0

                        # STOP if enough n_h
                        if n_h >= self.FIXE_H:
                            stop_sifting = True

                    # Stops after default stopping criteria are met
                    else:
                        if self.check_proto_imf(imf, imf_old, mean_env):
                            stop_sifting = True

                else:
                    stop_sifting = True

            IMF = np.vstack((IMF, imf.copy()[None,:]))
            imfNo += 1

            if self.end_condition(image, IMF) or (max_imf>0 and imfNo>=max_imf):
                notFinished = False
                break

        res = image_s - np.sum(IMF[:imfNo], axis=0)
        if not np.allclose(res, 0):
            IMF = np.vstack((IMF, res[None,:]))
            imfNo += 1

        return IMF

def saveLogFile(fileName, values):
    np.savetxt("/home/trongan93/Projects/log/landslide-segmentation" + fileName, values, delimiter=',', fmt='%s')

if __name__ == "__main__":
    print("Running example on BEMD")
    PLOT = True

    logging.basicConfig(level=logging.DEBUG)

    # Generate image
    print("Generating image... ", end="")
    rows, cols = 128, 128
    row_scale, col_scale = 128, 128
    x = np.arange(rows)/float(row_scale)
    y = np.arange(cols).reshape((-1,1))/float(col_scale)

    pi2 = 2*np.pi
    img = np.zeros((rows,cols))
    # img = img + np.sin(2*pi2*x)*np.cos(y*4*pi2+4*x*pi2)
    img = img + 3*np.sin(2*pi2*x)+2
    # img = img + 5*x*y + 2*(y-0.2)*y
    print("Done")

    # import cv2
    # img = cv2.imread('/home/trongan93/Projects/NCKUCubeSAT/CubeSATAI/data/pre-processing/brain2.jpg')
    # img = cv2.resize(img,(128,128))
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # img = (255-img)

    plt.title("Generated image")
    plt.imshow(img,cmap='Greys')
    plt.colorbar()
    plt.show()

    # Perform decomposition
    print("Performing decomposition... ", end="")
    bemd = BEMD()
    #HBEMD.FIXE_H = 5
    IMFs = bemd.bemd(img)
    imfNo = IMFs.shape[0]
    print("Done")

    if PLOT:
        print("Plotting results... ", end="")
        import pylab as plt

        # Save image for preview
        plt.figure(figsize=(4,4*(imfNo+1)))
        plt.subplot(imfNo+1, 1, 1)
        plt.imshow(img,cmap='Greys')
        plt.colorbar()
        plt.title("Input image")

        # Save reconstruction
        for n, imf in enumerate(IMFs):
            plt.subplot(imfNo+1, 1, n+2)
            plt.imshow(imf,cmap='Greys')
            plt.colorbar()
            plt.title("IMF %i"%(n+1))

        plt.savefig("image_decomp")
        print("Done")