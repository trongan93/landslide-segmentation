import cv2
import numpy as np
import pylab as plt
import math
from skimage.color import rgb2hsv
from BEMD import BEMD
import imageio
from image_condition_adjustment import imageConditionAdjustment

def main():
    # bgrInputImage = cv2.imread("/mnt/d/ProjectData/CubeSAT/SourceCodeData/pre-processing/LaPintada_Mexico2013Cause_rainfall.jpg")
    # img_path = "/mnt/d/ProjectData/CubeSAT/SourceCodeData/pre-processing/sunkoshi_oli_2014261.jpg"
    # img_path = "/home/trongan93/Desktop/25_94_Large_cropped_6.TIF"
    # img_path = "/mnt/d/ProjectData/CubeSAT/test-data/LC08_L1TP_141041_20140918_20170419_01_T1/cropped/27.770733_85.868467_VERY_LARGE_cropped.TIF"
    # img_path = "/mnt/d/ProjectData/CubeSAT/test-data/LC08_L1TP_141041_20130915_20170502_01_T1/cropped/27.770733_85.868467_VERY_LARGE_cropped.TIF"
    # img_path = "/mnt/d/ProjectData/CubeSAT/test-data/44_-123_Large_cropped_4.TIF"
    img_path = "/mnt/d/ProjectData/CubeSAT/test-data/27.770733_85.868467_CNN_FIX_cropped_before.TIF"
    # img_path = "/mnt/d/ProjectData/CubeSAT/test-data/27.770733_85.868467_CNN_FIX_cropped_after.TIF"
    bgrInputImage = cv2.imread(img_path)
    plt.imshow(cv2.cvtColor(bgrInputImage,cv2.COLOR_BGR2RGB), origin='lower', cmap='Greys')
    plt.show()
    
    
    bgrInputImage = cv2.resize(bgrInputImage,(224,224))
    rgb_resized_img = cv2.cvtColor(bgrInputImage,cv2.COLOR_BGR2RGB)
    plt.imshow(rgb_resized_img, origin='lower', cmap='Greys')
    plt.show()

    # ### Only for testing 0
    # plt.title("Original image with resized")
    # plt.imshow(cv2.cvtColor(bgrInputImage,cv2.COLOR_BGR2RGB),cmap='nipy_spectral',origin='lower')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.colorbar()
    # plt.show()

    # plt.title("B channel in RGB")
    # plt.imshow(rgb_resized_img[:,:,2], interpolation='none', cmap='Greys', origin='lower')
    # plt.show()
    # # 3D view in Greys-RGB channel
    # xx_greys, yy_greys = np.mgrid[0:rgb_resized_img[:,:,2].shape[0], 0:rgb_resized_img[:,:,2].shape[1]]
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('value')
    # ax.plot_surface(xx_greys, yy_greys, rgb_resized_img[:,:,2], rstride=1, cstride=1, cmap='Greys', linewidth=0)
    # plt.show()

    ### Only for testing 0


    # Testing for Image Adjustment
    rgb_adjusted = imageConditionAdjustment(rgb_resized_img, r_added_value = 0, g_added_value= 0, b_added_value= 0)
    # plt.imshow(rgb_adjusted, origin='lower', cmap='Greys')
    # plt.show()
    # End of Testing Adjustment

    # hsvInputImage_Full = cv2.cvtColor(bgrInputImage.astype(np.float32), cv2.COLOR_BGR2HSV_FULL)
    hsvInputImage_Full = cv2.cvtColor(rgb_adjusted.astype(np.float32), cv2.COLOR_RGB2HSV_FULL)
    hue_full, sat_full, val_full = cv2.split(hsvInputImage_Full)
    hue_16bit = np.array(hue_full,dtype=np.uint16)
    # OutputRaw_Y8(hue_16bit,'hue_input_16bit')
    #saveLogFile("huechannel.csv",hue_channel) #log Hue channel

    # Change read and convert image without opencv
    # from skimage import io
    # rgb_img = io.imread(img_path)
    # rgb_img_16 = np.array(rgb_img,dtype=np.uint16)
    # plt.imshow(rgb_img)
    # plt.colorbar()
    # plt.show()
    # hsv_img = rgb2hsv(rgb_img_16)
    # plt.imshow(hsv_img,vmin=0, vmax=1)
    # plt.colorbar()
    # plt.show()
    # hue_img = hsv_img[:, :, 0]
    # plt.imshow(hue_img,vmin=0, vmax=1, cmap='Greys', origin='lower')
    # plt.colorbar()
    # plt.show()

    hueradians = np.deg2rad(hue_16bit)
    cos_hueradians = np.cos(hueradians)
    sin_hueradians = np.sin(hueradians)
    
    ### Only for testing 1
    plt.title("Hue channel")
    plt.imshow(hue_16bit,cmap='hsv',origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
    # 3D view in Hue channel
    ###Ref: https://stackoverflow.com/questions/31805560/how-to-create-surface-plot-from-greyscale-image-with-matplotlib
    xx_hue, yy_hue = np.mgrid[0:hue_16bit.shape[0], 0:hue_16bit.shape[1]]
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('value')
    ax.plot_surface(xx_hue, yy_hue, hue_16bit, rstride=1, cstride=1, cmap='hsv', linewidth=0)
    # plt.title("Hue channel in 3D")
    plt.show()
    ### End of Only for testing 1

    # test arctan2 of sin and cos
    # hue_arctan2 = np.arctan2(sin_hueradians,cos_hueradians)
    # plt.title("Arctan2 of sin and cos of hue")
    # tmpHue = np.asarray(changeNegativeHueToPositive(np.rad2deg(hue_arctan2)),dtype='int')
    # plt.imshow(tmpHue,cmap='hsv')
    # plt.colorbar()
    # plt.show()

    # 
    # '''Todo BEMD on sin and cos of hue
    # 
    # ### Only for testing 2
    # plt.title("Cos value of hue")
    # plt.imshow(np.array(cos_hueradians,dtype='float32'), interpolation='none', cmap='Greys', origin='lower')
    # plt.show()
    # # 3D view in Hue channel
    # xx_cos_hue, yy_cos_hue = np.mgrid[0:cos_hueradians.shape[0], 0:cos_hueradians.shape[1]]
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('value')
    # ax.plot_surface(xx_cos_hue, yy_cos_hue, cos_hueradians, rstride=1, cstride=1, cmap='Greys', linewidth=0)
    # plt.show()

    # plt.title("Sin value of hue")
    # plt.imshow(np.array(sin_hueradians,dtype='float32'), interpolation='none', cmap='Greys', origin='lower')
    # plt.show()
    # # 3D view in Hue channel
    # xx_sin_hue, yy_sin_hue = np.mgrid[0:sin_hueradians.shape[0], 0:sin_hueradians.shape[1]]
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('value')
    # ax.plot_surface(xx_sin_hue, yy_sin_hue, sin_hueradians, rstride=1, cstride=1, cmap='Greys', linewidth=0)
    # plt.show()
    # ### End of Only for testing 2

    print("BEMD in sin value of hue channel")
    bemd2 = BEMD()
    imfs_sin_hue = bemd2.bemd(sin_hueradians, max_imf=2)

    print("BEMD in cos value of hue channel")
    bemd = BEMD()
    imfs_cos_hue = bemd.bemd(cos_hueradians, max_imf=2)
    
    
    
    landslide_feature_img = np.empty_like(hue_16bit)

    imfs_no = min(imfs_sin_hue.shape[0], imfs_cos_hue.shape[0])

    for i in range(0,imfs_no):
        imf_cos_hue = imfs_cos_hue[i]
        imf_sin_hue = imfs_sin_hue[i]
        # imf_hue = imfs_hue[i]

        #saveLogFile("sin_hue_imf_" + str(i) + "_.csv", imf_sin_hue) #save imf i of sin(hue)
        #saveLogFile("cos_hue_imf_" + str(i) + "_.csv", imf_cos_hue) #save imf i of cos(hue)
        
        # plt.title('IMF ' + str(i) + ' of hue')
        # plt.imshow(imf_hue,cmap='hsv')
        # plt.show()
        
        # plt.title('IMF ' + str(i) + ' cos of hue')
        # plt.imshow(imf_cos_hue,cmap='hsv')
        # plt.show()

        # plt.title('IMF ' + str(i) + ' sin of hue')
        # plt.imshow(imf_sin_hue,cmap='hsv')
        # plt.show()

        # imf_arctan_hue = np.arctan2(imf_cos_hue,imf_sin_hue)*180/np.pi
        imf_arctan_hue = np.arctan2(imf_sin_hue,imf_cos_hue)
        imf_hue_degree = np.rad2deg(imf_arctan_hue)
        imf_hue_degree_old = np.copy(imf_hue_degree)
        imf_hue_degree = changeNegativeHueToPositive(imf_hue_degree_old)
        #saveLogFile("arctan2_hue_imf_" + str(i) + "_radian.csv", imf_arctan_hue) #save imf i of arctan2(sin and cos) - radian value
        #saveLogFile("arctan2_hue_imf_" + str(i) + "_degree.csv", imf_hue_degree) #save imf i of arctan2(sin and cos) - degree value
        plt.title('IMF ' + str(i+1) + ' arctan of hue')
        plt.imshow(imf_hue_degree,cmap='hsv', origin='lower')
        plt.colorbar()
        plt.show()
        landslide_feature_img = imf_hue_degree

    # show final BEMD
    plt.title("Landslide feature")
    plt.imshow(landslide_feature_img, cmap='hsv', origin='lower')
    plt.colorbar()
    plt.show()

    # # Apply Threshold 1
    # ret,landslide_reduced_top = cv2.threshold(landslide_feature_img,310,360,cv2.THRESH_BINARY)
    # ret,landslide_reduced_bottom = cv2.threshold(landslide_feature_img,101,360,cv2.THRESH_BINARY)
    # landslide_reduced_bottom = 360 - landslide_reduced_bottom
    # # plt.imshow(landslide_reduced_top, cmap='Greys', origin='lower')
    # # plt.colorbar()
    # # plt.show()
    # # plt.imshow(landslide_reduced_bottom, cmap='Greys', origin='lower')
    # # plt.colorbar()
    # # plt.show()
    # landslide_reduced = landslide_reduced_top + landslide_reduced_bottom
    # plt.imshow(landslide_reduced, cmap='Greys', origin='lower')
    # plt.colorbar()
    # plt.show()

    # Apply Threshold 2
    landslide_feature_range_1 = cv2.inRange(landslide_feature_img,0,90)
    landslide_feature_range_2 = cv2.inRange(landslide_feature_img,330,360) 
    plt.imshow(landslide_feature_range_1+landslide_feature_range_2, cmap='Greys', origin='lower')
    plt.show()

    imageio.imsave('/mnt/d/ProjectData/CubeSAT/test-data/27.770733_85.868467_CNN_FIX_cropped_before_thredhold_applied.TIF',landslide_feature_range_1+landslide_feature_range_2)

    

    # # 
    # # '''Todo BEMD on complex number of hue
    # # 
    # hue_j = multiple_j(hueradians)
    # print("Done multiple j")
    # hue_cplx = np.exp(hue_j*np.pi)
    # print("Done exp complex hue")
    # HBEMD = BEMD()
    # imfs_hue_cplx_real = HBEMD.HBEMD(hue_cplx.real,max_imf = 4)
    # print("Done HBEMD on comlex real")
    # bemd2 = BEMD()
    # imfs_hue_cplx_imag = bemd2.HBEMD(hue_cplx.imag,max_imf = 4)
    # print("Done HBEMD on comlex imag")
    # imfs_no = min(imfs_hue_cplx_real.shape[0], imfs_hue_cplx_imag.shape[0])

    # for i in range(0,imfs_no):
    #     imf_hue_real = imfs_hue_cplx_real[i]
    #     imf_hue_imag = imfs_hue_cplx_imag[i]
    #     # imf_arctan_hue = np.arctan2(imf_hue_imag,imf_hue_real)/np.pi
    #     imf_arctan_hue = np.arctan2(imf_hue_real,imf_hue_imag)/np.pi
    #     imf_hue_degree = np.rad2deg(imf_arctan_hue)
    #     imf_hue_degree_old = np.copy(imf_hue_degree)
    #     imf_hue_degree = changeNegativeHueToPositive(imf_hue_degree_old)
    #     plt.imshow(imf_arctan_hue,cmap='hsv')
    #     plt.colorbar()
    #     plt.show()

def saveLogFile(fileName, values):
    np.savetxt("/home/trongan93/Projects/NCKUCubeSAT/CubeSATAI/log/" + fileName, values, delimiter=',', fmt='%s')

def changeNegativeHueToPositive(hue_channel):
    for j in range(0,hue_channel.shape[0]):
            for k in range(0, hue_channel.shape[1]):
                if(hue_channel[j][k] < 0):
                    hue_channel[j][k] = hue_channel[j][k] + 360
    return hue_channel

def multiple_j(hue_channel):
    result = np.zeros(hue_channel.shape,dtype=np.complex)
    for l in range(0,hue_channel.shape[0]):
        for m in range(0,hue_channel.shape[1]):
            result[l][m] = complex(0,hue_channel[l][m])
    return result

def OutputRaw_Y8(Raw,FileName):
    # save the residual
	out = Raw
	out.flatten()
	# print ('output image shape:',out.shape,'data type',out.dtype)
	FileName = 'DataTest/' + FileName
	fo = open(FileName, 'wb')
	fo.write(out)
	fo.close()
	# print ('Output Raw success',FileName)
	return 

if __name__ == '__main__':
    main()
