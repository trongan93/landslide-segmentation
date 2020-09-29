from HBEMD.HBEMD import HBEMD
import pylab as plt
def main():
    img_path_1 = "test-data/27.770733_85.868467_VERY_LARGE_cropped_LC08_L1TP_141041_20140918_20170419_01_T1.tiff"

    h_bemd = HBEMD(img_path_1)
    hue_img = h_bemd.hue_value()
    plt.imshow(hue_img, cmap='hsv')
    plt.show()
    imfs = h_bemd.sifting(hue_img, max_imf_value=2)
    for imf in imfs:
        plt.imshow(imf, cmap='hsv', origin='lower')
        plt.colorbar()
        plt.show()
if __name__ == '__main__':
    main()

