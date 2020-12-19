import os, re
import random
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from utils.sh import shTerms, shEvaluate, relit, sh_visualize, plot_points, plot_sh, sample_sh
from utils.loader import collect_images, parse_image_angle
from matplotlib.colors import LinearSegmentedColormap

def main(max_level=3):
    
    dataset = np.load("dataset/cloud_sh_set.npy", allow_pickle=True)
    colors = dataset.item().get('colors')
    coords = dataset.item().get('coords')
    num_imgs, imgres, _ = colors.shape

    diff_perlevel = []

    for lmax in range(0, max_level):
        print("Test LMAX {0}".format(lmax))
        # compute coeff per sh level
        data_path = "dataset/cloud_sh_{0}.npy".format(lmax)
        if os.path.exists(data_path):
            coeffs = np.load(data_path)
        else:
            coeffs = sample_sh(colors=colors, coords=coords, lmax=lmax)
            np.save("dataset/cloud_sh_{0}".format(lmax), coeffs)

        # compute loss per sh level
        diff_all = 0.0
        count = 0
        for i in range(0, num_imgs, 10):
            img_orig = colors[i]
            theta, phi = coords[i]
            lit = relit(theta, phi, lmax, coeffs)
            diff = np.sum(np.abs(lit - img_orig))/(imgres*imgres)
            diff_all += diff
            count += 1
            print("sample {0} loss is {1}".format(i, diff))
        diff_avg = diff_all / count
        print("Average pixel loss at lmax:{0} is {1}".format(lmax, diff_avg))
        diff_perlevel.append(diff_avg)

    plt.style.use('ggplot')
    plt.title('Pixel Loss with SH Degress Increases')
    plt.xlabel('Spherical Harmonic Degree')
    plt.ylabel('Average Loss Per Pixel')
    plt.plot(range(0, max_level), diff_perlevel)
    plt.savefig("loss.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--maxlevel", type=int, help="pixel x index", default=4)
    args = vars(parser.parse_args())
    main(args['maxlevel'])