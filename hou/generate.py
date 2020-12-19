import os, re
import random
import argparse
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from utils.sh import shTerms, shEvaluate, relit, sh_visualize, plot_points, plot_sh, sample_sh, normalize
from utils.loader import collect_images, parse_image_angle
from matplotlib.colors import LinearSegmentedColormap

def main(LMAX=2, recollect = False, resample = False):
    
    if recollect:
        colors, coords, alpha = collect_images()
        np.save("dataset/cloud_sh_set", {"colors":colors, "coords":coords, "alpha":alpha})
    dataset = np.load("dataset/cloud_sh_set.npy", allow_pickle=True)

    alpha = dataset.item().get('alpha')
    im = Image.fromarray(alpha.astype(np.uint8))
    im.save("cloud_sh_alpha.png")

    if resample:
        coeffs = sample_sh(colors=dataset.item().get('colors'), coords=dataset.item().get('coords'), lmax=LMAX)
        np.save("dataset/cloud_sh", coeffs)
    else:
        coeffs = np.load("dataset/cloud_sh.npy")

    coeffs_normalized = normalize(LMAX, coeffs)
    
    coeffs_length = (LMAX+1)*(LMAX+1)
    for i in range(0, coeffs_length, 3):
        im = Image.fromarray((coeffs_normalized[:,:,i:i+3]*255.0).astype(np.uint8))
        im.save("cloud_sh{0}.png".format(int(i/3)))

    # visualize sh coeffs
    sh_visualize(LMAX, coeffs_normalized)
    
    # validate the result
    _, axs = plt.subplots(nrows=1, ncols=2)

    imgs = [f for f in os.listdir("render") if f.startswith("cloud_sh")]
    img = imgs[random.randint(0,540)]
    theta, phi = parse_image_angle(img)
    img_original = np.array(Image.open('render/{0}'.format(img)))

    lit = relit(theta, phi, LMAX, coeffs)

    axs[0].set_title("Original")
    axs[0].imshow(img_original[:,:,0], cmap='gray')
    axs[1].set_title("SH Reconstruction")
    axs[1].imshow(lit, cmap='gray')
    plt.savefig("imgs/compares.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lmax", type=int, help="SH Level Max", default=2)
    parser.add_argument("-rc", "--recollect", type=bool, help="recollect images", default=False)
    parser.add_argument("-rs", "--resample", type=bool, help="resample sh", default=False)
    args = vars(parser.parse_args())

    main(args['lmax'], args['recollect'], args['resample'])