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

def main(LMAX=2, recollect = False, resample = False):
    
    if recollect:
        colors, coords = collect_images()
        np.save("dataset/cloud_sh_set", {"colors":colors, "coords":coords})
    dataset = np.load("dataset/cloud_sh_set.npy", allow_pickle=True)

    if resample:
        coeffs = sample_sh(colors=dataset.item().get('colors'), coords=dataset.item().get('coords'), lmax=LMAX)
        np.save("dataset/cloud_sh", coeffs)
    else:
        coeffs = np.load("dataset/cloud_sh.npy")
    
    # validate the result
    _, axs = plt.subplots(nrows=1, ncols=2)

    imgs = [f for f in os.listdir("render") if f.startswith("cloud_sh")]
    img = imgs[random.randint(0,540)]
    theta, phi = parse_image_angle(img)
    img_original = np.array(Image.open('render/{0}'.format(img)))

    lit = relit(theta, phi, LMAX, coeffs)
    #diff = np.sum(lit - img_original[:,:,0])
    #print(diff/(512*512))

    axs[0].set_title("Original")
    axs[0].imshow(img_original[:,:,0], cmap='gray')
    axs[1].set_title("SH Reconstruction")
    axs[1].imshow(lit, cmap='gray')
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--lmax", type=int, help="SH Level Max", default=2)
    parser.add_argument("-rc", "--recollect", type=bool, help="recollect images", default=False)
    parser.add_argument("-rs", "--resample", type=bool, help="resample sh", default=False)
    args = vars(parser.parse_args())

    main(args['lmax'], args['recollect'], args['resample'])

'''
plt.rc('text', usetex=True)

fig = plt.figure(figsize=2*plt.figaspect(1.))
ax = fig.add_subplot(projection='3d')
plot_points(ax, coords=dataset.item().get('coords'), colors=dataset.item().get('colors')[:,pix_y,pix_x])
plot_sh(ax, LMAX, coeffs[pix_y,pix_x,:])
plt.show()'''