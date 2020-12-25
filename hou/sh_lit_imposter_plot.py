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

def main(pix_x, pix_y):
    dataset = np.load("dataset/cloud_sh_set.npy", allow_pickle=True)
    coeffs = np.load("dataset/cloud_sh.npy")
    LMAX = int(np.sqrt(coeffs.shape[2])-1)
    
    plt.rc('text', usetex=True)

    fig = plt.figure(figsize=2*plt.figaspect(1.))
    ax = fig.add_subplot(projection='3d')
    plot_points(ax, coords=dataset.item().get('coords'), colors=dataset.item().get('colors')[:,pix_y,pix_x])
    plot_sh(ax, LMAX, coeffs[pix_y,pix_x,:])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-x", "--pixx", type=int, help="pixel x index", default=323)
    parser.add_argument("-y", "--pixy", type=int, help="pixel y index", default=236)
    args = vars(parser.parse_args())

    main(args['pixx'], args['pixy'])