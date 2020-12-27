from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils.sh import shTerms, shEvaluate, relit, sh_visualize, plot_points, plot_sh, sample_sh, normalize
from utils.loader import collect_images, parse_image_angle, collect_3d_imposter_images
import random

LMAX = 3
GRID = 16

collect_3d_imposter_images()
dataset = np.load("dataset/3d_imposter.npy", allow_pickle=True)
colors = dataset.item().get('colors')
colors = (colors + 0.5) * 127.5
coords = dataset.item().get('coords')
coords[:,0] = (coords[:,0] + 0.5 ) * np.pi
coords[:,1] = (coords[:,1] + 0.5 ) * np.pi * 2

coeffs_r = sample_sh(colors=colors[:,:,:,0], coords=coords, lat_step=180/GRID, long_step=360/GRID, lmax=LMAX)

coeffs_normalized = normalize(LMAX, coeffs_r)

coeffs_length = (LMAX+1)*(LMAX+1)

# visualize sh coeffs
sh_visualize(LMAX, coeffs_normalized)

# validate the result
_, axs = plt.subplots(nrows=1, ncols=2)
imgid = random.randint(0,256)
img_original = colors[imgid,:,:,0]
theta, phi = coords[imgid,:]

lit = relit(theta, phi, LMAX, coeffs_r)

axs[0].set_title("Original")
axs[0].imshow(img_original, cmap='gray')
axs[1].set_title("SH Reconstruction")
axs[1].imshow(lit, cmap='gray')
plt.show()