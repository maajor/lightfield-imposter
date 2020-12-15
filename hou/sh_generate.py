import os, re
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D

from sh.sh_util import shTerms, colour2grey, shEvaluate, relit, sh_visualize, plot_points, plot_sh, collect_images, sample_sh
from matplotlib.colors import LinearSegmentedColormap

RECOLLECT = False
RESAMPLE = False

imgs = [f for f in os.listdir("render") if f.startswith("cloud_sh")]
img = imgs[290]

if RECOLLECT:
    colors, coords = collect_images()
    np.save("cloud_sh_set", {"colors":colors, "coords":coords})
else:
    dataset = np.load("cloud_sh_set.npy", allow_pickle=True)

LMAX=2

if RESAMPLE:
    coeffs = sample_sh(colors=dataset.item().get('colors'), coords=dataset.item().get('coords'), lmax=LMAX)
    np.save("cloud_sh", coeffs)
else:
    coeffs = np.load("cloud_sh.npy")
    

imgs = [f for f in os.listdir("render") if f.startswith("cloud_sh")]
imgid = random.randint(0,540)
img = imgs[imgid]
print(img)
theta = np.deg2rad(float(img.split(".")[1])+90)
phi = np.deg2rad(float(img.split(".")[2]))
img_original = np.array(Image.open('render/{0}'.format(img)))

lit = relit(theta, phi, LMAX, coeffs)
diff = np.sum(lit - img_original[:,:,0])
print(diff/(512*512))

plt.imshow(lit, cmap='gray')
plt.show()

'''
plt.rc('text', usetex=True)

fig = plt.figure(figsize=2*plt.figaspect(1.))
ax = fig.add_subplot(projection='3d')
plot_points(ax, coords=dataset.item().get('coords'), colors=dataset.item().get('colors')[:,pix_y,pix_x])
plot_sh(ax, LMAX, coeffs[pix_y,pix_x,:])
plt.show()'''