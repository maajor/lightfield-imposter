import os
import re
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

from sh.sh_util import shTerms, shEvaluate, relit, sh_visualize
from matplotlib.colors import LinearSegmentedColormap

pattern = re.compile(r"\w*.[-0-9]*.[0-9]*.png")


LMAX=3
PI = 3.1415926

def deg2rad(deg):
    return deg * PI / 180.0

def sample_sh(lat_step=10.0, long_step=12.0, res=512, lmax=2):
    res = 512

    max_term = shTerms(lmax)

    coeffs = np.zeros((512, 512, max_term))

    lat_size = deg2rad(lat_step)
    long_size = deg2rad(long_step)

    weight_base = lat_size * long_size / ( PI * 4 )

    for f in os.listdir("render"):
        m = pattern.match(f)
        if m:
            im = np.array(Image.open('render/{0}'.format(f)))
            im = colour2grey(im) / 255.0
            theta = deg2rad(float(f.split(".")[1])+90)
            phi = deg2rad(float(f.split(".")[2]))
            weight = weight_base * math.sin(theta)
            item_coeff = shEvaluate(theta, phi, lmax)
            view = np.repeat(im[:,:,np.newaxis], max_term, axis=2) * np.tile(item_coeff, (res,res, 1))

            coeffs += view * weight
            print('img {0} at theta {1} phi {2}'.format(f, theta, phi))
    
    return coeffs

#coeffs = sample_sh(lmax=LMAX)

#np.save("cloud_sh", coeffs)

coeffs = np.load("cloud_sh.npy")

theta = deg2rad(7+90)
phi = deg2rad(264)
relit(phi, theta, LMAX, coeffs)
