import os, re
from PIL import Image
import numpy as np

pattern = re.compile(r"\w*.[-0-9]*.[0-9]*.png")

def parse_image_angle(img_name):
    theta = np.deg2rad(float(img_name.split(".")[1])+90)
    phi = np.deg2rad(float(img_name.split(".")[2]))
    return theta, phi

def collect_images():
    coords = []
    colors = []

    for f in os.listdir("render"):
        m = pattern.match(f)
        if m:
            im = np.array(Image.open('render/{0}'.format(f)))
            theta, phi = parse_image_angle(f)
            coords.append([theta,phi])
            colors.append(im[:,:,0])
            alpha = im[:,:,3]
            print('img {0} at theta {1} phi {2}'.format(f, theta, phi))
    return np.array(colors), np.array(coords), np.array(alpha)