
# https://github.com/chalmersgit/SphericalHarmonics/blob/master/sphericalHarmonics.py
import os, sys, re
import numpy as np
import imageio as im
from PIL import Image
import cv2 # resize images with float support
from scipy import ndimage # gaussian blur
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Spherical harmonics functions
def P(l, m, x):
	pmm = 1.0
	if(m>0):
		somx2 = np.sqrt((1.0-x)*(1.0+x))
		fact = 1.0
		for i in range(1,m+1):
			pmm *= (-fact) * somx2
			fact += 2.0
	
	if(l==m):
		return pmm * np.ones(x.shape)
	
	pmmp1 = x * (2.0*m+1.0) * pmm
	
	if(l==m+1):
		return pmmp1
	
	pll = np.zeros(x.shape)
	for ll in range(m+2, l+1):
		pll = ( (2.0*ll-1.0)*x*pmmp1-(ll+m-1.0)*pmm ) / (ll-m)
		pmm = pmmp1
		pmmp1 = pll
	
	return pll

def factorial(x):
	if(x == 0):
		return 1.0
	return x * factorial(x-1)

def K(l, m):
	return np.sqrt( ((2 * l + 1) * factorial(l-m)) / (4*np.pi*factorial(l+m)) )

def SH(l, m, theta, phi):
	sqrt2 = np.sqrt(2.0)
	if(m==0):
		if np.isscalar(phi):
			return K(l,m)*P(l,m,np.cos(theta))
		else:
			return K(l,m)*P(l,m,np.cos(theta))*np.ones(phi.shape)
	elif(m>0):
		return sqrt2*K(l,m)*np.cos(m*phi)*P(l,m,np.cos(theta))
	else:
		return sqrt2*K(l,-m)*np.sin(-m*phi)*P(l,-m,np.cos(theta))

def shEvaluate(theta, phi, lmax):
	if np.isscalar(theta):
		coeffsMatrix = np.zeros((1,1,shTerms(lmax)))
	else:
		coeffsMatrix = np.zeros((theta.shape[0],phi.shape[0],shTerms(lmax)))

	for l in range(0,lmax+1):
		for m in range(-l,l+1):
			index = shIndex(l, m)
			coeffsMatrix[:,:,index] = SH(l, m, theta, phi)
	return coeffsMatrix

def shTermsWithinBand(l):
	return (l*2)+1

def shTerms(lmax):
	return (lmax + 1) * (lmax + 1)

def sh_lmax_from_terms(terms):
	return int(np.sqrt(terms)-1)

def shIndex(l, m):
	return l*l+l+m

def sh_visualize(lmax, coeffs):
    rows = lmax+1
    cols = shTermsWithinBand(lmax)
    imgIndex = 0

    cdict =	{'red':	((0.0, 1.0, 1.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 0.0, 0.0)),
                'green':((0.0, 0.0, 0.0),
                        (0.5, 0.0, 0.0),
                        (1.0, 1.0, 1.0)),
                'blue':	((0.0, 0.0, 0.0),
                        (1.0, 0.0, 0.0))}

    _, axs = plt.subplots(nrows=rows, ncols=cols, gridspec_kw={'wspace':0.1, 'hspace':0.1}, squeeze=True, figsize=(8, 8))
    for c in range(0,cols):
        for r in range(0,rows):
            axs[r,c].axis('off')

    for l in range(0,lmax+1):
        nInBand = shTermsWithinBand(l)
        colOffset = int(cols/2) - int(nInBand/2)
        rowOffset = (l*cols)+1
        for i in range(0,nInBand):
            axs[l, i+colOffset].axis("off")
            axs[l, i+colOffset].imshow(coeffs[:,:,imgIndex], cmap=LinearSegmentedColormap('RedGreen', cdict), vmin=-1, vmax=1)
            imgIndex+=1
    plt.show()

def relit(theta, phi, lmax, coeffs):
    res = coeffs.shape[0]
    lit = np.zeros((res, res))
    samples = shEvaluate(theta, phi, lmax)
    for x in range(res):
        for y in range(res):
            lit[x,y] = np.sum(coeffs[x,y,:] * samples)
    #lit = np.sum(coeffs * np.tile(samples, (res,res, 1)), axis=2)
    #plt.imshow(lit, cmap='gray')
    #plt.show()
    return lit

def plot_points(ax, coords, colors, scale=2):
    xyz = np.array([np.sin(coords[:,0]) * np.sin(coords[:,1]),
                np.sin(coords[:,0]) * np.cos(coords[:,1]),
                np.cos(coords[:,0])])

    ax.scatter(xyz[0,:]*colors*scale, xyz[1,:]*colors*scale, xyz[2,:]*colors*scale)

    # Draw a set of x, y, z axes for reference.
    ax_lim = 0.5
    ax.plot([-ax_lim, 1.5*ax_lim], [0,0], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [-ax_lim, 1.5*ax_lim], [0,0], c='0.5', lw=1, zorder=10)
    ax.plot([0,0], [0,0], [-ax_lim, 1.5*ax_lim], c='0.5', lw=1, zorder=10)
    ax_lim = 0.5
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_zlim(-ax_lim, ax_lim)
    ax.axis('off')

def plot_sh(ax, el, coeff, scale=2):
    """Plot the spherical harmonic of degree el and order m on Axes ax."""

    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    # Create a 2-D meshgrid of (theta, phi) angles.
    theta, phi = np.meshgrid(theta, phi)
    # Calculate the Cartesian coordinates of each point in the mesh.
    xyz = np.array([np.sin(theta) * np.sin(phi),
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta)])

    item_coeff = shEvaluate(theta, phi, el)

    view = item_coeff * np.tile(coeff, (100,100, 1))
    lens = np.sum(view, axis=2)

    Yx, Yy, Yz = lens * xyz * scale

    # Colour the plotted surface according to the sign of Y.
    cmap = plt.cm.ScalarMappable(cmap=plt.get_cmap('PRGn'))
    cmap.set_clim(-0.5, 0.5)

    ax.plot_surface(Yx, Yy, Yz,
                    facecolors=cmap.to_rgba(lens),
                    rstride=2, cstride=2)

def sample_sh(colors, coords, lat_step=10.0, long_step=12.0, lmax=2):
    res = colors.shape[1]
    lens = colors.shape[0]

    max_term = shTerms(lmax)

    coeffs = np.zeros((res, res, max_term))

    lat_size = np.deg2rad(lat_step)
    long_size = np.deg2rad(long_step)

    weight_base = lat_size * long_size #* ( np.pi * 4 )

    total_weight=0

    for i in range(lens):
        theta, phi = coords[i]
        weight = weight_base * np.sin(theta)
        item_coeff = shEvaluate(theta, phi, lmax)
        total_weight += weight
        view = np.einsum("ij,ijk->ijk", colors[i], item_coeff)
        coeffs += view * weight
        print("sample {0}".format(i))
    return coeffs

pattern = re.compile(r"\w*.[-0-9]*.[0-9]*.png")

def collect_images():
    coords = []
    colors = []

    for f in os.listdir("render"):
        m = pattern.match(f)
        if m:
            im = np.array(Image.open('render/{0}'.format(f)))
            theta = np.deg2rad(float(f.split(".")[1])+90)
            phi = np.deg2rad(float(f.split(".")[2]))
            coords.append([theta,phi])
            colors.append(im[:,:,0])
            print('img {0} at theta {1} phi {2}'.format(f, theta, phi))
    return np.array(colors), np.array(coords)