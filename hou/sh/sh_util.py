
# https://github.com/chalmersgit/SphericalHarmonics/blob/master/sphericalHarmonics.py
import os, sys
import numpy as np
import imageio as im
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

def grey2colour(greyImg):
	return (np.repeat(greyImg[:,:][:, :, np.newaxis], 3, axis=2)).astype(np.float32)

def colour2grey(colImg):
	return ((colImg[:,:,0]+colImg[:,:,1]+colImg[:,:,2])/3).astype(np.float32)

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

def relit(phi, theta, lmax, coeffs):
    res = coeffs.shape[0]
    lit = np.zeros((res, res))
    samples = shEvaluate(theta, phi, lmax)
    lit = np.sum(coeffs * np.tile(samples, (res,res, 1)), axis=2)
    '''for l in range(0,lmax+1):
        for m in range(-l,l+1):
            index = shIndex(l, m)
            lit += coeffs[:,:,index] * SH(l, m, theta, phi)'''
    plt.imshow(lit, cmap='gray', vmin=0, vmax=0.1)
    plt.show()