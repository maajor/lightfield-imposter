# Several Lightfield Imposter Research

## 1 SH Lit Imposter  

Inspired by [6-way lightmap](https://realtimevfx.com/t/smoke-lighting-and-texture-re-usability-in-skull-bones), this is a better approach in quality.  
Basic idea is to encode per-pixel spherical harmonic coefficents of a imposter in texture.  

Compare with 6-way lightmap on right, sh lit imposter is on left.  
![img](imgs/sh_vs_6way.gif)  

Ramp color + flowmap implemented in unity (not included in this project)  
![img](imgs/in_unity.gif) 


### 1. Render GroundTruth
open `cloud_sh_imposter.hip` and render `mantra_sh` node, it will render 540 images under `/render` folder.

### 2. Reconstruct SH
```shell
python sh_lit_imposter_generate --lmax <max-degree-of-sh> --recollect True --reconstruct True
```
It will print remap range of each degree sh such as  
```
Normalize Bound is [[0.0, 849.3096980357466], [-219.28922181878514, 200.479331555482], [-96.79751392697558, 116.97197693521237]]
```
And It will generate all degrees and bands of sh coefficients stored in texture, visualize coeffs and reconstruction result.  

![img](imgs/sh_levels.png)
![img](imgs/compares.png)


### 3. Plot SH reconstructed at given pixel  
run
```shell
python sh_lit_imposter_plot.py --pixx <pixel-id-x> -- pixy <pixel-id-y>
```
![img](imgs/sh_plot.png)

### 4. Plot reconstruct loss with SH level increase  
run
```shell
python sh_lit_imposter_loss.py --maxlevel 10
```
![img](imgs/loss.png)


## 2 SH 3D Imposter  

Here I want to test compress a 3d imposter texture with SH, result is not good.  

![img](imgs/impostor_sh_N.jpg)

### 1. Render GroundTruth  
Render any geometry with SideFx Labs' Labs Imposter Texture, on "Full 3D Imposter" mode.  

### 2. Reconstruct SH
```shell
python sh_3dimposter_generate.py
```
It will reconstruction result.  

![img](imgs/compare_3dimposter_sh.png)

You can see the result is poor, as 3d imposter more high-freq at each pixel.  

## 3. Plot reconstruct loss with SH level increase  
run
```shell
python sh_3dimposter_loss.py --maxlevel 10
```
![img](imgs/loss_3dimposter_sh.png)

## 3 NN 3D Imposter

Here I want to test compress a 3d imposter texture with neural network (fully connected network).  

![img](imgs/impostor_sh_N.jpg)

### 1. Render GroundTruth
Same as SH 3D Imposter

### 2. Train Neural Network
```shell
python nn_imposter_train.py
```
It will train the network and plot training loss.  

![img](imgs/train_imposternn.png)

## 3. Plot result  
run
```shell
python nn_imposter_plot.py
```
![img](imgs/compare_nn_imposter.gif)

# Refs.  
Read more about explanation of this experiment on my blog.  