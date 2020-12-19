# Cloud SH Imposter  


## 1. Render GroundTruth
open `cloud_sh_imposter.hip` and render `mantra_sh` node, it will render 540 images under `/render` folder.

## 2. Reconstruct SH
```shell
python generate --lmax <max-degree-of-sh> --recollect True --reconstruct True
```
It will print remap range of each degree sh such as  
```
Normalize Bound is [[0.0, 849.3096980357466], [-219.28922181878514, 200.479331555482], [-96.79751392697558, 116.97197693521237]]
```
And It will generate all degrees and bands of sh coefficients stored in texture, visualize coeffs and reconstruction result.  

![img](imgs/sh_levels.png)
![img](imgs/compares.png)


## 3. Plot SH reconstructed at given pixel  
run
```shell
python plot.py --pixx <pixel-id-x> -- pixy <pixel-id-y>
```
![img](imgs/sh_plot.png)

## 4. Plot reconstruct loss with SH level increase  
run
```shell
python loss.py --maxlevel 10
```
![img](imgs/loss.png)