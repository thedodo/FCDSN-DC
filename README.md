# FCDSN-DC
![Teaser image](./docs/Network_header.png)
### An Accurate and Lightweight end-to-end Trainable Neural Network for Stereo Estimation with Depth Completion
Dominik Hirner, Friedrich Fraundorfer

An pytorch implementation of our accurate and lightweight end-to-end trainable CNN for stereo estimation with depth completion.
This method has been accepted and will be published at the **ICPR 2022** conference. If you use our work please cite our paper

The whole project is in pure python 3 and pytorch 1.2.0

This repository contains

- jupyter notebooks for training and inference of disparity via a stereo-pair
- python3.6 code for training and inference
- trained weights for many publicly available datasets

### TODO
all scripts/notebooks tested for:
- [X] MB
- [ ] KITTI2012
- [ ] KITTI2015
- [ ] ETH


## Examples
RGB             |  Disparity
:-------------------------:|:-------------------------:
![AdironRGB](./docs/examples/MB_Adrion_RGB.png)  |  ![AdironDisp](./docs/examples/MB_Adiron.png) |
![PlaytRGB](./docs/examples/MB_playtable_RGB.png)  |  ![PlaytDisp](./docs/examples/playtable.png) |
![2RGB](./docs/examples/2RGB.png)  |  ![2Disp](./docs/examples/2Disp.png) |
![3RGB](./docs/examples/3RGB.png)  |  ![3Disp](./docs/examples/3Disp.png) |
![ArchRGB](./docs/examples/architecture1_flickrRGB.png)  |  ![ArchDisp](./docs/examples/architecture1_flickrDisp.png) |
![WomanRGB](./docs/examples/woman_holopixRGB.jpg)  |  ![WomanDisp](./docs/examples/woman_holopixDisp.png) |
![SheepRGB](./docs/examples/sheep_holopixRGB.jpg)  |  ![SheepDisp](./docs/examples/sheep_holopixDisp.png) |

