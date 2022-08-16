# FCDSN-DC
![Teaser image](./docs/Network_header.png)
### An Accurate and Lightweight end-to-end Trainable Neural Network for Stereo Estimation with Depth Completion
Dominik Hirner, Friedrich Fraundorfer

An pytorch implementation of our accurate and lightweight end-to-end trainable CNN for stereo estimation with depth completion.
This method has been accepted and will be published at the **ICPR 2022** conference. If you use our work please cite our paper

The whole project is in pure python 3 and pytorch 1.2.0

A demo of the whole end-to-end method is available online in Google Colab: 
[Demo](https://colab.research.google.com/drive/10_QRckJdc19unydikcZIRZbTk_g1peHu?usp=sharing)

This repository contains

- jupyter notebooks for training and inference of disparity via a stereo-pair
- python3.6 code for training and inference
- trained weights for many publicly available datasets



## Trained weights
Dataset | branch |  simB | Incons
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
Middlebury | [mb](https://drive.google.com/file/d/1Bo3INQhCK1N17EuLkX7nnie46zHeqrQ2/view?usp=sharing) | [mb_simB](https://drive.google.com/file/d/1jJG3ZfjBRIrWzN2MQZ1TxC_9JWMh2l64/view?usp=sharing) | [Incons](https://drive.google.com/file/d/11DNIJmpCTZpmwEC-rrKjRmoHaUEEma-C/view?usp=sharing) |
Kitti2012 | [kitti2012](https://drive.google.com/file/d/1mQtJsb8gesI_9Sy16SdXT_f_FgnKfdCP/view?usp=sharing) | [kitti2012_simB](https://drive.google.com/file/d/1mcxjhUZO6JuokMHLSkq3Q6psdBATOJBJ/view?usp=sharing) | [Icons_KITTI2012](https://drive.google.com/file/d/1SwSke9euif9Kfa4qPRBW7TZwA3z555lv/view?usp=sharing) | 
Kitti2015 | [kitti2015](https://drive.google.com/file/d/1wY6h1D89e_Mx9aOFSab3FxedDn0n6WiP/view?usp=sharing) | [kitti2015_simB](https://drive.google.com/file/d/1tQRzwjeUE16WS9V2U9P_YHw5fMuVJ7uE/view?usp=sharing) | [Incons_KITTI2015](https://drive.google.com/file/d/1L5QcqW5Ph9gmFpqV1rlMW0-pkINM3y4I/view?usp=sharing) | 
ETH3D | [ETH](https://drive.google.com/file/d/1i2oNAEk3gX4a_B2f7ei818btuLPk-gpV/view?usp=sharing) | [ETH_simB](https://drive.google.com/file/d/1gZWA6f_Gfm7-Qmfdxim5CeZ__gxj15vb/view?usp=sharing) | [Incons_ETH](https://drive.google.com/file/d/1BYwput_eSdcQYPDp5G7tJmsm9YRY5aJi/view?usp=sharing) |

## Usage
We use a trainable guided filter for the cost-volume (see [project](http://wuhuikai.me/DeepGuidedFilterProject/)). This can be installed via pip.

 ```pip install guided-filter-pytorch```
### Training 
If you want to train our method (either from scratch or continue from one of our provided weights), use the provided config files from the root of this repository and change them to fit your needs. If you want to train from scratch, first train the feature extractor and the similarity function as follows:
```python FCDSN_train.py config/FCDSN-CONFIG-FILE.cfg```

Afterwards, use the output created by this file to train the depth-completion part as follows: 

```python DC_train.py config/DC-CONFIG-FILE.cfg```

Note that if you want to do transfer training on the depth-completion, the following files must be found for each sample in an individual folder: 
- im0.png
- disp0GT.pfm
- disp_s.pfm
- keep_mask.png
- upd_mask.png

### Inference 
If you want to do inference on any rectified image-pair call the *test.py* function from the root of this repository as follows: 

 ```python test.py --weights_b path/to/branch_weight --weights_s path/to/sim_weights --weights_f path/to/fill_weights --left path/to/left_im.png --right /path/to/right_im.png --max_disp max_disp --out /path/to/out/out_name```
 
#### Example on Middlebury
Download the Middlebury weights from the link above and put it in the *weights* folder in the root of this repository. Then copy and paste the following: 

```python test.py --weights_b weights/branch/mb --weights_s weights/simb/mb_simB --weights_f weights/fill/Incons --left example/im0.png --right example/im1.png --max_disp 145 --out adirondack```

If everything went ok this should produce the following output: 

- adirondack.pfm: filtered disparity output of the FCDSN network
<img src=./docs/adirondack_disp.png width=50% height=50%>
- adirondack_s.pfm: disparity map with removed inconsistencies
<img src=./docs/adirondack_disp_s.png width=50% height=50%>
- adirondack_filled.pfm: output of DC network
<img src=./docs/adirondack_disp_filled.png width=50% height=50%>


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

