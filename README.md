# WMH_CNS
WMH segmentation using well-known architectures in the literature

The purpose of this document is to enable the reproducibility of our experiments related to White-matter Hyperintensity (WMH) segmentation using public and available U-NET architectures.
This pipeline can be run in any dataset related to WMH. However, we strongly suggest using 256x256x64 images, or images that have multiple of 64 dimension. 

We patch our image into volumes of 64x64x64 to detect the WMH in smaller level. We train our network based on these patches. 
To apply the network, two ways are possible: 1) input the entire image to the prediction model; 2) (recommended) patch the image in smaller volumes and concatenate them in the end.

Please find the models on the following link: __
Main File: train_model_3DUNET.ipynb
