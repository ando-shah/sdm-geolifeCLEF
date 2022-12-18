# Introduction
Species Distribution Model based on GeoLifeCLEF 2022 dataset.

The goal of species distribution modeling or habitat suitability mapping is to predict the locations in which a given species could be present which is typically accomplished by statistical models that use environmental covariates to predict species observation data. This project aims to predict species distributions based on the [GeoLifeCLEF 2022 dataset](https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9/data) which pairs a set of satellite images, altitude, land cover and environmental covariates with each observation location. 

In this work, I pursue two goals:
1. To reduce the bias due to the long-tailed nature of species observation data inherent to the GeoLifeCLEF dataset.
2. Explore a framework for fine-tuning existing large scale (biased) models with localized data to make higher quality predictions. 

## Goal 1

I investigate the use of a modified kernel estimator to spatially diffuse the probability of species observations. Steps: 

1. Create 0.05 x 0.05 degree latitude-longitude grid over the bounds of the AOI and clip it to the boundaries of the AOI. Aggregate all species observations within a single grid cell and ascribe those observations to that grid cell.
2. Apply a modified kernel density estimator (KDE) to spatially diffuse the probability of a species being observed over an area defined by the radius of the kernel - this has to be done per species. While this will not resolve the hard positive assignment of species to samples, it provides some notion of implicit spatial regularisation. The modification to the KDE was the removal of the normalization term in the denominator, to convert the output to a pure probability over space, instead of the probability density. [Kellenberger et. al 2022](https://ieeexplore.ieee.org/abstract/document/9883627/?casa_token=cihPmizbDJgAAAAA:j2Vt-9PC8HeShYgwagXkQcJ4xyjy273AVUzadGVKixagCms8gCv-xoIdwJT3NTngiq7dj7RbL_kX) employed a semantically similar technique which they named ’spatial label swap’ whereby a species observation is replaced by another species within a localized region, with a 10% probability. I apply a fixed radius to all species, leading to equal diffusion for oversampled and undersampled species alike, a drawback of this approach.

A data pipeline was developed to fetch Sentinel-2 RGB-NIR tiles, cropped to the same size as the grid (0.05 x 0.05 degrees) and crop the environmental rasters to the same area after projecting them to the CRS of the Sentinel-2 tiles. Finally they are harmonized, since the resolution of the environmental layers (either 30arcsec or 250m) doesnt match that of the satellite images (10m). This step ensures that all the layers are referring to the same patch of real space.


## Goal 2

The central question being pursued in the second goal is: can local predictions be de-biased using large-scale models that are inherently biased? In the context of conservation financing, it is often very beneficial to have an accurate picture of the health of local spatially bounded habitats. For mechanisms such as voluntary biodiversity credits and biodiversity offsets (an emerging instrument), green and blue bonds, and debt-for-nature swaps, projects are required to provide regular assessment of the health of habitats covered by the project. Typically these assessments comprise of point samples of high effort (i.e. usually considered presence-absence) and usually employ naive interpolation methods to fill large spatial gaps. SDMs could provide a way to interpolate between those high quality datapoints.

At the time of writing, I did not have access to a high quality localized dataset. So I use the GeoLifeCLEF dataset itself as one. I extract observations within California (the area of interest or AOI) and try to predict the species distribution (k=5079 classes or unique species).

But first, I use an Inception-V3 model trained by [Deneu et. al 2021](https://gitlab.inria.fr/bdeneu/cnn-sdm) on the France subset of the same dataset as a base model. Then I aim to transfer learn features from this model by retraining on the California dataset. The two primary methods of transfer learning are fine-tuning and feature extraction. With fine tuning, the base model parameters are loaded and then the model is trained with only the out-of-sample transfer dataset. In feature extraction, all except the last layer (typically a classification or regression head) parameters are frozen and only the last layer is trained with the transfer dataset. In this project, the base model was trained by Deneu et. al on a total of 77 raster layers (of which 33 are one-hot land cover layers), but I only had 31 layers (the major missing layers were land cover and altitude, which I was not able to extract in time, and probably had an effect on the outcome). This meant that the parameters of the first convolution layer would not match between our models, and this completely invalidated the approach of feature extraction based transfer learning. I implemented fine-tuning by replacing the base model’s first convolution layer with an untrained one that could ingest 31 layers, and a classification head that would provide predictions for K=5079 classes. This entire network was trained with the AOI training set.

## Dependencies
Dependencies for this repo:

Packages:
- pytorch
- scikit-learn
- pandas
- numpy
- rasterio
- geopandas 

Data dependencies (not hosted on this repo):
- Environmental rasters from the Kaggle competition. Visit [this link](https://www.kaggle.com/competitions/geolifeclef-2022-lifeclef-2022-fgvc9/data) to download them.
- Once the tiles are processed, they will need to be stored locally and references in `/training/cnn_pretrained.ipynb` and `/training/cnn_scratch.ipynb` will have to be updated.

## Organization

`/preprocessing`: contains the gridding and KDE code

`/raster` : contains all data pipeline to fetch and harmonize all data layers

`/models` : Inception-V3 models created by Deneu et. al

`/training` : Notebooks for training from scratch and transfer learning

`/postprocessing` : Notebooks for visualizing results

`/data` : shapefiles for California

## End notes

This is still a work in progress, and near-future work includes:
- Use [MOSAIKS](https://arxiv.org/abs/2010.08168) as a feature extractor
- Using multiple pretrained ResNet
- Use land cover (or habitat type) and elevation covariates
- Create an adaptive Kernel which applies different radii for different species according to their popularity.


Thanks!





