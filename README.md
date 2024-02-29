---
title: SIDDIS 
output: pdf_document
---

# SIDDIS: Satellite Imagery Downscaling via Deep Image Super-resolution

## Project thesis

Present trends in climate and land use change clearly point to an ever increasing flood risk that can lead to severe riverine and coastal flooding across the globe. Leveraging remote sensing imagery is key to estimating and predicting future flood inundation extents. Currently, though, long-term records of flood inundation observations from publicly-accessible imaging products are at a spatio-temporal resolution, whose coarsity significantly curtails their potential for high-accuracy predictions.

We explore the potential of using simulated Flood Inundation Maps (FIMs) to train such models for super-resolution and explore the viability of its application in various regions across the world. We presented a preliminary version of this at AGU 2021 [Fall meeting](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=qLU7cGgAAAAJ&citation_for_view=qLU7cGgAAAAJ:UeHWp8X0CEIC). While we haven't officially released this repository, any questions and issues regarding the code can be addressed towards [Akshay Aravamudan](https://aaravamudan2014.github.io/Akshay-Aravamudan/). 


More information about the project and its participants can be found in [here](https://gio-research.ai/project/siddis/).

## Repository Details
This repository contains all the source code for the SIDDIS project. As of right now, we are in the process of submitting this work to a journal. More information regarding this will be revealed shortly.

1. Architectural code in PyTorch for our proposed Residual Dense Networks (RDNs) and Residual Channel Attention Models (RCANs).
2. Extendable code for training, testing and performing hyperparameter searches on a chosen architecture. 
3. Jupyter notebooks to visualize the outputs from these models and generate required metrics. 
4. Code for Geomorpohological based downscaling operations. 
5. Templates for newer architectures. 

## Steps to run the code

While we trained the model on a PBS cluster [UCAR CASPER](https://arc.ucar.edu/knowledge_base/70549550), we provide generic shell scripts so it can be run on any GPU/CPU dependant system. We can also provide the PBS scripts upon request.

### Pre-requisites 
1. Have a data folder with the appropriate datasets in .h5 form. This is a pytorch dataset that has been saved in the hdf5 =format. Each entry in the dataset follows the format
```
h5_filename = 'test_file.h5'

h5_val_file = h5py.File(h5_filename, 'w')
hr_val_group = h5_val_file.create_group('hr')
lr_val_group = h5_val_file.create_group('lr')
topo_val_group_1 = h5_val_file.create_group('topo_1')
topo_val_group_2 = h5_val_file.create_group('topo_2')

req_indices = range(req_indices)

for image_index in req_indices:
    # try:
    hr = pil_image.open(str(dataset_type)+'/high_res/'+str(image_index)+'.png').convert('LA')
    hr = np.array(hr)[:,:,0:1]
    lr = pil_image.open(str(dataset_type)+'/low_res/'+str(image_index)+'.png').convert('LA')
    lr = np.array(lr)[:,:,0:1]
    topo_1 = pil_image.open(str(dataset_type)+'/topo/'+str(image_index)+'_1.png')
    topo_2 = pil_image.open(str(dataset_type)+'/topo/'+str(image_index)+'_2.png')
    
    # except:
      # continue
    lr_group.create_dataset(str(patch_idx), data=lr)
    hr_group.create_dataset(str(patch_idx), data=hr)
    topo_group_1.create_dataset(str(patch_idx), data=topo_1)
    topo_group_2.create_dataset(str(patch_idx), data=topo_2)
    
    patch_idx += 1
  
```
 This can also be achieved by simply running the `create_dataset.py` file for each individual dataset.

### Training the model
1. Once the dataset has been prepared (training.h5 and validation.h5), create a `runs` folder in the main directory, change the `mode` variable in `run_models.sh` to `train`.


### Running hyperparameter searches
1. First make sure that there is an empty folder called `optuna_searches`. Once the datasets have been prepared (training.h5 and validation.h5), create a `runs` folder in the main directory, change the `mode` variable in `run_models.sh` to `optuna_search`.

### Model Evaluation
1.  Once the model has been trained and best models are saved in `runs/*run_directory/*` and the test dataset has been prepared (.h5 files), change the the `mode` variable in `run_models.sh` to `evaluate`. An alternative to running evaluation and replicating the paper results is by running the Colab notebook described in the following subsection.


### Supporting Notebooks

In order to allow ease of evaluation of the model, we provide a google colab notebook found here https://colab.research.google.com/drive/1ANo82R6XpnW5LVg7aOPNMwkwVDEwhuOh?authuser=1. 

This notebook is designed to 
1. Load evaluation data for both synthetic data and real-world data
2. Run and obtain predictions for individual model
3. An interactive tool to visualize individual predictions from various models for each dataset
4. Obtain Clopper-Pearson Prediction Intervals (PI) for accuracies.
5. Conduct multiple comparison test using the Holm-Bonferroni procedure to compare several models. This procedure is designed to control the Family-Wise Error Rate (FWER).

### Dataset

The dataset is available under the `data/` directory. In order to run the code, make sure to unzip all the files under the data folder. The following paragraph describes the dataset in detail. 

Abstract: This dataset consists of synthetic and real-world data of flooding events for selected regions in the world. For both the synthetic and real-world data, once the flood inundation maps (FIM) were obtained, chips of 100x100 pixels were extracted with each pixel representing a 30mx30m area. The low-resolution water fraction maps were constructed by a bilinear downsampling of the aforementioned high-resolution maps to produce 10x10 images with each pixel representing a 300mx300m area. 
The synthetic data was produced from hydrodynamic simulations of riverine flooding in the State of Iowa using the HEC-RAS model. The low- and high-resolution synthetic images were constructed from model outputs. These raw image patches are provided as low-resolution (300m) and high-resolution (30m) Tiff files. Additionally, we provide processed versions of these GeoTiff files that have been split into training, test and validation dataset, each of which contains low-res (10x10 floats between 0 and 1) and high-res (100x100 binary values) images as both .npy and .png files. 
Secondly, we provide real-world Landsat-8 flooding data for the regions of Des Moines and Cedar River in Iowa, Meuse River in Western Europe, Red River of the North and the Nasia River in Ghana. We extract images with 30mx30m patches from Landsat-8’s 30m product and then evaluate the Spectral Water Index (SWI) of the multi-channel Landsat-8 images. The SWI was used to threshold the image to classify pixels as flooded and non-flooded. This threshold was determined based on quantitative comparison of the histograms between flooded and non-flooded regions. Similar to the synthetic data, this dataset contains the GeoTiff files for both low- and high-resolution images. We also provide the processed versions of each of the real-world datasets in .npy and .png files. Details on the processing and generation of the datasets are provided in Aravamudan et. al. 2024 [placeholder until paper is published]. 

Instructions:
We provide processed data from two sources: (i) synthetic data from the Iowa Flood Center and (ii) Landsat-8 real-world data. Firstly, this dataset contains the raw GeoTiff images obtained from physics-based simulations and data post-processed from Landsat-8. In addition to this, for convenience, we also provide the processed images for all datasets. For each of the provided datasets and for each region, there is a folder containing the high-resolution flood inundation map —  a 100x100 binary-valued image — and a folder containing the low-resolution map — a 10x10 float-valued image with values between 0 and 1. 
The dataset is organized as follows.

1. SYN-raw — contains tiff files for low- and high-resolution images 
   - IowaSynthetic_HRES_FIMs.zip
   - IowaSynthetic_LRES_FIMs.zip

2. SYN-processed
   - test.zip
   - training.zip
   - validation.zip
3. RW-raw
   - EU_HRES_FIMs.zip
   - EU_LRES_FIMs.zip
   - IowaCedar_HRES_FIMs.zip
   - IowaCedar_LRES_FIMs.zip
   - IowaDesMoines_HRES_FIMs.zip
   - IowaDesMoines_LRES_FIMs.zip
   - Ghana_HRES_FIMs.zip
   - Ghana_LRES_FIMs.zip
   - RedRiver_HRES_FIMs.zip
   - RedRiver_LRES_FIMs.zip
4.  RW-processed
   - EU.zip
   - Ghana.zip
   - IowaCedar.zip
   - IowaDesMoines.zip
   - RedRiver.zip

For each of the processed data folders, each zip contains two sub-folders namely high_res and low_res which further contain the .npy and .png images of the high-resolution and low-resolution images respectively. The file names have been indexed by the same number so that the low-resolution image can be mapped to the high-resolution image accurately. 




