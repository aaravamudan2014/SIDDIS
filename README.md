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

As of now, we are unable to publish the dataset, however, we have published a curated dataset to serve as an example for training, testing and validation. Here are all the features offered by this repository. 

1. Architectural code in PyTorch for our proposed Residual Dense Networks (RDNs) and Residual Channel Attention Models (RCANs).
2. Extendable code for training, testing and performing hyperparameter searches on a chosen architecture. 
3. Jupyter notebooks to visualize the outputs from these models and generate required metrics. 
4. Code for Geomorpohological based downscaling operations. 
5. Templates for newer architectures. 

Features coming soon
1. Neural operators to enforce constraints. 

## Steps to run the code

While we trained the model on a PBS cluster [UCAR CASPER](https://arc.ucar.edu/knowledge_base/70549550), we provide generic shell scripts so it can be run on any GPU/CPU dependant system. We can also provide the PBS scripts upon request.

### Pre-requisites =
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

Note that we assume the original dataset has the following format (for a single image as an example)

```
-dataset
 --high_res
  --- 1.png
 -- low_res
  --- 1.png
 -- topo
  --- 1_1.png
  --- 1_2.png
```
2. An empty results/ folder.
3. An empty runs/ folder
4. A correctly populated run_models.sh


### Training the model
1. 


### Running hyperparameter searches

1. 

### Model Evaluation

1.  

