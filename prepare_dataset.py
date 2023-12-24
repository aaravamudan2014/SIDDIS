from osgeo import gdal
import os
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
high_res_files = []
low_res_files = []
topo_files = []

dataset_name='RedRiver'
dataset_dict = {'name':'data/'+dataset_name,
                'hres_location':'data/'+dataset_name+'_HRES_FIMs/',
                'lres_location':'data/'+dataset_name+'_LRES_FIMs/'
                }

if not os.path.exists(dataset_dict['name']):
    os.makedirs(dataset_dict['name'])
    os.makedirs(dataset_dict['name']+'/low_res/')
    os.makedirs(dataset_dict['name']+'/high_res/')
    # os.makedirs('data/Ghana/topo/')

sample_dict = {}

for file in os.listdir(dataset_dict['hres_location']):
    if file.endswith(".tif"):
        high_res_files.append(os.path.join(dataset_dict['hres_location'], file))

for file in os.listdir(dataset_dict['lres_location']):
    if file.endswith(".tif"):
        low_res_files.append(os.path.join(dataset_dict['lres_location'], file))

# for file in os.listdir("data/Ghana_Dataset_v0/GH_External_topo_files_v0/"):
#     if file.endswith(".tif"):
#         topo_files.append(os.path.join("data/Ghana_Dataset_v0/GH_External_topo_files_v0/", file))

for index, filename in enumerate(high_res_files):
  substring = filename.split('/')[-1].split('.')[0].split('_')
  image_id = substring[-2] + '_'+ substring[-1] 
  num = filename.split('_')[-1].split('.')[0]
  sample_dict[num] = index

indices = range(len(sample_dict))

def convert_files(filename_list, res_string):
    for sample_num, filename in enumerate(filename_list):
        if sample_num % 10 == 0:
            print('sample number: ', sample_num , ' out of ', len(filename_list))
        # if res_string == "topo":
        #     substring = filename.split('/')[-1].split('.')[0].split('_')
        #     image_id = substring[-2] + '_'+ substring[-1][:-4] 
        #     search_string = image_id
        #     index = sample_dict[search_string]
        #     src_ds = gdal.Open(filename)
        #     src_ds_1 = src_ds.ReadAsArray()[0]
        #     src_ds_2 = src_ds.ReadAsArray()[1]
        # else:
            # substring = filename.split('/')[-1].split('.')[0].split('_')
        num = filename.split('_')[-1].split('.')[0]
        search_string = num 
        index = sample_dict[search_string]
        src_ds = gdal.Open(filename)
        src_ds = src_ds.ReadAsArray()
        # if res_string == "topo":
        #     plt.imsave(name + '/'+res_string+'/'+str(index)+'_1.png',src_ds_1)
        #     plt.imsave('data/Ghana/'+res_string+'/'+str(index)+'_2.png',src_ds_2)
        #     np.save('data/Ghana/'+res_string+'/'+str(index)+'_1.npy', src_ds_1)
        #     np.save('data/Ghana/'+res_string+'/'+str(index)+'_2.npy', src_ds_2)
        
        # else:
        plt.imsave(dataset_dict['name']+'/'+res_string+'/'+str(index)+'.png',src_ds)
        np.save(dataset_dict['name']+'/'+res_string+'/'+str(index)+'.npy', src_ds)
        

convert_files(high_res_files, res_string = 'high_res')
convert_files(low_res_files, res_string = 'low_res')
# convert_files(topo_files, res_string = 'topo')