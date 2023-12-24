from osgeo import gdal
import os
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import numpy as np


high_res_files = []
low_res_files = []
topo_files = []


for file in os.listdir("data/IowaSynthetic_HRES_FIMs/"):
    if file.endswith(".tif"):
        high_res_files.append(os.path.join("data/IowaSynthetic_HRES_FIMs", file))
for file in os.listdir("data/IowaSynthetic_LRES_FIMs/"):
    if file.endswith(".tif"):
        low_res_files.append(os.path.join("data/IowaSynthetic_LRES_FIMs", file))

# for file in os.listdir("topo_files/"):
#     if file.endswith(".tif"):
#         topo_files.append(os.path.join("topo_files", file))


# create the training, validation and test directories
if not os.path.exists('data/training'):
    os.makedirs('data/training')
    os.makedirs('data/training/low_res/')
    os.makedirs('data/training/high_res/')
    # os.makedirs('training/topo/')
    
if not os.path.exists('data/test'):
    os.makedirs('data/test')
    os.makedirs('data/test/low_res/')
    os.makedirs('data/test/high_res/')
    # os.makedirs('test/topo/')
    

if not os.path.exists('data/validation'):
    os.makedirs('data/validation')
    os.makedirs('data/validation/low_res/')
    os.makedirs('data/validation/high_res/')
    # os.makedirs('validation/topo/')

sample_dict = {}

for index, filename in enumerate(high_res_files):
  image_id = filename.split('/')[1].split('.')[0].split('_')[-1]
  num = filename.split('_')[-1].split('.')[0]
  sample_dict[ num] = index
  # print(num)
  # input()

from sklearn.model_selection import train_test_split
import random
indices = range(len(sample_dict))
training_indices, test_indices = train_test_split(indices,random_state=11)
test_indices, validation_indices = train_test_split(test_indices, test_size=0.5,random_state=15)


def convert_files(filename_list, res_string):
    for sample_num, filename in enumerate(filename_list):
      if sample_num % 10000 == 0:
        print('sample number: ', sample_num , ' out of ', len(filename_list))
      # if res_string == "topo":
      #   image_id = filename.split('/')[1].split('.')[0][:-8]
      #   num = filename.split('_')[-1][1:-8]
      #   search_string = image_id + '_' + num
      #   index = sample_dict[search_string]
      #   src_ds = gdal.Open(filename)
      #   src_ds_1 = src_ds.ReadAsArray()[0]
      #   src_ds_2 = src_ds.ReadAsArray()[1]
        
      # else:
      # image_id = filename.split('/')[1].split('.')[0][:-4]
      num = filename.split('_')[-1].split('.')[0]
      search_string = num
      index = sample_dict[search_string]
      src_ds = gdal.Open(filename)
      src_ds = src_ds.ReadAsArray()

      

      if index in test_indices:
        # if res_string == "topo":
        #   plt.imsave('test/'+res_string+'/'+str(index)+'_1.png',src_ds_1)
        #   plt.imsave('test/'+res_string+'/'+str(index)+'_2.png',src_ds_2)
        # else:
        plt.imsave('data/test/'+res_string+'/'+str(index)+'.png',src_ds)
        np.save('data/test/'+res_string+'/'+str(index)+'.npy',src_ds)
      elif index in validation_indices:
        # if res_string == "topo":
        #   plt.imsave('validation/'+res_string+'/'+str(index)+'_1.png',src_ds_1)
        #   plt.imsave('validation/'+res_string+'/'+str(index)+'_2.png',src_ds_2)
        # else:
        plt.imsave('data/validation/'+res_string+'/'+str(index)+'.png',src_ds)
        np.save('data/validation/'+res_string+'/'+str(index)+'.npy',src_ds)
      elif index in training_indices:
        # if res_string == "topo":
        #   plt.imsave('training/'+res_string+'/'+str(index)+'_1.png',src_ds_1)
        #   plt.imsave('training/'+res_string+'/'+str(index)+'_2.png',src_ds_2)
        # else:
        plt.imsave('data/training/'+res_string+'/'+str(index)+'.png',src_ds)
        np.save('data/training/'+res_string+'/'+str(index)+'.npy',src_ds)

convert_files(high_res_files, res_string = 'high_res')
convert_files(low_res_files, res_string = 'low_res')
# convert_files(topo_files, res_string = 'topo')