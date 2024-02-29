import numpy as np
import h5py
import glob

# create h5 files for each of training, test and validation
def create_dataset(dataset_type, location):
  h5_file = h5py.File(location+str(dataset_type)+'.h5', 'w')

  lr_group = h5_file.create_group('lr')
  hr_group = h5_file.create_group('hr')
  topo_group_1 = h5_file.create_group('topo_1')
  topo_group_2 = h5_file.create_group('topo_2')

  req_indices = glob.glob(location+str(dataset_type)+'/high_res/*')

  req_indices = [x.split('.')[-2].split('/')[-1] for x in req_indices]
  patch_idx = 0
  for image_index in req_indices:
    hr = np.load(location+str(dataset_type)+'/high_res/'+str(image_index)+'.npy')
    hr = np.reshape(hr, (hr.shape[0],hr.shape[1],1))
    lr = np.load(location+str(dataset_type)+'/low_res/'+str(image_index)+'.npy')
    lr = np.reshape(lr, (lr.shape[0],lr.shape[1],1))
    if np.sum(lr) == 0:
      print("ignoring index ", image_index, " in ", dataset_type)
      continue

    lr_group.create_dataset(str(patch_idx), data=lr)
    hr_group.create_dataset(str(patch_idx), data=hr)
  
    patch_idx += 1
  
  print("Dataset: ", dataset_type, " has been created")

create_dataset(dataset_type = 'EU', location='data/RW-processed/')
create_dataset(dataset_type = 'training', location='data/SYN-processed/')
create_dataset(dataset_type = 'validation', location='data/SYN-processed/')
create_dataset(dataset_type = 'test', location='data/SYN-processed/')

