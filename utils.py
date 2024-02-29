import torch
import numpy as np
from dataset_classes import *
from torch.utils.data.dataloader import DataLoader
from RDN import *
from loss import *
from dataset_classes import *
from utils import *
from tqdm import tqdm
import sys

# Pixel steps to be taken when converting from low resolution to high resolution image
HR_step = 8


# Peak signal to noise ratio
# the maximum value is 1, which corresponds to 255
def calc_psnr(img1, img2, max=1.0):
    return 10. * ((max ** 2) / ((img1 - img2) ** 2).mean()).log10()

def calc_bce(output,target):
    ret_loss = torch.zeros(1, 1, requires_grad=False).to(device)
    ret_loss = ret_loss - torch.mean((torch.mul(target, torch.log(output)) + torch.mul(1-target, torch.log(1-output))))
    
    return ret_loss
     
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

from sklearn.metrics import confusion_matrix



def matthews_correlation_coefficient(pred, ground_truth, low_res_image, threshold = 0.5):
    if np.amax(pred) > 1:
      pred = pred/255
    pred[pred>=threshold] = 1
    pred[pred<threshold] = 0

    req_positions = np.argwhere(low_res_image > 0)

    updated_pred_array = []
    updated_gt_array = []
    for position in req_positions:
      row_start = position[0] * HR_step
      col_start = position[1] * HR_step
      pred_elems = pred[row_start:row_start + HR_step, col_start: col_start + HR_step]
      gt_elems = ground_truth[row_start:row_start + HR_step, col_start: col_start + HR_step]
      
      updated_pred_array.extend(pred_elems.flatten())
      updated_gt_array.extend(gt_elems.flatten())
    if len(updated_gt_array) > 0:
        
        cf = confusion_matrix(updated_gt_array, updated_pred_array).ravel()
        if len(cf) ==4:
          TN, FP, FN, TP = cf
          if TN +FP == 0 or TN +FN == 0 or TN +TP == 0 or FP +FN == 0 or FP +TP == 0 or FN +TP == 0:
            MCC = 0.0
          else:
            MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
        else:
          MCC = np.nan
          cf = [np.nan]
    else:
      MCC = np.nan
      cf = [np.nan]
    return  MCC,cf

def accuracy(pred, ground_truth,low_res_image):
    if np.amax(pred) > 1:
      pred = pred/255
    pred[pred>=0.5] = 1
    pred[pred<0.5] = 0

    req_positions = np.argwhere(low_res_image > 0)

    updated_pred_array = []
    updated_gt_array = []
    for position in req_positions:
      row_start = position[0] * HR_step
      col_start = position[1] * HR_step
      pred_elems = pred[row_start:row_start + HR_step, col_start: col_start + HR_step]
      gt_elems = ground_truth[row_start:row_start + HR_step, col_start: col_start + HR_step]
      
      updated_pred_array.extend(pred_elems.flatten())
      updated_gt_array.extend(gt_elems.flatten())


    num_equal = 0
    for x,y in zip(updated_pred_array,updated_gt_array):
      if x == y:
        num_equal += 1

    if len(updated_pred_array)>0:
      acc = num_equal/len(updated_pred_array)
      return acc
    else:
      return np.nan



def gen_test_results(model, device, dataset):
  batch_size = 1
  dataset_object = EvalDataset('data/SYN-processed/'+dataset+'.h5')
  print("Total size of ", dataset, " ", len(dataset_object))
  dataloader = DataLoader(dataset=dataset_object, batch_size=batch_size)
  epoch_mcc = AverageMeter()
  epoch_tpr = AverageMeter()
  epoch_acc = AverageMeter()
  
  total_calculated_mcc = 0
  total_calculated_acc = 0
  cf_total = None
  
  gt_labels = []
  preds_probs = []
  for index, data in enumerate(dataloader):
      print("Currently at batch: ", index, end='\r')
      inputs, labels = data

      inputs = inputs.to(device)
      labels = labels.to(device)
      # topo_1 = topo_1.to(device)
      # topo_2 = topo_2.to(device)

      with torch.no_grad():
          preds = model(inputs)

      loss, cf = matthews_correlation_coefficient(preds.cpu()[0,0,:,:].numpy(), labels.cpu()[0,0,:,:].numpy(), inputs.cpu()[0,0,:,:].numpy())
      acc = accuracy(preds.cpu()[0,0,:,:].numpy(), labels.cpu()[0,0,:,:].numpy(), inputs.cpu()[0,0,:,:].numpy())
      
      gt_labels.extend(labels.cpu()[0,0,:,:].numpy())
      
      if cf_total is None:
        if len(cf) == 4:
          cf_total = cf
      else:
        if len(cf) == 4:
          cf_total += cf
      if not np.isnan(loss):
        total_calculated_mcc += 1
        epoch_mcc.update(loss, len(inputs))
        
      if not np.isnan(acc):
        total_calculated_acc += 1
        epoch_acc.update(acc, len(inputs))
      sys.stdout.flush()

  
  # print(dataset+' mcc: {:.4f}'.format(epoch_mcc.avg))
  # print(dataset+' acc: {:.4f}'.format(epoch_acc.avg))

  TN, FP, FN, TP = cf_total
  if TN +FP == 0 or TN +FN == 0 or TN +TP == 0 or FP +FN == 0 or FP +TP == 0 or FN +TP == 0:
    MCC = 0.0
  else:
    #MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    numerator = np.log(TP*TN-FP*FN) 
    denominator = 0.5*(np.log(TP+FP) + np.log(TP+FN) + np.log(TN+FN) + np.log(TN+FP))
    MCC = np.exp(numerator - denominator)
  
  ACC = (TP+TN)/(TP+TN+FP+FN)
  
  print('test mcc: {:.4f}'.format(MCC))
  print('test acc: {:.4f}'.format(ACC))
  print('total calculated : ', total_calculated_mcc, " out of: ", len(dataloader)*batch_size)

  
  sys.stdout.flush()
  return MCC, ACC , total_calculated_mcc, total_calculated_acc

def gen_Landsat8_results(model, device, dataset_type='Landsat8'):
  batch_size  = 1
  dataset = EvalDataset('data/RW-processed/'+dataset_type+'.h5')
  
  landsat_dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
  epoch_mcc = AverageMeter()
  epoch_tpr = AverageMeter()
  epoch_acc = AverageMeter()
  
  total_calculated_mcc = 0
  total_calculated_acc = 0
  cf_total = None

  for index, data in enumerate(landsat_dataloader):
      print("Currently at index: ", index, end='\r')
      inputs, labels = data

      inputs = inputs.to(device)
      labels = labels.to(device)
      
      # topo_1 = topo_1.to(device)
      # topo_2 = topo_2.to(device)
      
      with torch.no_grad():
          preds = model(inputs)
      
      loss, cf = matthews_correlation_coefficient(preds.cpu()[0,0,:,:].numpy(), labels.cpu()[0,0,:,:].numpy(), inputs.cpu()[0,0,:,:].numpy())
      if cf_total is None:
        if len(cf) == 4:
          cf_total = cf
      else:
        if len(cf) == 4:
          cf_total += cf
      
      acc = accuracy(preds.cpu()[0,0,:,:].numpy(), labels.cpu()[0,0,:,:].numpy(), inputs.cpu()[0,0,:,:].numpy())
      
      # print(loss)
      if not np.isnan(loss):
        total_calculated_mcc += 1
        epoch_mcc.update(loss, len(inputs))
      if not np.isnan(acc):
        total_calculated_acc += 1
        epoch_acc.update(acc, len(inputs))
        sys.stdout.flush()
  
  TN, FP, FN, TP = cf_total
  if TN +FP == 0 or TN +FN == 0 or TN +TP == 0 or FP +FN == 0 or FP +TP == 0 or FN +TP == 0:
    MCC = 0.0
  else:
    #MCC = (TP*TN-FP*FN)/(np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)))
    numerator = np.log(TP*TN-FP*FN) 
    denominator = 0.5*(np.log(TP+FP) + np.log(TP+FN) + np.log(TN+FN) + np.log(TN+FP))
    MCC = np.exp(numerator - denominator)
  
  ACC = (TP+TN)/(TP+TN+FP+FN)
    
  print(dataset_type + ' mcc: {:.4f}'.format(MCC))
  print(dataset_type + ' acc: {:.4f}'.format(ACC))
  print('total calculated : ', total_calculated_acc, " out of: ", len(landsat_dataloader))
  
  
  sys.stdout.flush()
  return MCC, ACC