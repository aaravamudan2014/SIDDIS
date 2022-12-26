from dis import dis
import numpy as np
from tokenize import Double
import matplotlib.pyplot as plt
import numpy
from dataset_classes import *
from Interpolation import *
from Downscaler import *
from Distance_Ranking import *
from RDN import *
from RCAN import *
import torch
import sys
from torch.utils.data.dataloader import DataLoader
import numpy as np
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef


def load_pytorch_models(model_name, input_type):
    config = {}
    config['run_dir'] = 'runs/run_final'
    config['model_type'] = model_name
    config['topo_inclusion'] = input_type

    device = f"cuda:0"
    device = torch.device(device if torch.cuda.is_available() else "cpu")


    model_dict = np.load(config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.npy', allow_pickle=True).item()
    if model_name == 'RDN':
      model = RDN(scale_factor=8,
                  num_channels=1,
                  num_features=model_dict['num_features'],
                  growth_rate=model_dict['num_features'],
                  num_blocks=model_dict['num_blocks'],
                  num_layers=model_dict['num_layers'],
                  topo_inclusion=config['topo_inclusion'],
                  dropout_prob_input=model_dict['dropout_prob_input'],
                  dropout_prob_topo_1=model_dict['dropout_prob_topo_1'],
                  dropout_prob_topo_2=model_dict['dropout_prob_topo_2']
                  ).to(device)
    else:
      model = RCAN(scale=8, 
                  num_features=model_dict['num_features'], 
                  num_rg=model_dict['num_rg'], 
                  num_rcab=model_dict['num_rcab'], 
                  reduction=model_dict['reduction'],
                  topo_inclusion=config['topo_inclusion'],
                  dropout_prob_input=model_dict['dropout_prob_input'],
                  dropout_prob_topo_1=model_dict['dropout_prob_topo_1'],
                  dropout_prob_topo_2=model_dict['dropout_prob_topo_2']).to(device)

    model.load_state_dict(torch.load(config['run_dir']+'/best_model'+config['model_type']+config['topo_inclusion']+'.h5',map_location=torch.device('cpu')))
    # model.eval()
    
    return model, device

def generate_neural_network_output(model,device, inputs, topo_1, topo_2):
    inputs = np.reshape(inputs,(1, 1, inputs.shape[0], inputs.shape[1]))
    topo_1 = np.reshape(topo_1,(1, 1, topo_1.shape[0], topo_1.shape[1]))
    topo_2 = np.reshape(topo_2,(1, 1, topo_2.shape[0], topo_2.shape[1]))
    model = model
    inputs = torch.from_numpy(inputs).to(device).float()
    topo_1 = torch.from_numpy(topo_1).to(device).float()
    topo_2 = torch.from_numpy(topo_2).to(device).float()
    
    with torch.no_grad():
        preds = model(inputs,topo_1,topo_2)
      
    output_image_probs = preds.cpu().numpy()[0,0,:,:]

    return output_image_probs

#TODO: currently this uses the raw image files directly, it needs to instead use the h5 files
# need to check if this is the most efficient way of doing things
def plots(ind, dataset):
    """ Creates plots of the method inundation results for the data corresponding to the list of indices
    :param
    ind: 1D array
        Contains test dataset indices for plot visualization
    :return:
    figure: Plots
        Flood inundation plots grid with 8 columns and same number of rows as
    length of array parameter ind.
    """
    ### Initialize Inundation Plot Grid
    # test_index = test_indices[ind]
    # print(test_index)
    batch_size = 1
    dataset_object = EvalDataset('data/'+dataset+'.h5')
    dataloader = DataLoader(dataset=dataset_object, batch_size=batch_size)
    epoch_mcc = AverageMeter()
    epoch_tpr = AverageMeter()
    epoch_acc = AverageMeter()
    
    total_calculated_mcc = 0
    total_calculated_acc = 0
    cf_total = None
    
    gt_labels = []
    preds_probs = []
    updated_ind = ind
    rows = len(dataloader)
    num_columns = 9
    f, axarr = plt.subplots(rows, num_columns, figsize=(30, 30))
    f.set_figheight(20)
    f.set_figwidth(20)


    RDN_model_none, device = load_pytorch_models(model_name = "RDN", input_type='none')
    RDN_model_beggining, device = load_pytorch_models(model_name = "RDN", input_type='beggining')

    RCAN_model_none, device = load_pytorch_models(model_name = "RCAN", input_type='none')
    RCAN_model_beggining, device = load_pytorch_models(model_name = "RCAN", input_type='beggining')
    
    # Remove axis titles for all plots
    for i in range(num_columns):
        for j in range(len(updated_ind)):
            axarr[j, i].set_axis_off()
    # Label Plot Grid Columns
    axarr[0, 0].title.set_text('Ori Low Res')
    axarr[0, 1].title.set_text('Gt High Res')
    axarr[0, 2].title.set_text('Bicubic')
    axarr[0, 3].title.set_text('Geomorphological Model')
    axarr[0, 4].title.set_text('RDN')
    axarr[0, 5].title.set_text('RDN+topo')
    axarr[0, 6].title.set_text('RCAN+topo')
    axarr[0, 7].title.set_text('Vertical')
    axarr[0, 8].title.set_text('Horizontal')
    
    ### Generate Inundatiion Plots for each method and test indice
    for j in range(len(updated_ind)):
        test_index = updated_ind[j]
        input_image = cv2.imread(dataset+'/low_res/' + str(test_index) + '.png', cv2.IMREAD_GRAYSCALE)
        axarr[j, 1].imshow(cv2.imread(dataset+'/high_res/' + str(test_index) + '.png', cv2.IMREAD_GRAYSCALE), cmap='gray')
        input_image = cv2.imread(dataset+'/low_res/' + str(test_index) + '.png')
        #lapSRN_image = LapSRN(input_image) / 255
        #lapSRN_image[lapSRN_image >= 0.5] = 1
        #lapSRN_image[lapSRN_image < 0.5] = 0
        #axarr[j, 2].imshow(lapSRN_image, cmap='gray')
        #axarr[j, 2].imshow(random_plot())
        bicubic_image = inter2high(input_image, cv2.INTER_CUBIC) / 255
        bicubic_image[bicubic_image >= 0.5] = 1
        bicubic_image[bicubic_image < 0.5] = 0
        axarr[j, 2].imshow(bicubic_image, cmap='gray')
        # lanczos_image = inter2high(input_image, cv2.INTER_LANCZOS4) / 255
        # lanczos_image[lanczos_image >= 0.5] = 1
        # lanczos_image[lanczos_image < 0.5] = 0
        # axarr[j, 4].imshow(lanczos_image, cmap='gray')

        #RDN_topo_image = cv2.imread('RDN_topo_results/' + str(test_index) + '.png', cv2.IMREAD_GRAYSCALE)
        #RDN_image = cv2.imread('RDN_results/' + str(test_index) + '.png', cv2.IMREAD_GRAYSCALE)
        # filestring = [k for k, v in sample_zdict.items() if v == test_index][0]
        # file_index = filestring.split('_')[-1]
        # print('results/DS_'+filestring[:-3]+'C'+str(file_index)+'.png')
        # #input()
        # #get results for poor man's downscaling method
        # axarr[j, 5].imshow(cv2.imread('results/DS_' + filestring[:-3] + '_C' + str(file_index) + '.png', cv2.IMREAD_GRAYSCALE),
        #                 cmap='gray')
        #Generate Downscaled Image
        L_npy = np.load(dataset+'/low_res/' + str(test_index) + '.npy')
        H_Data = np.load(dataset+'/topo/' + str(test_index) + '_1.npy')
        V_Data = np.load(dataset+'/topo/' + str(test_index) + '_2.npy')
 
        RDN_none_output = generate_neural_network_output(RDN_model_none, device, L_npy, H_Data, V_Data)
        RDN_beggining_output = generate_neural_network_output(RDN_model_beggining, device, L_npy, H_Data, V_Data)
        RCAN_beggining_output = generate_neural_network_output(RCAN_model_beggining, device, L_npy, H_Data, V_Data)
        
        Poor_man = generate_downscaled_image(L_npy, H_Data, V_Data, distance_ranking,'L1')
        
        axarr[j, 0].imshow(L_npy)
        axarr[j, 3].imshow(Poor_man, cmap='gray')
        axarr[j, 4].imshow(RDN_none_output, cmap='gray')
        axarr[j, 5].imshow(RDN_beggining_output, cmap='gray')
        axarr[j, 6].imshow(RCAN_beggining_output, cmap='gray')
        axarr[j, 7].imshow(V_Data, cmap='gray')
        axarr[j, 8].imshow(H_Data, cmap='gray')

def generate_predictions(dataset):
    dataset_object = EvalDataset('data/'+dataset+'.h5')
    batch_size  = 1
    dataloader = DataLoader(dataset=dataset_object, batch_size=batch_size)
    
    
    RDN_model_none, device = load_pytorch_models(model_name = "RDN", input_type='none')
    RDN_model_vertical, _ = load_pytorch_models(model_name = "RDN", input_type='vertical')
    RDN_model_horizontal, _ = load_pytorch_models(model_name = "RDN", input_type='horizontal')
    RDN_model_beggining, _ = load_pytorch_models(model_name = "RDN", input_type='beggining')

    RCAN_model_none, _ = load_pytorch_models(model_name = "RCAN", input_type='none')
    RCAN_model_vertical, _ = load_pytorch_models(model_name = "RCAN", input_type='vertical')
    RCAN_model_horizontal, _ = load_pytorch_models(model_name = "RCAN", input_type='horizontal')
    RCAN_model_beggining, _ = load_pytorch_models(model_name = "RCAN", input_type='beggining')

    gt_array = []
    bicubic_pred = []
    lanczos_pred = []
    dummy_pred = []
    l1_pred = []
    l2_pred = []
    RDN_none_pred = []
    RDN_vertical_pred = []
    RDN_horizontal_pred = []
    RDN_beggining_pred = []
    RCAN_none_pred = []
    RCAN_vertical_pred = []
    RCAN_horizontal_pred = []
    RCAN_beggining_pred = []

    counted = 0
    for index, data in enumerate(dataloader):
        print("Currently at index: ", index," of ", len(dataloader), end='\r')
        sys.stdout.flush()
        inputs, labels, topo_1, topo_2 = data
        
        L_npy = inputs.cpu()[0,0,:,:].numpy()
        H_Data = topo_1[0,0,:,:].numpy()
        V_Data = topo_2[0,0,:,:].numpy()
        gt_npy = labels[0,0,:,:].numpy()
        
        
        input_rgb = cv2.cvtColor(inputs[0,0,:,:].numpy(),cv2.COLOR_GRAY2RGB)
        bicubic_image_probs = inter2high(input_rgb, cv2.INTER_CUBIC) / 255
        Poor_man_l1 = generate_downscaled_image(L_npy, H_Data, V_Data, distance_ranking,'L1')
        Poor_man_l2 = generate_downscaled_image(L_npy, H_Data, V_Data, distance_ranking,'L2')
        lanczos_image_probs = inter2high(input_rgb, cv2.INTER_LANCZOS4) / 255
       
        counted += 1
        
        
        RDN_none_output_probs = generate_neural_network_output(RDN_model_none, device, L_npy, H_Data, V_Data)
        RDN_vertical_output_probs = generate_neural_network_output(RDN_model_vertical, device, L_npy, H_Data, V_Data)
        RDN_horizontal_output_probs = generate_neural_network_output(RDN_model_horizontal, device, L_npy, H_Data, V_Data)
        RDN_beggining_output_probs = generate_neural_network_output(RDN_model_beggining, device, L_npy, H_Data, V_Data)

        RCAN_none_output_probs = generate_neural_network_output(RCAN_model_none, device, L_npy, H_Data, V_Data)
        RCAN_vertical_output_probs = generate_neural_network_output(RCAN_model_vertical, device, L_npy, H_Data, V_Data)
        RCAN_horizontal_output_probs = generate_neural_network_output(RCAN_model_horizontal, device, L_npy, H_Data, V_Data)
        RCAN_beggining_output_probs = generate_neural_network_output(RCAN_model_beggining, device, L_npy, H_Data, V_Data)
        

        def get_preds(gt_npy, pred, L_npy):
          HR_step = 8
          req_positions = np.argwhere(L_npy > 0)
          updated_pred_array = []
          updated_gt_array = []
          for position in req_positions:
            row_start = position[0] * HR_step
            col_start = position[1] * HR_step
            pred_elems = pred[row_start:row_start + HR_step, col_start: col_start + HR_step]
            gt_elems = gt_npy[row_start:row_start + HR_step, col_start: col_start + HR_step]
            updated_gt_array.extend(gt_elems.flatten())
            updated_pred_array.extend(pred_elems.flatten())

          return updated_gt_array, updated_pred_array
        
        gt, bicubic_preds = get_preds(gt_npy, bicubic_image_probs, L_npy)
        gt, dummy_preds = get_preds(gt_npy, np.zeros_like(Poor_man_l1), L_npy)
        gt, lanczos_preds = get_preds(gt_npy, lanczos_image_probs, L_npy)
        gt, l1_preds = get_preds(gt_npy, Poor_man_l1, L_npy)
        gt, l2_preds = get_preds(gt_npy, Poor_man_l2, L_npy)

        gt, RDN_beggining_preds = get_preds(gt_npy, RDN_beggining_output_probs, L_npy)
        gt, RDN_none_preds = get_preds(gt_npy, RDN_none_output_probs, L_npy)
        gt, RDN_vertical_preds = get_preds(gt_npy, RDN_vertical_output_probs, L_npy)
        gt, RDN_horizontal_preds = get_preds(gt_npy, RDN_horizontal_output_probs, L_npy)

        gt, RCAN_beggining_preds = get_preds(gt_npy, RCAN_beggining_output_probs, L_npy)
        gt, RCAN_none_preds = get_preds(gt_npy, RCAN_none_output_probs, L_npy)
        gt, RCAN_vertical_preds = get_preds(gt_npy, RCAN_vertical_output_probs, L_npy)
        gt, RCAN_horizontal_preds = get_preds(gt_npy, RCAN_horizontal_output_probs, L_npy)
        
        gt_array.extend(gt)
        bicubic_pred.extend(bicubic_preds)
        lanczos_pred.extend(lanczos_preds)
        dummy_pred.extend(dummy_preds)
        l1_pred.extend(l1_preds)
        l2_pred.extend(l2_preds)
        RDN_none_pred.extend(RDN_none_preds)
        RDN_vertical_pred.extend(RDN_vertical_preds)
        RDN_horizontal_pred.extend(RDN_horizontal_preds)
        RDN_beggining_pred.extend(RDN_beggining_preds)
        RCAN_none_pred.extend(RCAN_none_preds)
        RCAN_vertical_pred.extend(RCAN_vertical_preds)
        RCAN_horizontal_pred.extend(RCAN_horizontal_preds)
        RCAN_beggining_pred.extend(RCAN_beggining_preds)
    
    
    print(counted)
    np.save('results/gt_predictions_'+dataset+'.npy', gt_array)
    np.save('results/bicubic_predictions_'+dataset+'.npy', bicubic_pred)
    np.save('results/lanczos_predictions_'+dataset+'.npy', lanczos_pred)
    np.save('results/dummy_predictions_'+dataset+'.npy', dummy_pred)
    np.save('results/l1_predictions_'+dataset+'.npy', l1_pred)
    np.save('results/l2_predictions_'+dataset+'.npy', l2_pred)
    np.save('results/RDN_none_predictions_'+dataset+'.npy', RDN_none_pred)
    np.save('results/RDN_vertical_predictions_'+dataset+'.npy', RDN_vertical_pred)
    np.save('results/RDN_horizontal_predictions_'+dataset+'.npy', RDN_horizontal_pred)
    np.save('results/RDN_beggining_predictions_'+dataset+'.npy', RDN_beggining_pred)
    np.save('results/RCAN_none_predictions_'+dataset+'.npy', RCAN_none_pred)
    np.save('results/RCAN_vertical_predictions_'+dataset+'.npy', RCAN_vertical_pred)
    np.save('results/RCAN_horizontal_predictions_'+dataset+'.npy', RCAN_horizontal_pred)
    np.save('results/RCAN_beggining_predictions_'+dataset+'.npy', RCAN_beggining_pred)

def run_statistical_test():
  
  def create_contingency_table(y_target,y_model1,y_model2):
      tb = mcnemar_table(y_target=y_target, 
                    y_model1=y_model1, 
                    y_model2=y_model2)
      return tb
  

  for i in range(len(arrays)):
      for j in range(i+1, len(arrays)):
          tb = create_contingency_table(gt_predictions, arrays[i],arrays[j])
          chi2, p = mcnemar(ary=tb, corrected=False)
          if p > 0.005:
            print('chi-squared:', chi2)
            print('p-value for '+models[i]+' '+models[j]+':', np.round(p,2))

def generate_roc_curves():
  cf_super_dict = {}
  for i in range(len(arrays)):
    cf_list = []
    for threshold in np.linspace(0,1):
      y_true = gt_predictions
      y_pred = arrays[i].copy()
      y_pred[y_pred >= threshold] = 1
      y_pred[y_pred < threshold] = 0
      cf = confusion_matrix(y_true, y_pred)
      acc = (cf[0,0] + cf[1,1])/np.sum(cf) 
      cf_list.append(cf)
      print(models[i], "Accuracy: ", acc)
    cf_super_dict[models[i]] = cf_list

  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.set_title('ROC curves for ' + dataset)
  ax.set_xlabel('FP rate')
  ax.set_ylabel('TP rate')

  for key, value in enumerate(cf_super_dict):
    cf_list = cf_super_dict[value]
    tpr_list = []
    fpr_list = []
    for cf in cf_list:
      tn, fp, fn, tp = cf.ravel()
      tpr_list.append(tp/(tp+fn))
      fpr_list.append(fp/(fp+tn))
    ax.plot(fpr_list, tpr_list, label=value)

  ax.legend()
  fig.savefig('results/'+dataset+'_roc.png', dpi=200,bbox_inches = "tight")

def generate_statistics():
  for i in range(len(arrays)):
    cf_list = []
    y_true = gt_predictions
    y_pred = arrays[i].copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    cf = confusion_matrix(y_true, y_pred)
    acc = (cf[0,0] + cf[1,1])/np.sum(cf) 
    cf_list.append(cf)
    print(models[i], "Accuracy: ", acc)
    


datasets = ['test', 'test_landsat8_updated', 'EU_external','RR_Trimmed_External' ]

for dataset in datasets:
  generate_predictions(dataset)
  filename_qualifier = 'results/'



  gt_predictions = np.load(filename_qualifier+'gt_predictions_'+dataset+'.npy')

  RDN_beggining_predictions = np.load(filename_qualifier+'RDN_beggining_predictions_'+dataset+'.npy')
  RDN_none_predictions = np.load(filename_qualifier+'RDN_none_predictions_'+dataset+'.npy')
  RDN_horizontal_predictions = np.load(filename_qualifier+'RDN_horizontal_predictions_'+dataset+'.npy')
  RDN_vertical_predictions = np.load(filename_qualifier+'RDN_vertical_predictions_'+dataset+'.npy')
  RCAN_beggining_predictions = np.load(filename_qualifier+'RCAN_beggining_predictions_'+dataset+'.npy')
  RCAN_none_predictions = np.load(filename_qualifier+'RCAN_none_predictions_'+dataset+'.npy')
  RCAN_horizontal_predictions = np.load(filename_qualifier+'RCAN_horizontal_predictions_'+dataset+'.npy')
  RCAN_vertical_predictions = np.load(filename_qualifier+'RCAN_vertical_predictions_'+dataset+'.npy')

  bicubic_predictions = np.load(filename_qualifier+'bicubic_predictions_'+dataset+'.npy')
  lanczos_predictions = np.load(filename_qualifier+'lanczos_predictions_'+dataset+'.npy')
  dummy_predictions = np.load(filename_qualifier+'dummy_predictions_'+dataset+'.npy')
  l1_predictions = np.load(filename_qualifier+'l1_predictions_'+dataset+'.npy')
  l2_predictions = np.load(filename_qualifier+'l2_predictions_'+dataset+'.npy')

  arrays = [RCAN_beggining_predictions, RCAN_none_predictions, RCAN_horizontal_predictions, RCAN_vertical_predictions,
            RDN_beggining_predictions, RDN_none_predictions, RDN_horizontal_predictions, RDN_vertical_predictions, 
            bicubic_predictions, lanczos_predictions, dummy_predictions, l1_predictions, l2_predictions]

  models = ['RCAN_all', 'RCAN_none','RCAN_horizontal','RCAN_vertical','RDN_all', 'RDN_none','RDN_horizontal',
            'RDN_vertical','bicubic','lanczos','dummy','p_l1','p_l2']
  


  run_statistical_test()
  generate_statistics()

