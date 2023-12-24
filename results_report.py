
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from dataset_classes import *
from Interpolation import *
from Downscaler import *
from Distance_Ranking import *
from RDN import *
from RCAN import *
from ESRT import esrt, utils
import torch
import sys
from torch.utils.data.dataloader import DataLoader
import numpy as np
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef
from toh import *
import pandas as pd

# Some global settings for figure sizes
normalFigSize = (8, 6) # (width,height) in inches
largeFigSize = (12, 9)
xlargeFigSize = (16, 12)

    

def load_pytorch_models(model_name, input_type):
    config = {}
    config['run_dir'] = 'runs/run_final'
    config['model_type'] = model_name
    config['topo_inclusion'] = input_type

    device = f"cuda:0"
    device = torch.device(device if torch.cuda.is_available() else "cpu")


    model_dict = np.load(config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.npy', allow_pickle=True).item()
    if model_name == 'RDN':
      model = RDN(scale_factor=10,
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
    elif model_name == 'RCAN':
      model = RCAN(scale=10, 
                  num_features=model_dict['num_features'], 
                  num_rg=model_dict['num_rg'], 
                  num_rcab=model_dict['num_rcab'], 
                  reduction=model_dict['reduction'],
                  topo_inclusion=config['topo_inclusion'],
                  dropout_prob_input=model_dict['dropout_prob_input'],
                  dropout_prob_topo_1=model_dict['dropout_prob_topo_1'],
                  dropout_prob_topo_2=model_dict['dropout_prob_topo_2']).to(device)
    elif model_name == 'ESRT':
        model = esrt.ESRT(topo_inclusion=config['topo_inclusion'],
                          num_features = model_dict['num_features'], 
                          upscale=10,
                          dropout_prob_input=model_dict['dropout_prob_input'],
                          dropout_prob_topo_1=model_dict['dropout_prob_topo_1'],
                          dropout_prob_topo_2=model_dict['dropout_prob_topo_2']).to(device) 
   
    model.load_state_dict(torch.load(config['run_dir']+'/best_model'+config['model_type']+config['topo_inclusion']+'.h5',map_location=torch.device('cpu')))
    
    return model, device

def generate_neural_network_output(model,device, inputs, topo_1=None, topo_2=None):
    inputs = np.reshape(inputs,(1, 1, inputs.shape[0], inputs.shape[1]))
    # topo_1 = np.reshape(topo_1,(1, 1, topo_1.shape[0], topo_1.shape[1]))
    # topo_2 = np.reshape(topo_2,(1, 1, topo_2.shape[0], topo_2.shape[1]))
    model = model
    inputs = torch.from_numpy(inputs).to(device).float()
    # topo_1 = torch.from_numpy(topo_1).to(device).float()
    # topo_2 = torch.from_numpy(topo_2).to(device).float()
    
    with torch.no_grad():
        preds = model(inputs)
      
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
    # print(len(dataloader))
    # input()
    RDN_model_none, device = load_pytorch_models(model_name = "RDN", input_type='none')
    # RDN_model_beggining, _ = load_pytorch_models(model_name = "RDN", input_type='beggining')

    RCAN_model_none, _ = load_pytorch_models(model_name = "RCAN", input_type='none')
    # RCAN_model_beggining, _ = load_pytorch_models(model_name = "RCAN", input_type='beggining')

    ESRT_model_none, _ = load_pytorch_models(model_name = "ESRT", input_type='none')
    # ESRT_model_beggining, _ = load_pytorch_models(model_name = "ESRT", input_type='beggining')

    gt_array = []
    bicubic_pred = []
    lanczos_pred = []
    dummy_pred = []
    l1_pred = []
    l2_pred = []
    RDN_none_pred = []
    RDN_beggining_pred = []
    RCAN_none_pred = []
    RCAN_beggining_pred = []
    ESRT_none_pred = []
    ESRT_beggining_pred = []

    counted = 0

    constraint_violations_bicubic = 0
    constraint_violations_lanczos= 0
    constraint_violations_dummy= 0
    constraint_violations_RDN= 0
    constraint_violations_RCAN= 0
    constraint_violations_ESRT= 0
    for index, data in enumerate(dataloader):
        print("Currently at index: ", index," of ", len(dataloader))
        sys.stdout.flush()
        inputs, labels = data
        
        L_npy = inputs.cpu()[0,0,:,:].numpy()
        if np.sum(L_npy) == 0:
           continue
        counted += 1

        # H_Data = topo_1[0,0,:,:].numpy()
        # V_Data = topo_2[0,0,:,:].numpy()
        gt_npy = labels[0,0,:,:].numpy()

        
        # input_rgb = cv2.cvtColor(inputs[0,0,:,:].numpy(),cv2.COLOR_GRAY2RGB)
        
        uint_img = np.array(inputs[0,0,:,:].numpy()*255).astype('uint8')

        grayImage = cv2.cvtColor(uint_img, cv2.COLOR_GRAY2BGR)
        rgbImage = cv2.cvtColor(uint_img,cv2.COLOR_GRAY2RGB)
        bicubic_image_probs = inter2high(grayImage, cv2.INTER_CUBIC) / 255

        # _min = np.amin(bicubic_image_probs)
        # _max = np.amax(bicubic_image_probs)
        # bicubic_image_probs = (bicubic_image_probs - _min) / (_max - _min)
        

        # Poor_man_l1 = generate_downscaled_image(L_npy, H_Data, V_Data, distance_ranking,'L1')
        # Poor_man_l2 = generate_downscaled_image(L_npy, H_Data, V_Data, distance_ranking,'L2')
        lanczos_image_probs = inter2high(grayImage, cv2.INTER_LANCZOS4) /255
        
        # _min = np.amin(lanczos_image_probs)
        # _max = np.amax(lanczos_image_probs)
        # lanczos_image_probs = (lanczos_image_probs - _min) / (_max - _min)

        
        
        RDN_none_output_probs = generate_neural_network_output(RDN_model_none, device, L_npy)
        # RDN_beggining_output_probs = generate_neural_network_output(RDN_model_beggining, device, L_npy, H_Data, V_Data)

        RCAN_none_output_probs = generate_neural_network_output(RCAN_model_none, device, L_npy)
        # RCAN_beggining_output_probs = generate_neural_network_output(RCAN_model_beggining, device, L_npy, H_Data, V_Data)
        
        ESRT_none_output_probs = generate_neural_network_output(ESRT_model_none, device, L_npy)
        # ESRT_beggining_output_probs = generate_neural_network_output(ESRT_model_beggining, device, L_npy, H_Data, V_Data)
        
        def get_preds(gt_npy, pred, L_npy):
          HR_step = 10
          req_positions = np.argwhere((L_npy > 0.25) & (L_npy < 0.85))
          updated_pred_array = []
          updated_gt_array = []
          constraint_violations = 0
          for position in req_positions:
            row_start = position[0] * HR_step
            col_start = position[1] * HR_step
            pred_elems = pred[row_start:row_start + HR_step, col_start: col_start + HR_step]
            gt_elems = gt_npy[row_start:row_start + HR_step, col_start: col_start + HR_step]

            low_res_fraction_value = L_npy[position[0],position[1]]
            fraction_inundated = np.sum(pred_elems)/100
            constraint_violations += np.abs(fraction_inundated-low_res_fraction_value)

            updated_gt_array.extend(gt_elems.flatten())
            updated_pred_array.extend(pred_elems.flatten())
            
          return updated_gt_array, updated_pred_array,constraint_violations
        
        gt, bicubic_preds,cv_bicubic = get_preds(gt_npy, bicubic_image_probs, L_npy)
        gt, dummy_preds,cv_dummy = get_preds(gt_npy, np.zeros_like(bicubic_image_probs), L_npy)
        gt, lanczos_preds, cv_lanczos = get_preds(gt_npy, lanczos_image_probs, L_npy)
        
        # gt, l1_preds = get_preds(gt_npy, Poor_man_l1, L_npy)
        # gt, l2_preds = get_preds(gt_npy, Poor_man_l2, L_npy)

        # gt, RDN_beggining_preds = get_preds(gt_npy, RDN_beggining_output_probs, L_npy)
        gt, RDN_none_preds,cv_RDN = get_preds(gt_npy, RDN_none_output_probs, L_npy)

        
        # gt, RCAN_beggining_preds = get_preds(gt_npy, RCAN_beggining_output_probs, L_npy)
        gt, RCAN_none_preds,cv_RCAN = get_preds(gt_npy, RCAN_none_output_probs, L_npy)
        
        # gt, ESRT_beggining_preds = get_preds(gt_npy, ESRT_beggining_output_probs, L_npy)
        gt, ESRT_none_preds, cv_ESRT = get_preds(gt_npy, ESRT_none_output_probs, L_npy)
        


        constraint_violations_bicubic +=cv_bicubic
        constraint_violations_lanczos +=cv_lanczos
        constraint_violations_dummy +=cv_dummy
        constraint_violations_RDN +=cv_RDN
        constraint_violations_RCAN +=cv_RCAN
        constraint_violations_ESRT +=cv_ESRT

        gt_array.extend(gt)
        bicubic_pred.extend(bicubic_preds)
        lanczos_pred.extend(lanczos_preds)
        dummy_pred.extend(dummy_preds)
        # l1_pred.extend(l1_preds)
        # l2_pred.extend(l2_preds)
        RDN_none_pred.extend(RDN_none_preds)
        # RDN_beggining_pred.extend(RDN_beggining_preds)
        RCAN_none_pred.extend(RCAN_none_preds)
        # RCAN_beggining_pred.extend(RCAN_beggining_preds)
        ESRT_none_pred.extend(ESRT_none_preds)
        # ESRT_beggining_pred.extend(ESRT_beggining_preds)
    
    
    # constraint_violations_df = pd.DataFrame(columns=['model', 'dataset', 'violation_metric'])
    constraint_violations_df = pd.read_csv('results/constraint_violations.csv')
    
    constraint_violations_df = constraint_violations_df.append({'model':'bicubic', 'dataset':dataset, 'violation_metric':constraint_violations_bicubic}, ignore_index=True)
    constraint_violations_df = constraint_violations_df.append({'model':'lanczos', 'dataset':dataset, 'violation_metric':constraint_violations_lanczos}, ignore_index=True)
    constraint_violations_df = constraint_violations_df.append({'model':'dummy', 'dataset':dataset, 'violation_metric':constraint_violations_dummy}, ignore_index=True)
    constraint_violations_df = constraint_violations_df.append({'model':'RDN', 'dataset':dataset, 'violation_metric':constraint_violations_RDN}, ignore_index=True)
    constraint_violations_df = constraint_violations_df.append({'model':'RCAN', 'dataset':dataset, 'violation_metric':constraint_violations_RCAN}, ignore_index=True)
    constraint_violations_df = constraint_violations_df.append({'model':'ESRT', 'dataset':dataset, 'violation_metric':constraint_violations_ESRT}, ignore_index=True)
   
    constraint_violations_df.to_csv('results/constraint_violations.csv', index=False)
    print(counted)
    print("bicubic violations", constraint_violations_bicubic)
    print("lanczos violations", constraint_violations_lanczos)
    print("dummy violations", constraint_violations_dummy)
    print("RDN violations", constraint_violations_RDN)
    print("RCAN violations", constraint_violations_RCAN)
    print("ESRT violations", constraint_violations_ESRT)

    np.save('results/gt_predictions_'+dataset+'.npy', gt_array)
    np.save('results/bicubic_predictions_'+dataset+'.npy', bicubic_pred)
    np.save('results/lanczos_predictions_'+dataset+'.npy', lanczos_pred)
    np.save('results/dummy_predictions_'+dataset+'.npy', dummy_pred)
    # np.save('results/l1_predictions_'+dataset+'.npy', l1_pred)
    # np.save('results/l2_predictions_'+dataset+'.npy', l2_pred)
    np.save('results/RDN_none_predictions_'+dataset+'.npy', RDN_none_pred)
    # np.save('results/RDN_beggining_predictions_'+dataset+'.npy', RDN_beggining_pred)
    np.save('results/RCAN_none_predictions_'+dataset+'.npy', RCAN_none_pred)
    # np.save('results/RCAN_beggining_predictions_'+dataset+'.npy', RCAN_beggining_pred)
    np.save('results/ESRT_none_predictions_'+dataset+'.npy', ESRT_none_pred)
    # np.save('results/ESRT_beggining_predictions_'+dataset+'.npy', ESRT_beggining_pred)

def run_statistical_test(arrays,model_names, gt_predictions,dataset):
  MatchedTuples = None

  for model_index in range(len(model_names)):
    error_vector = get_matched_tuples(arrays[model_index], gt_predictions, model_names[model_index])
    if MatchedTuples is None:
        MatchedTuples = error_vector
    else:
        MatchedTuples = np.concatenate((MatchedTuples,error_vector), axis=1)
    
  # Clopper-Pearson CIs #####################################################
  p = 0.999   # nominal coverage probability of CIs
  N, M = MatchedTuples.shape
  sorted_MatchedTuples, sorted_model_names = sort_matchedtuples_models(MatchedTuples, model_names)

  num_correct_decisions = np.sum(sorted_MatchedTuples, axis=0)
  estimated_accuracies = num_correct_decisions / N

  lb_array = np.empty(M, dtype=float)
  ub_array = np.empty(M, dtype=float)
  print("Estimated Accuracies & {}%-level Clopper-Pearson Intervals".format(100.0 * p))
  print()
  print("Model Name\tLB\tACC\tUB")
  for m in range(M):
      lb, ub = ClopperPearsonCI(num_correct_decisions[m], N, p)
      lb_array[m] = lb
      ub_array[m] = ub
      print("{}\t\t{:0.03f}\t{:0.03f}\t{:0.03f}".format(sorted_model_names[m], lb, estimated_accuracies[m], ub))

  # Create a plot
  fig = plt.figure(figsize=xlargeFigSize)
  ax = fig.add_subplot(1, 1, 1)

  fontsize = 16
  labelsize = 14
  ax.set_ylim([0.0, 1.0])
  ax.tick_params(axis='x', labelsize=labelsize)
  ax.tick_params(axis='y', labelsize=labelsize)
  ax.set_yticks(np.linspace(0.0, 1.0, 21))
  ax.grid(axis='y')
  plt.setp(ax.get_xticklabels(), rotation=90)

  # Stacked bar chart
  ax.bar(sorted_model_names, lb_array, color='white',alpha=0.0)
  ax.bar(sorted_model_names, estimated_accuracies-lb_array,bottom =lb_array , color='blue')
  ax.bar(sorted_model_names, ub_array-estimated_accuracies,bottom =estimated_accuracies , color='orange')

  ax.set_title("{}%-level Clopper-Pearson confidence intervals".format(100.0 * p), fontsize=14)
  ax.set_xlabel("", fontsize=fontsize)
  ax.set_ylabel("Estimated Accuracy", fontsize=fontsize)

  fig.savefig('results/accuracies_cis_'+dataset+'.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)


  # Compute Adjusted p-Values via the HB procedure ##########################
  mode=''   #'best_only'
  adj_pValues, ModelPairIndeces = HolmBonferroniProcedure(sorted_MatchedTuples, ExactMcNemarsTest, mode)

  max_FWER = 0.01

  significant_indices = np.where(adj_pValues <= max_FWER)[0]
  ModelPairIndeces_significant = ModelPairIndeces[significant_indices, :]
  print('Significant differences:')
  for pair in range(ModelPairIndeces_significant.shape[0]):
      m1 = ModelPairIndeces_significant[pair, 0]
      m2 = ModelPairIndeces_significant[pair, 1]
      print("{} vs {}    log10(adj-p-value)={:.03f}".format(sorted_model_names[m1], sorted_model_names[m2], np.log10(adj_pValues[significant_indices[pair]])))
  print()
  num_total_comparisons = len(adj_pValues)
  num_significant_comparisons = ModelPairIndeces_significant.shape[0]
  percent_significant_comparisons = 100.0 * num_significant_comparisons / num_total_comparisons
  print("Out of a total of {} comparisons, {} were found significant ({:.02f}%) for an FWER not exceeding {}.".format(num_total_comparisons, num_significant_comparisons, percent_significant_comparisons, max_FWER))

  # Create & plot Significance Matrix #######################################
  AdjPvalueMatrix = mk_adj_pvalue_matrix(adj_pValues, ModelPairIndeces, mode)

  max_FWER_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]

  SignificanceMatrix = np.zeros_like(AdjPvalueMatrix, dtype=int)
  num_max_FWER_values = len(max_FWER_values)

  for level in range(num_max_FWER_values):
      max_FWER = max_FWER_values[level]
      idx = np.where(AdjPvalueMatrix <= max_FWER)
      SignificanceMatrix[idx] = level + 1

  # Plot Significance Matrix
  fig = plt.figure(figsize=xlargeFigSize)
  ax = fig.add_subplot(1, 1, 1)

  fontsize = 16
  labelsize = 14

  # Customize colormap
  cmap = plt.get_cmap(name='jet', lut=num_max_FWER_values+1)
  bounds =  np.arange(-1.0, num_max_FWER_values+1)+0.5
  norm = colors.BoundaryNorm(bounds, cmap.N)

  # Customize axes
  ax.tick_params(axis='x', labelsize=labelsize)
  ax.tick_params(axis='y', labelsize=labelsize)
  ax.set_xticks(np.arange(M))
  ax.set_xticklabels(sorted_model_names)
  plt.setp(ax.get_xticklabels(), rotation=90)
  ax.set_yticks(np.arange(M))
  ax.set_yticklabels(sorted_model_names)
  ax.set_title("Significant Model Differences", fontsize=fontsize)

  # Plot heatmap
  heatmap = ax.imshow(SignificanceMatrix, cmap=cmap, norm=norm, interpolation='nearest') 

  # Add a vertical colorbar
  cbar = plt.colorbar(heatmap, ticks=range(num_max_FWER_values+1))
  cbar_ticklabels = ['FWER>{}'.format(max_FWER_values[0])]
  for level in range(num_max_FWER_values):
      max_FWER = max_FWER_values[level]
      cbar_ticklabels.append('FWER<={}'.format(max_FWER))
  cbar.ax.set_yticklabels(cbar_ticklabels) 
  cbar.ax.tick_params(labelsize=labelsize)

  # Turn spines off
  for key, spine in ax.spines.items():
      spine.set_visible(False)

  # Add a white grid    
  ax.set_xticks(np.arange(M+1)-.5, minor=True)
  ax.set_yticks(np.arange(M+1)-.5, minor=True)
  ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
  ax.tick_params(which="minor", bottom=False, left=False)
  fig.savefig('results/significant_model_differences'+dataset+'.png', dpi=300, transparent=True, bbox_inches='tight', pad_inches=0)
  print("Done with statistical test")

def generate_roc_curves(arrays,gt_predictions,models,dataset):
  cf_super_dict = {}
  center_threshold_cfs = {}

  for i in range(len(arrays)):
    cf_list = []
    y_true = gt_predictions
    for threshold in np.linspace(0,1,20):
      print(models[i], " ", threshold, " ", dataset)
      y_pred = arrays[i].copy()
      y_pred[y_pred >= threshold] = 1
      y_pred[y_pred < threshold] = 0
      cf = confusion_matrix(y_true, y_pred)
      cf_list.append(cf)
      cf_super_dict[models[i]] = cf_list

  for i in range(len(arrays)):
    y_true = gt_predictions
    threshold = 0.5
    y_pred = arrays[i].copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0
    cf = confusion_matrix(y_true, y_pred)
    center_threshold_cfs[models[i]] = cf

  if dataset == 'Landsat8':
    modified_dataset_name = 'RW Iowa'
  elif dataset == 'test':
    modified_dataset_name = 'SYN Iowa'
  elif dataset == 'EU_external':
    modified_dataset_name = 'RW Europe'
  elif dataset == 'RR_Trimmed_External':
    modified_dataset_name = 'RW Red River'
  else:
    modified_dataset_name = dataset
  
  fig = plt.figure()
  ax = fig.add_axes([0,0,1,1])
  ax.set_title('ROC curves for ' + modified_dataset_name)
  ax.set_xlabel('False Positive rate')
  ax.set_ylabel('True Positive rate')
  ax.set_xlim([0,1])
  ax.set_ylim([0,1])
  ax.set_aspect('equal', 'box')
  ax.set_title('ROC curves for ' + dataset)
  
  for key, value in enumerate(cf_super_dict):
    cf_list = cf_super_dict[value]
    tpr_list = []
    fpr_list = []
    for cf in cf_list:
      tn, fp, fn, tp = cf.ravel()
      tpr_list.append(tp/(tp+fn))
      fpr_list.append(fp/(fp+tn))
    ax.plot(fpr_list, tpr_list, label=value)
    center_threshold_cf = center_threshold_cfs[value]
    tn, fp, fn, tp = center_threshold_cf.ravel()
    tpr_center = tp/(tp+fn)
    fpr_center = fp/(fp+tn)
    ax.scatter(fpr_center, tpr_center,s=280, marker=(5, 1))
    print("Accuracy for threshold = 0.5 for model ", value,(tp+tn)/(fp+fn + tp+tn))

  ax.legend()
  fig.savefig('results/'+dataset+'_roc.png', dpi=200,bbox_inches = "tight")

def generate_statistics(arrays,gt_predictions, models, dataset):
  
  # dataset_df = pd.DataFrame(columns=['model', 'dataset', 'acc', 'mcc'])
  dataset_df = pd.read_csv('results/results.csv')
  print( "Number of samples for "+dataset+": ", len(gt_predictions))
  for i in range(len(arrays)):
    y_true = gt_predictions
    y_pred = arrays[i].copy()
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    cf = confusion_matrix(y_true, y_pred)
    acc = (cf[0,0] + cf[1,1])/np.sum(cf) 
    try:
      TN, FP, FN, TP = cf.ravel()
      numerator = np.log(TP*TN-FP*FN) 
      denominator = 0.5*(np.log(TP+FP) + np.log(TP+FN) + np.log(TN+FN) + np.log(TN+FP))
      MCC = np.exp(numerator - denominator)
    except:
      MCC = 0
    print(models[i], "Accuracy: ", acc)
    print(models[i], "MCC: ", MCC)
    dataset_df = dataset_df.append({'model': models[i], 'dataset':dataset, 'acc':acc, 'mcc': MCC}, ignore_index=True)

    sys.stdout.flush()
  dataset_df.to_csv('results/results.csv', index=False)
datasets = ['test','Ghana','IowaCedar','IowaDesMoines','RedRiver']
# datasets = ['EU']


def get_matched_tuples(predictions, gt_predictions, model_name):
    error_vector = np.zeros(len(predictions))
    print(model_name)
    if model_name.split('_')[0] in ['RCAN', 'RDN', 'ESRT', 'bicubic', 'lanczos']:
      predictions[predictions >= 0.5] = 1
      predictions[predictions < 0.5] = 0
    for i in range(len(predictions)):
        if predictions[i]==gt_predictions[i]:
            error_vector[i] = 1
        
    error_vector = np.reshape(error_vector, (len(error_vector),1))
    return error_vector

def main():
  for dataset in datasets:
    generate_predictions(dataset)
    filename_qualifier = 'results/'
    gt_predictions = np.load(filename_qualifier+'gt_predictions_'+dataset+'.npy')

    # RDN_beggining_predictions = np.load(filename_qualifier+'RDN_beggining_predictions_'+dataset+'.npy')
    RDN_none_predictions = np.load(filename_qualifier+'RDN_none_predictions_'+dataset+'.npy')
    # RDN_horizontal_predictions = np.load(filename_qualifier+'RDN_horizontal_predictions_'+dataset+'.npy')
    # RDN_vertical_predictions = np.load(filename_qualifier+'RDN_vertical_predictions_'+dataset+'.npy')
    
    # RCAN_beggining_predictions = np.load(filename_qualifier+'RCAN_beggining_predictions_'+dataset+'.npy')
    RCAN_none_predictions = np.load(filename_qualifier+'RCAN_none_predictions_'+dataset+'.npy')
    # RCAN_horizontal_predictions = np.load(filename_qualifier+'RCAN_horizontal_predictions_'+dataset+'.npy')
    # RCAN_vertical_predictions = np.load(filename_qualifier+'RCAN_vertical_predictions_'+dataset+'.npy')
    
    # ESRT_beggining_predictions = np.load(filename_qualifier+'ESRT_beggining_predictions_'+dataset+'.npy')
    ESRT_none_predictions = np.load(filename_qualifier+'ESRT_none_predictions_'+dataset+'.npy')
    
    bicubic_predictions = np.load(filename_qualifier+'bicubic_predictions_'+dataset+'.npy')
    lanczos_predictions = np.load(filename_qualifier+'lanczos_predictions_'+dataset+'.npy')
    dummy_predictions = np.load(filename_qualifier+'dummy_predictions_'+dataset+'.npy')
    # l1_predictions = np.load(filename_qualifier+'l1_predictions_'+dataset+'.npy')
    # l2_predictions = np.load(filename_qualifier+'l2_predictions_'+dataset+'.npy')
    
    arrays = [RDN_none_predictions, 
              RCAN_none_predictions,
              ESRT_none_predictions,
              bicubic_predictions, 
              lanczos_predictions,
              dummy_predictions]
    

    # models = ['RCAN_all', 'RCAN_none','RCAN_horizontal','RCAN_vertical','RDN_all', 'RDN_none','RDN_horizontal',
    #           'RDN_vertical','bicubic','lanczos','dummy','p_l1','p_l2']

    models = ['RDN',  'RCAN', 'ESRT','bicubic','lanczos','dummy']
    # run_statistical_test(arrays,models,gt_predictions, dataset)
    generate_statistics(arrays,gt_predictions, models, dataset)
    # generate_roc_curves(arrays,gt_predictions,models,dataset)

if __name__ == "__main__":
   main()
