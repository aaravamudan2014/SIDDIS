import torch
from torch import nn
from RDN import *
from RCAN import *
from RDN_Comb import *
from loss import *
from dataset_classes import *
from utils import *
import os
import copy
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import gc
import os
import argparse
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
from DeepRivSRM import *
import optuna
import matplotlib.pyplot as plt

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 32,
    'epochs': 100,
}


def get_args() -> Dict:
    """Parse input arguments

    Returns
    -------
    dict
        Dictionary containing the run config.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=["train", "evaluate", "optuna_search", "comparison"])
    parser.add_argument('--model_type', choices=["RDN","RDN_comb", "RCAN", "ViT"])
    parser.add_argument('--topo_inclusion', choices=["beggining", "none", "vertical", "horizontal", "combination"])
    parser.add_argument('--study', choices=["start", "continue"])

    parser.add_argument('--seed', type=int, required=False, help="Random seed")
    parser.add_argument('--run_dir', type=str, help="For evaluation mode. Path to run directory.")
    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help="User-selected GPU ID - if none chosen, will default to cpu")
    # parser.add_argument(
    #     '--num_workers', type=int, default=12, help="Number of parallel threads for data loading")
    cfg = vars(parser.parse_args())

    # Validation checks
    if (cfg["mode"] in ["train", "optuna_search", "comparison"]) and (cfg["seed"] is None):
        # generate random seed for this run
        cfg["seed"] = int(np.random.uniform(low=0, high=1e6))

    if (cfg["mode"] in ["evaluate"]) and (cfg["run_dir"] is None):
        raise ValueError("In evaluation mode a run directory (--run_dir) has to be specified")

    # GPU selection
    device = f"cuda:{cfg['gpu']}"
    DEVICE = torch.device(device if torch.cuda.is_available() else "cpu")
    cfg['device'] = DEVICE
    # combine global settings with user config
    cfg.update(GLOBAL_SETTINGS)

    if cfg["mode"] in ["train", "optuna_search"]:
        # print config to terminal
        for key, val in cfg.items():
            print(f"{key}: {val}")

    return cfg

def optuna_search(config):
    print("starting optuna trial")
    def objective(trial):
        gc.collect()
        try:
            device = config['device']

            if config['model_type'] == "RDN":
                num_features = trial.suggest_int("num_features", 16, 64,8)
                num_blocks = trial.suggest_int("num_blocks", 8,16,4)
                num_layers = trial.suggest_int("num_layers", 8,64,8)
                eta = trial.suggest_float("eta", 0, 2000)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)

                # model based size constraints
                # growth_rate = trial.suggest_int("growth_rate", 8,16,4)
                growth_rate = num_features

                print("Optuna parameters: \n")
                print("num_features: ", num_features)
                print("growth_rate: ", growth_rate)
                print("num_blocks: ", num_blocks)
                print("num_layers: ", num_layers)
                print("learning_rate: ", learning_rate)
                print("eta: ", eta)

                torch.manual_seed(config['seed'])
                model = RDN(scale_factor=8,
                            num_channels=1,
                            num_features=num_features,
                            growth_rate=growth_rate,
                            num_blocks=num_blocks,
                            num_layers=num_layers,
                            topo_inclusion=config['topo_inclusion']).to(device)
                        
            elif config['model_type'] == "RCAN":
                num_features = 64#trial.suggest_int("n_features", 4, 64,4)
                num_rg = trial.suggest_int("num_rg", 10, 40,5)
                num_rcab = 20#trial.suggest_int("num_rcab", 20, 50, 5)
                reduction = trial.suggest_int("reduction", 16, 64, 4)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                eta = trial.suggest_float("eta", 0, 2000)
                
                print("Optuna parameters: \n")
                print("num_features: ", num_features)
                print("num_rg: ", num_rg)
                print("num_rcab: ", num_rcab)
                print("reduction: ", reduction)
                print("learning_rate: ", learning_rate)
                print("eta: ", eta)
                
                torch.manual_seed(config['seed'])
                model = RCAN(scale=8, 
                        num_features=num_features, 
                        num_rg=num_rg, 
                        num_rcab=num_rcab, 
                        reduction=reduction,
                        topo_inclusion=config['topo_inclusion']).to(device)

            criterion = SRLoss(device,eta=eta)
            
            # training
            torch.cuda.empty_cache()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate,)
            
            best_epoch = 0
            best_val_mcc = 0.0
            
            batch_size = config['batch_size']
            epochs = 15

            train_dataset = TrainDataset('training.h5', patch_size=4, scale=scaling_factor)
            train_dataloader = DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True)
            eval_dataset = EvalDataset('validation.h5')
            eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size)

            for epoch in range(epochs):
                torch.cuda.empty_cache()
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = learning_rate * (0.1 ** (epoch // int(epochs * 0.8)))
            
                model.train()
                epoch_losses = AverageMeter()

                with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size), ncols=80) as t:
                    t.set_description('epoch: {}/{}'.format(epoch, epochs - 1))

                    for data in train_dataloader:
                        inputs, target, topo_1, topo_2 = data
                        inputs = inputs.to(device)
                        target = target.to(device)
                        topo_1 = topo_1.to(device)
                        topo_2 = topo_2.to(device)
                        
                        preds = model(inputs, topo_1, topo_2)
                        # with torch.autograd.set_detect_anomaly(True):
                        loss = criterion(preds, target, inputs)
                        epoch_losses.update(loss.item(), len(inputs))

                        optimizer.zero_grad()

                        loss.backward()
                        optimizer.step()

                        t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                        t.update(len(inputs))

                if (epoch + 1) % 10 == 0:
                    torch.save(model.state_dict(), os.path.join(config['run_dir']+'epoch_{}.pth'.format(epoch)))

                model.eval()
                avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results(model, device, dataset='validation')
                avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results(model, device, dataset='test')
                
                if avg_mcc > best_val_mcc:
                    best_val_mcc = avg_mcc

        except:
            return 0.0

        return best_val_mcc
        
    
    # if config['study'] == "continue":
    #     study = optuna.load_study(storage='sqlite:///optuna_study_none_304.db',
    #                                 study_name="RDN_None_seed304")
    # else:
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(),
                                storage='sqlite:///optuna_studoptuna_study_none_304_none.db',
                                )
    


    study.optimize(objective, n_trials=100)

    print("Study statistics: ")
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    optuna.visualization.plot_optimization_history(study)
    plt.savefig('hyperparameter_search.png')

def train(config):
    gc.collect()
    torch.cuda.empty_cache()
    device = config['device']
    RDN_config = {'num_features':64 ,'num_blocks':8, 'num_layers':8}
    lr = 0.0001
    batch_size = config['batch_size']
    epochs = config['epochs']

    # RCAN_config = {'num_features':64 ,'num_rg':10, 'num_rcab':20, 'reduction':48}
    # lr = 0.000972271859302153
    # eta = 497.537661766163
    # batch_size = config['batch_size']
    # epochs = config['epochs']
    
    if config['model_type'] == "RDN":
        model = RDN(scale_factor=8, # no change
                        num_channels=1, # no change
                        num_features=RDN_config['num_features'], 
                        growth_rate=RDN_config['num_features'], # same as num_features
                        num_blocks=RDN_config['num_blocks'],
                        num_layers=RDN_config['num_layers'],
                        topo_inclusion=config['topo_inclusion']).to(device)
    elif config['model_type'] == "RCAN":
        model = RCAN(scale=8, 
                        num_features=RCAN_config['num_features'], 
                        num_rg=RCAN_config['num_rg'], 
                        num_rcab=RCAN_config['num_rcab'], 
                        reduction=RCAN_config['reduction'],
                        topo_inclusion=config['topo_inclusion']).to(device)
    elif config['model_type'] == "RDN_comb":
        print("Model being used is RDN Combination")
        model = RDN_comb(num_channels=1, # no change
                        num_features=RDN_config['num_features'], 
                        growth_rate=RDN_config['num_features'], # same as num_features
                        num_blocks=RDN_config['num_blocks'],
                        num_layers=RDN_config['num_layers'],
                        topo_inclusion=config['topo_inclusion']).to(device)

    criterion = BCELoss(device)

    torch.cuda.empty_cache()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_loss = np.inf
    train_dataset = TrainDatasetCombination('training_combination.h5', patch_size=4, scale=scaling_factor)
    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    pin_memory=True)
    eval_dataset = EvalDatasetCombination('validation_combination.h5')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size)
    for epoch in range(epochs):
        torch.cuda.empty_cache()
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * (0.1 ** (epoch // int(epochs * 0.8)))
        
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, epochs - 1))

            for data in train_dataloader:
                rdn_high_res, gmu, high_res, low_res = data
                rdn_high_res = rdn_high_res.to(device)
                low_res = low_res.to(device)
                
                gmu = gmu.to(device)
                high_res = high_res.to(device)

                with torch.autograd.set_detect_anomaly(False):
                    preds = model(rdn_high_res, gmu)
                    bce_loss = criterion(preds, high_res)
                    epoch_losses.update(bce_loss.item(), len(rdn_high_res))

                    optimizer.zero_grad()
                    bce_loss.backward()
                    optimizer.step()

                    t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                    t.update(len(rdn_high_res))
                    

               
        model.eval()
        epoch_sr = AverageMeter()
        with tqdm(total=(len(eval_dataset) - len(eval_dataset) % batch_size), ncols=80) as t:
            eval_epoch = 0
            t.set_description('eval epoch: {}/{}'.format(eval_epoch, 1))
            # validation section
            for data in eval_dataloader:
                rdn_high_res, gmu, high_res, low_res = data
                rdn_high_res = rdn_high_res.to(device)
                low_res = low_res.to(device)
                gmu = gmu.to(device)
                high_res = high_res.to(device)
                
                with torch.no_grad():
                    preds = model(rdn_high_res, gmu)


                bce_loss = criterion(preds, high_res).item()
                epoch_sr.update(bce_loss, len(rdn_high_res))
                t.set_postfix(loss='{:.6f}'.format(epoch_sr.avg))
                t.update(len(rdn_high_res))
                

        avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results_combination(model, device, dataset='test_combination')

        if epoch_sr.avg < best_val_loss:
            print("Saving best weights...")
            best_epoch = epoch
            best_val_loss = epoch_sr.avg
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.h5')
            if config['model_type'] == "RDN":
                model_dict = RDN_config
            elif config['model_type'] == "RCAN":
                model_dict = RCAN_config
            elif config['model_type'] == "RDN_comb":
                model_dict = RDN_config
                    
            np.save(config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.npy',model_dict)
    
    model.load_state_dict(best_weights)

    return best_val_loss

def evaluate(config):
    # test_dataset = EvalDataset('test.h5')
    # test_dataloader = DataLoader(dataset=test_dataset, batch_size=config['batch_size'])
    # epoch_mcc = AverageMeter()
    # total_calculated = 0
    
    
    device = config['device']
    model_type = config['model_type']
    if model_type == "RDN":
        model_dict = np.load(config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.npy', allow_pickle=True).item()
        model = RDN(scale_factor=8,
                    num_channels=1,
                    num_features=model_dict['num_features'],
                    growth_rate=model_dict['num_features'],
                    num_blocks=model_dict['num_blocks'],
                    num_layers=model_dict['num_layers'],
                    topo_inclusion=config['topo_inclusion']).to(device)
    elif model_type == "RCAN":
        model_dict = np.load(config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.npy', allow_pickle=True).item()
        model = RCAN(scale=8, 
                        num_features=model_dict['num_features'], 
                        num_rg=model_dict['num_rg'], 
                        num_rcab=model_dict['num_rcab'], 
                        reduction=model_dict['reduction'],
                        topo_inclusion=config['topo_inclusion']).to(device)
    elif config['model_type'] == "RDN_comb":
        print("Model being used is RDN Combination")
        model_dict = np.load(config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.npy', allow_pickle=True).item()
        model = RDN_comb(num_channels=1, # no change
                        num_features=model_dict['num_features'], 
                        growth_rate=model_dict['num_features'], # same as num_features
                        num_blocks=model_dict['num_blocks'],
                        num_layers=model_dict['num_layers'],
                        topo_inclusion=config['topo_inclusion']).to(device)
    
    model.load_state_dict(torch.load('runs/run_final/best_model'+config['model_type']+config['topo_inclusion']+'.h5'))
    model.eval()
    avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results_combination(model, device, dataset='test_combination')
    gen_Landsat8_results_combination(model, device)
    gen_Landsat8_results_combination(model, device, dataset_type='EU_external_combination')
    
    return None


def comparison(config):
    # use hyper-parameters as mentioned in the paper.
    
    gc.collect()

    device = config['device']

    model = ResUnetppModified(n_channels=1,
                                n_classes=1, 
                                scale_factor=8, 
                                bilinear=False).to(device)

    criterion = SRLoss(device,eta=0)

    torch.cuda.empty_cache()
    lr = 0.001
    optimizer = optim.AdamW(model.parameters(), lr=lr)
  
    best_epoch = 0
    best_val_loss = np.inf
    
    batch_size = config['batch_size']
    epochs = config['epochs']

    train_dataset = TrainDataset('training.h5', patch_size=4, scale=scaling_factor)
    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=2,
                                    pin_memory=True)
    eval_dataset = EvalDataset('validation.h5')
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size)

    for epoch in range(epochs):
      for param_group in optimizer.param_groups:
          param_group['lr'] = lr * (0.1 ** (epoch // int(epochs * 0.8)))
      model.train()
      epoch_losses = AverageMeter()

      with tqdm(total=(len(train_dataset) - len(train_dataset) % batch_size), ncols=80) as t:
          t.set_description('epoch: {}/{}'.format(epoch, epochs - 1))

          for data in train_dataloader:
              inputs, target, topo_1, topo_2 = data
              inputs = inputs.to(device)
              target = target.to(device)
              topo_1 = topo_1.to(device)
              topo_2 = topo_2.to(device)
              
              # As per the paper, this model does not 
              # take in topographic features as input 
              preds = model(inputs, None, None)
              with torch.autograd.set_detect_anomaly(True):
                loss = criterion(preds, target, inputs)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

              t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
              t.update(len(inputs))

      if (epoch + 1) % 10 == 0:
          torch.save(model.state_dict(), os.path.join(config['run_dir']+'/epoch__drsrm_{}.pth'.format(epoch)))

      model.eval()
      epoch_sr = AverageMeter()
      
      # validation section
      for data in eval_dataloader:
          inputs, labels, topo_1, topo_2 = data

          inputs = inputs.to(device)
          labels = labels.to(device)
          topo_1 = topo_1.to(device)
          topo_2 = topo_2.to(device)
          
          with torch.no_grad():
              preds = model(inputs, None, None)


          sr_loss = criterion(preds, labels,inputs).item()
          epoch_sr.update(sr_loss, len(inputs))

      print('eval loss: {:.4f}'.format(epoch_sr.avg))
      
      avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results(model, device, dataset='validation')

      if epoch_sr.avg < best_val_loss:
          best_epoch = epoch
          best_val_loss = epoch_sr.avg
          best_weights = copy.deepcopy(model.state_dict())
          torch.save(model.state_dict(), config['run_dir'] + '/best_model_drsrm.h5')


    # best_weights = torch.load(config['run_dir'] + '/best_model_drsrm.h5')
    print('best epoch: {}, val loss: {:.2f}'.format(best_epoch, best_val_loss))
    model.load_state_dict(best_weights)
    avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results(model, device, dataset='test')
    gen_Landsat8_results(model, device)
    gen_Landsat8_results(model, device, dataset_type='EU_external')
    

    

if __name__ == "__main__":
    config = get_args()
    globals()[config["mode"]](config)

