import torch
from RDN import *
from RCAN import *
from RDN_Comb import *
from loss import *
from dataset_classes import *
from utils import *
import os
import copy
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
import gc
import os
import argparse
from typing import Dict
from DeepRivSRM import *
import optuna
import matplotlib.pyplot as plt
import traceback

###########
# Globals #
###########

# fixed settings for all experiments
GLOBAL_SETTINGS = {
    'batch_size': 16,
    'epochs': 400,
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
    parser.add_argument('--model_type', choices=["RDN", "RCAN", "ViT"])
    parser.add_argument('--topo_inclusion', choices=["beggining", "none", "vertical", "horizontal"])
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
                
                num_features = trial.suggest_int("num_features", 8, 64,8)
                num_blocks = trial.suggest_int("num_blocks", 2,16,2)
                num_layers = trial.suggest_int("num_layers", 2,32,2)
                eta = trial.suggest_float("eta", 0, 2000)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                dropout_prob_input = trial.suggest_float("dropout_prob_input", 1e-2,2e-1)
                dropout_prob_topo_1 = trial.suggest_float("dropout_prob_topo_1", 1e-2,2e-1)
                dropout_prob_topo_2 = trial.suggest_float("dropout_prob_topo_2", 1e-2,2e-1)
                seed = trial.suggest_categorical("seed", [100,200,300,400,500,600,700,800,900])
                
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
                print("dropout_prob_input: ", dropout_prob_input)
                print("dropout_prob_topo_1: ", dropout_prob_topo_1)
                print("dropout_prob_topo_2: ", dropout_prob_topo_2)
                

                torch.manual_seed(seed)
                model = RDN(scale_factor=8,
                            num_channels=1,
                            num_features=num_features,
                            growth_rate=growth_rate,
                            num_blocks=num_blocks,
                            num_layers=num_layers,
                            topo_inclusion=config['topo_inclusion'],
                            dropout_prob_input=dropout_prob_input,
                            dropout_prob_topo_1=dropout_prob_topo_1,
                            dropout_prob_topo_2=dropout_prob_topo_2).to(device)
                        
            elif config['model_type'] == "RCAN":
                num_features = trial.suggest_int("n_features", 4, 64,4)
                num_rg = trial.suggest_int("num_rg", 10, 40,5)
                num_rcab = trial.suggest_int("num_rcab", 20, 50, 5)
                reduction = trial.suggest_int("reduction", 16, 64, 4)
                learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
                eta = trial.suggest_float("eta", 0, 2000)
                dropout_prob_input = trial.suggest_float("dropout_prob_input", 1e-1,5e-1)
                dropout_prob_topo_1 = trial.suggest_float("dropout_prob_topo_1", 1e-1,5e-1)
                dropout_prob_topo_2 = trial.suggest_float("dropout_prob_topo_2", 1e-1,5e-1)
                seed = trial.suggest_categorical("seed", [100,200,300,400,500,600,700,800,900])
                
                
                print("Optuna parameters: \n")
                print("num_features: ", num_features)
                print("num_rg: ", num_rg)
                print("num_rcab: ", num_rcab)
                print("reduction: ", reduction)
                print("learning_rate: ", learning_rate)
                print("eta: ", eta)
                print("dropout_prob_input: ", dropout_prob_input)
                print("dropout_prob_topo_1: ", dropout_prob_topo_1)
                print("dropout_prob_topo_2: ", dropout_prob_topo_2)
                
                torch.manual_seed(seed)
                model = RCAN(scale=8, 
                        num_features=num_features, 
                        num_rg=num_rg, 
                        num_rcab=num_rcab, 
                        reduction=reduction,
                        topo_inclusion=config['topo_inclusion'],
                        dropout_prob_input=dropout_prob_input,
                        dropout_prob_topo_1=dropout_prob_topo_1,
                        dropout_prob_topo_2=dropout_prob_topo_2).to(device)

            criterion = SRLoss(device,eta=eta)
            
            # training
            torch.cuda.empty_cache()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate,)
            
            best_val_mcc = 0.0
            best_val_acc = 0.0
            
            batch_size = config['batch_size']
            epochs = 10

            train_dataset = TrainDataset('data/training.h5', patch_size=4, scale=scaling_factor)
            train_dataloader = DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=2,
                                            pin_memory=True)
            eval_dataset = EvalDataset('data/validation_rw_comb.h5')
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
                avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results(model, device, dataset='validation_rw_comb')
                
                if avg_mcc > best_val_mcc:
                    best_val_mcc = avg_mcc
                
                if avg_acc > best_val_acc:
                    best_val_acc = avg_acc
        except Exception as e:
            traceback.print_exc()
            return 0.0

        return best_val_mcc
        
    
    # if config['study'] == "continue":
    #     study = optuna.load_study(storage='sqlite:///optuna_study_none_304.db',
    #                                 study_name="RDN_None_seed304")
    # else:
    study = optuna.create_study(direction="maximize",
                                pruner=optuna.pruners.MedianPruner(),
                                storage='sqlite://optuna_search'+config['topo_inclusion']+'_'+config['model_type']+'.db')
    


    study.optimize(objective, n_trials=100)

    print("Study statistics: ")
    
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    optuna.visualization.plot_optimization_history(study)
    plt.savefig('results/hyperparameter_search.png')

def train(config):
    gc.collect()
    device = config['device']
    RDN_config = {'num_features': 64, 'num_blocks': 6, 'num_layers': 22, 'eta': 1587.0264621542353, 'learning_rate': 5.0187976191403665e-05, 'dropout_prob_input': 0.05637435083783727, 'dropout_prob_topo_1': 0.06378997926991206, 'dropout_prob_topo_2': 0.19550513533348995, 'seed': 500}
    
    # none   Trial 6 finished with value: 0.7631142345550341 and parameters: {'num_features': 64, 'num_blocks': 6, 'num_layers': 22, 'eta': 1587.0264621542353, 'learning_rate': 5.0187976191403665e-05, 'dropout_prob_input': 0.05637435083783727, 'dropout_prob_topo_1': 0.06378997926991206, 'dropout_prob_topo_2': 0.19550513533348995, 'seed': 500}
    # vertical Trial 2 finished with value: 0.8154224616246308 and parameters: {'num_features': 40, 'num_blocks': 6, 'num_layers': 24, 'eta': 793.8541797917389, 'learning_rate': 0.0005852923434708634, 'dropout_prob_input': 0.055525141849995595, 'dropout_prob_topo_1': 0.09179782782035235, 'dropout_prob_topo_2': 0.19875675010727667, 'seed': 700}. Best is trial 2 with value: 0.8154224616246308
    # horizontal Trial 11 finished with value: 0.8064141049643685 and parameters: {'num_features': 24, 'num_blocks': 10, 'num_layers': 14, 'eta': 1358.6359050019717, 'learning_rate': 0.00018420577498247598, 'dropout_prob_input': 0.014919203222285985, 'dropout_prob_topo_1': 0.08350285551825995, 'dropout_prob_topo_2': 0.06912989930590557, 'seed': 900}. Best is trial 11 with value: 0.8064141049643685.  
    # beggining Trial 8 finished with value: 0.8153885867955213 and parameters: {'num_features': 24, 'num_blocks': 12, 'num_layers': 6, 'eta': 769.2459733987456, 'learning_rate': 0.0003187699499501108, 'dropout_prob_input': 0.0913698966196942, 'dropout_prob_topo_1': 0.08564581178387892, 'dropout_prob_topo_2': 0.01883397860868869, 'seed': 900}. Best is trial 8 with value: 0.8153885867955213
    
    batch_size = config['batch_size']
    epochs = config['epochs']

    RCAN_config = {'num_rg': 10, 'num_features':64, 'num_rcab':20,'reduction': 36, 'learning_rate': 3.050329222073434e-05, 'eta': 1881.6920361768118, 'dropout_prob_input': 0.25241658385132404, 'dropout_prob_topo_1': 0.1995129806101335, 'dropout_prob_topo_2': 0.12158360258766605, 'seed': 500}

    # none Trial 5 finished with value: 0.7583894980256135 and parameters: {'num_rg': 15, 'num_rcab':20,'num_features':64, 'reduction': 28, 'learning_rate': 0.00019332308902926046, 'eta': 1086.6298309371657, 'dropout_prob_input': 0.2727037500725949, 'dropout_prob_topo_1': 0.4497705316859635, 'dropout_prob_topo_2': 0.40846088328284125, 'seed': 900}. Best is trial 5 with value: 0.7583894980256135.
    # vertical Trial 0 finished with value: 0.7947969408127296 and parameters: {'num_rg': 25, 'reduction': 16, 'learning_rate': 8.03258742015366e-05, 'eta': 1030.5174808544998, 'dropout_prob_input': 0.36224609638036687, 'dropout_prob_topo_1': 0.1726326521798335, 'dropout_prob_topo_2': 0.48474970955774555, 'seed': 600}. Best is trial 0 with value: 0.7947969408127296.
    # horizontal Trial 2 finished with value: 0.8124379192605219 and parameters: {'num_rg': 20, 'reduction': 48, 'learning_rate': 8.137495002849483e-05, 'eta': 645.8492111016832, 'dropout_prob_input': 0.11858267431161425, 'dropout_prob_topo_1': 0.4290829332025423, 'dropout_prob_topo_2': 0.2868212478347976, 'seed': 500}. Best is trial 2 with value: 0.8124379192605219.
    # beggining Trial 17 finished with value: 0.8118214165271584 and parameters: {'num_rg': 10, 'reduction': 36, 'learning_rate': 3.050329222073434e-05, 'eta': 1881.6920361768118, 'dropout_prob_input': 0.25241658385132404, 'dropout_prob_topo_1': 0.1995129806101335, 'dropout_prob_topo_2': 0.12158360258766605, 'seed': 500}. Best is trial 17 with value: 0.8118214165271584.
    batch_size = config['batch_size']
    epochs = config['epochs']
    
    if config['model_type'] == "RDN":
        lr = RDN_config['learning_rate']
        eta = RDN_config['eta']
        torch.manual_seed(RDN_config['seed'])
        model = RDN(scale_factor=8, # no change
                    num_channels=1, # no change
                    num_features=RDN_config['num_features'], 
                    growth_rate=RDN_config['num_features'], # same as num_features
                    num_blocks=RDN_config['num_blocks'],
                    num_layers=RDN_config['num_layers'],
                    topo_inclusion=config['topo_inclusion'],
                    dropout_prob_input=RDN_config['dropout_prob_input'],
                    dropout_prob_topo_1=RDN_config['dropout_prob_topo_1'],
                    dropout_prob_topo_2=RDN_config['dropout_prob_topo_2']
                    ).to(device)
    elif config['model_type'] == "RCAN":
        torch.manual_seed(RCAN_config['seed'])
        lr = RCAN_config['learning_rate']
        eta = RCAN_config['eta']
        model = RCAN(scale=8, 
                    num_features=RCAN_config['num_features'], 
                    num_rg=RCAN_config['num_rg'], 
                    num_rcab=RCAN_config['num_rcab'], 
                    reduction=RCAN_config['reduction'],
                    topo_inclusion=config['topo_inclusion'],
                    dropout_prob_input=RCAN_config['dropout_prob_input'],
                    dropout_prob_topo_1=RCAN_config['dropout_prob_topo_1'],
                    dropout_prob_topo_2=RCAN_config['dropout_prob_topo_2']).to(device)

    criterion = SRLoss(device,eta=eta)

    torch.cuda.empty_cache()

    optimizer = optim.AdamW(model.parameters(), lr=lr)
  
    best_weights = copy.deepcopy(model.state_dict())
    
    best_epoch = 0
    best_val_loss = np.inf
    


    train_dataset = TrainDataset('data/training.h5', patch_size=4, scale=scaling_factor)
    train_dataloader = DataLoader(dataset=train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=2,
                                    pin_memory=True)
    eval_dataset = EvalDataset('data/validation_rw_comb.h5')
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
              
              
              with torch.autograd.set_detect_anomaly(True):
                preds = model(inputs, topo_1, topo_2)
              
                loss = criterion(preds, target, inputs)
                epoch_losses.update(loss.item(), len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

              t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
              t.update(len(inputs))

    #   if (epoch + 1) % 10 == 0:
    #       torch.save(model.state_dict(), '/best_model'+config['model_type']+config['topo_inclusion']+'epoch_{}.pth'.format(epoch))

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
              preds = model(inputs, topo_1, topo_2)


          sr_loss = criterion(preds, labels,inputs ).item()
          epoch_sr.update(sr_loss, len(inputs))

      print('eval loss: {:.4f}'.format(epoch_sr.avg))
      avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results(model, device, dataset='test')
    
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
                    topo_inclusion=config['topo_inclusion'],
                    dropout_prob_input=model_dict['dropout_prob_input'],
                    dropout_prob_topo_1=model_dict['dropout_prob_topo_1'],
                    dropout_prob_topo_2=model_dict['dropout_prob_topo_2']).to(device)
    elif model_type == "RCAN":
        model_dict = np.load(config['run_dir'] + '/best_model'+config['model_type']+config['topo_inclusion']+'.npy', allow_pickle=True).item()
        model = RCAN(scale=8, 
                        num_features=model_dict['num_features'], 
                        num_rg=model_dict['num_rg'], 
                        num_rcab=model_dict['num_rcab'], 
                        reduction=model_dict['reduction'],
                        topo_inclusion=config['topo_inclusion'],
                        dropout_prob_input=model_dict['dropout_prob_input'],
                        dropout_prob_topo_1=model_dict['dropout_prob_topo_1'],
                        dropout_prob_topo_2=model_dict['dropout_prob_topo_2']).to(device)

    
    model.load_state_dict(torch.load('runs/run_final/best_model'+config['model_type']+config['topo_inclusion']+'.h5'))
    model.eval()
    avg_mcc, avg_acc, total_calculated_mcc, total_calculated_acc = gen_test_results(model, device, dataset='test')
    acc_landsat_8, mcc_landsat_8 = gen_Landsat8_results(model, device)
    acc_eu_external, mcc_eu_external = gen_Landsat8_results(model, device, dataset_type='EU_external')
    acc_rr_external, mcc_rr_external = gen_Landsat8_results(model, device, dataset_type='RR_Trimmed_External')
    
    
    
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
    eval_dataset = EvalDataset('validation_rw_comb.h5')
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

