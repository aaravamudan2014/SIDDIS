from scipy import signal
import torch
import numpy as np

scaling_factor = 10

class SRLoss(torch.nn.Module):
    def __init__(self, device, eta):
        super(SRLoss,self).__init__()
        self.device = device
        self.eta=eta
    
    # preds, label, input
    def forward(self,output,target,inp):
        ret_loss = torch.zeros(1, 1, requires_grad=False).to(self.device)
        
        conv_filter = np.ones((scaling_factor,scaling_factor))/scaling_factor**2
        sample_matrices = output[:,0].cpu().detach().numpy()

        conv_output = np.zeros_like(inp[:,0].cpu().detach().numpy())
        # scaling factor here is actually the stride
        for i in range(len(sample_matrices)):
          conv_output[i] = signal.convolve2d(sample_matrices[i], conv_filter, mode='valid')[::scaling_factor, ::scaling_factor]

        ret_loss = ret_loss + torch.mean(self.eta*(inp[:,0] - torch.from_numpy(conv_output).float().to(self.device))**2)
        
        #print(target)
        
        # This is done in order to avoid nans: why is this happening? 
        # In case of very large outputs from previous layer, the loss is NaN

        output_complement = 1-output
        output_complement[output_complement==0] = 1.e-20
        output_main = torch.clone(output)
        output_main[output_main==0] = 1.e-20

        ret_loss = ret_loss - torch.mean((torch.mul(target, torch.log(output_main)) + torch.mul(1.0-target, torch.log(output_complement) )))

        # output_complement = 1-output
        # output_complement[output_complement==0] = 1.e-20
        # output_main = torch.clone(output)
        # output_main[output_main==0] = 1.e-20

        # ret_loss = ret_loss - torch.mean((torch.mul(target, torch.log(output_main)) + torch.mul(1.0-target, torch.log(output_complement))))
        return ret_loss

class BCELoss(torch.nn.Module):
    def __init__(self, device):
        super(BCELoss,self).__init__()
        self.device = device
    # preds, label
    def forward(self,output,target):
        ret_loss = torch.zeros(1, 1, requires_grad=False).to(self.device)

        # BCE loss
        # This is done in order to avoid nans.
        output_complement = 1-output
        output_complement[output_complement==0] = 1.e-20
        output_main = torch.clone(output)
        output_main[output_main==0] = 1.e-20

        ret_loss = ret_loss - torch.mean((torch.mul(target, torch.log(output_main)) + torch.mul(1.0-target, torch.log(output_complement) )))
        return ret_loss