from torch import Tensor, import_ir_module, nn
import torch
from transformers import Trainer
import numpy as np
import robust_loss_pytorch
from torch.nn.modules.loss import _Loss
from torch.overrides import (has_torch_function, has_torch_function_unary, has_torch_function_variadic, handle_torch_function)
from typing import Optional
from torch.nn import _reduction as _Reduction
from torch._C import _infer_size, _add_docstr
from yaml import warnings
from utils import pearsonr
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union



class CustomTrainerMSE(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels") # Vou buscar as labels à variavel inputs e guardo em 'labels'
        inputs.pop('labels', None) # remover 'labels' da var inputs para não ser passado ao modelo

        outputs = model(**inputs)
        logits = outputs.get("logits")
        
        #%%%%%%%%%%%%%%%%%%%%%%%% MSE Loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        loss_fct = torch.nn.MSELoss()       # Loss function being used
        mse_loss = loss_fct(logits.view(-1), labels.view(-1)) # Logits (^y), labels (y)

        loss = mse_loss
        # print(loss)
        return (loss, outputs) if return_outputs else loss
    
    
        

class CustomTrainerCCC(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


    # This functions overrides class Trainer's compute_loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels") 
        inputs.pop('labels', None) 

        outputs = model(**inputs)
        logits = outputs.get("logits")


        logits_v = logits[:,0]
        logits_a = logits[:,1]
        labels_v = labels[:,0]
        labels_a = labels[:,1]
        # CCC valence
        num_V = 2*pearsonr(logits_v, labels_v)*torch.std(logits_v)*torch.std(labels_v)
        den_V = torch.var(logits_v) + torch.var(labels_v) + torch.square(torch.mean(logits_v) - torch.mean(labels_v))
        ccc_V = num_V/den_V
        cccl_V = 1 - ccc_V


        # CCC arousal
        num_A = 2*pearsonr(logits_a, labels_a)*torch.std(logits_a)*torch.std(labels_a)
        den_A = torch.var(logits_a) + torch.var(labels_a) + torch.square(torch.mean(logits_a) - torch.mean(labels_a))
        ccc_A = num_A/den_A
        cccl_A = 1 - ccc_A

        alpha = 0.5
        beta = 0.5
        cccl_Total = alpha * cccl_V + beta * cccl_A

    
        ccc_loss = cccl_Total
    

        loss =  ccc_loss 
        
        
        return (loss, outputs) if return_outputs else loss

class CustomTrainerMSE_CCC(Trainer): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    # def ccc(gold, pred):
    #     gold_mean = torch.mean(gold)
    #     pred_mean = torch.mean(pred)
    #     covariance = (gold-gold_mean)*(pred-pred_mean)
    #     torch.cov


    # This functions overrides class Trainer's compute_loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels") # Vou buscar as labels à variavel inputs e guardo em 'labels'
        inputs.pop('labels', None) # remover 'labels' da var inputs para não ser passado ao modelo

        outputs = model(**inputs)
        # predictions = outputs[0]
        # predictions = torch.sigmoid(predictions)
        logits = outputs.get("logits")

        #%%%%%%%%%%%%%%%%%%%%%%%% MSE Loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        mse_loss_fct = torch.nn.MSELoss()      
        mse_loss = mse_loss_fct(logits.view(-1), labels.view(-1)) # Logits (^y), labels (y)

        

        #%%%%%%%%%%%%%%%%%%%%%%%% CCC Loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        logits_v = logits[:,0]
        logits_a = logits[:,1]
        labels_v = labels[:,0]
        labels_a = labels[:,1]

        # CCC valence
        num_V = 2*pearsonr(logits_v, labels_v)*torch.std(logits_v)*torch.std(labels_v)
        den_V = torch.var(logits_v) + torch.var(labels_v) + torch.square(torch.mean(logits_v) - torch.mean(labels_v))
        ccc_V = num_V/den_V
        cccl_V = 1 - ccc_V
        # CCC arousal
        num_A = 2*pearsonr(logits_a, labels_a)*torch.std(logits_a)*torch.std(labels_a)
        den_A = torch.var(logits_a) + torch.var(labels_a) + torch.square(torch.mean(logits_a) - torch.mean(labels_a))
        ccc_A = num_A/den_A
        cccl_A = 1 - ccc_A

        alpha = 0.5
        beta = 0.5
        cccl_Total = alpha * cccl_V + beta * cccl_A
        ccc_loss = cccl_Total
        
        #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Sum of the losses
        final_loss = (mse_loss + ccc_loss)/2  # TODO revert if needed
        loss = final_loss
        return (loss, outputs) if return_outputs else loss

class CustomTrainerRobustCCC(Trainer): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_dims = 2
        adaptive_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=self.num_dims,
            float_dtype=np.float32,
            device=adaptive_device,
        )

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        adaptive_params = [p for p in self.adaptive.parameters() if p.requires_grad]
        if adaptive_params:
            existing_param_ids = {
                id(param)
                for group in optimizer.param_groups
                for param in group["params"]
            }
            new_params = [param for param in adaptive_params if id(param) not in existing_param_ids]
            if new_params:
                optimizer.add_param_group({"params": new_params, "weight_decay": 0.0})
        return optimizer

    # This functions overrides class Trainer's compute_loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        #Common to both
        labels = inputs.get("labels") # Vou buscar as labels à variavel inputs e guardo em 'labels'
        inputs.pop('labels', None) # remover 'labels' da var inputs para não ser passado ao modelo

        outputs = model(**inputs)
        # predictions = outputs[0]
        # predictions = torch.sigmoid(predictions)
        logits = outputs.get("logits")

        #%%%%%%%%%%%%%%%%%%%%%%%% CCC Loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        logits_v = logits[:,0]
        logits_a = logits[:,1]
        labels_v = labels[:,0]
        labels_a = labels[:,1]

        # CCC valence
        num_V = 2*pearsonr(logits_v, labels_v)*torch.std(logits_v)*torch.std(labels_v)
        den_V = torch.var(logits_v) + torch.var(labels_v) + torch.square(torch.mean(logits_v) - torch.mean(labels_v))
        ccc_V = num_V/den_V
        cccl_V = 1 - ccc_V


        # CCC arousal
        num_A = 2*pearsonr(logits_a, labels_a)*torch.std(logits_a)*torch.std(labels_a)
        den_A = torch.var(logits_a) + torch.var(labels_a) + torch.square(torch.mean(logits_a) - torch.mean(labels_a))
        ccc_A = num_A/den_A
        cccl_A = 1 - ccc_A

        alpha = 0.5
        beta = 0.5
        cccl_Total = alpha * cccl_V + beta * cccl_A

    
        ccc_loss = cccl_Total
       
        #%%%%%%%%%%%%%%%%%%%%%%%% Robust Loss %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if next(self.adaptive.parameters()).device != logits.device:
            self.adaptive.to(logits.device)
        x = (labels - logits)
        robust_loss_adaptive = torch.mean(self.adaptive.lossfun(x))
        
        #%%%%%%%%%%%%%%%%%%%%%%%% Sum both Losses %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        loss = (ccc_loss + robust_loss_adaptive)/2
        
        return (loss, outputs) if return_outputs else loss
    
    

class CustomTrainerRobust(Trainer):
    # TO USE with ROBUST LOSS
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_dims = 2 
        adaptive_device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
            num_dims=self.num_dims, 
            float_dtype=np.float32, 
            device=adaptive_device
        )
    
    def create_optimizer(self):
        optimizer = super().create_optimizer()
        adaptive_params = [p for p in self.adaptive.parameters() if p.requires_grad]
        if adaptive_params:
            existing_param_ids = {
                id(param)
                for group in optimizer.param_groups
                for param in group["params"]
            }
            new_params = [param for param in adaptive_params if id(param) not in existing_param_ids]
            if new_params:
                optimizer.add_param_group({"params": new_params, "weight_decay": 0.0})
        return optimizer

    # This functions overrides class Trainer's compute_loss function
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels") # Vou buscar as labels à variavel inputs e guardo em 'labels'
        inputs.pop('labels', None) # remover 'labels' da var inputs para não ser passado ao modelo

        outputs = model(**inputs)
        # predictions = outputs[0]
        # predictions = torch.sigmoid(predictions)
        logits = outputs.get("logits")
        

        ########################## Robust Loss ###########################
        # adaptive = robust_loss_pytorch.adaptive.AdaptiveLossFunction(
        #     num_dims=self.num_dims, float_dtype=np.float32, device=0
        # )
        # params = list(model.parameters()) + list(adaptive.parameters())
        # optimizer = torch.optim.Adam(params, lr=0.00002)

        # print('\n')
        # print('logits', logits.view(-1))
        # print('labels', labels.view(-1))
        # print('difference', logits.view(-1) - labels.view(-1))
        # print('difference', labels.view(-1) - logits.view(-1))


        # DOC funcao lossfun()
        """Computes the loss on a matrix.
        Args:
        x: The residual for which the loss is being computed. Must be a rank-2
            tensor, where the innermost dimension is the batch index, and the
            outermost dimension must be equal to self.num_dims. Must be a tensor or
            numpy array of type self.float_dtype.
        **kwargs: Arguments to be passed to the underlying distribution.nllfun().
        Returns:
        A tensor of the same type and shape as input `x`, containing the loss at
        each element of `x`. These "losses" are actually negative log-likelihoods
        (as produced by distribution.nllfun()) and so they are not actually
        bounded from below by zero. You'll probably want to minimize their sum or
        mean.
        """
        ###### UNCOMENT WHEN USING ROBUST LOSS   robust_loss_adaptive = torch.mean(self.adaptive.lossfun((logits.view(-1) - labels.view(-1))[:,None]))#.detach() # True
        
        x = (labels - logits)
        # print('labels ', '\n', labels)
        # print('logits ', '\n', logits)
        # print('labels-logits ', '\n', x)
        # print('(labels-logits)[:,None] ', '\n', x[:,None])
        # print('labels.view(-1) ', '\n', labels.view(-1))
        # print('logits.view(-1) ', '\n', logits.view(-1))
        # print('labels-logits(-1) ', '\n', labels.view(-1) - logits.view(-1))
        # print('labels-logits(-1) ', '\n', (labels.view(-1) - logits.view(-1))[:,None])

        # xx = x
        # print('xx: ', xx)
        # x = torch.transpose(xx,0,1)
        # print('x: ',x)

        # print('x.shape: ', x.shape)
        # print('x.shape[1]: ', x.shape[1])
        # print('self.num_dims: ', self.num_dims)
        # print('len(x.shape[1]): ', len(x.shape[1]))
        # print('num_dims: ', self.num)
        
        # robust_loss_adaptive = torch.mean(self.adaptive.lossfun((labels.view(-1) - logits.view(-1))[:,None]))
        
        if next(self.adaptive.parameters()).device != logits.device:
            self.adaptive.to(logits.device)
        robust_loss_adaptive = torch.mean(self.adaptive.lossfun(x))

        # print('momentary_loss(mean): ', robust_loss_adaptive)
        # print('\n')
        # print('lossfun(x): ', self.adaptive.lossfun(x))
        # print('\n')
        # print('loss={:03f}  alpha={:03f}  scale={:03f}'.format(
            # robust_loss_adaptive.data, self.adaptive.alpha()[0,0].data, self.adaptive.scale()[0,0].data))
        # optimizer.zero_grad()   # clears old gradients from the last step (otherwise 
                                #you’d just accumulate the gradients from all loss.backward() calls).
        # robust_loss_adaptive.backward()     # computes the derivative of the loss w.r.t. the 
                            # parameters (or anything requiring gradients) using backpropagation.  
        # optimizer.step()        # causes the optimizer to take a step based on the gradients of the parameters.

    
        ##################################################################

        # Chose which loss to use (mse_loss or robust_loss_adaptive)
        loss =  robust_loss_adaptive # mse_loss
        
        
        return (loss, outputs) if return_outputs else loss

