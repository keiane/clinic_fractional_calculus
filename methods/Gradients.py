# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:24:38 2021

@author: ianni
"""

import torch
import numpy as np
# from ROAR_Eval_Avg_All_Multi_Map import model, criterion
import model as m
from typing import Any, Callable, cast, Dict, List, overload, Tuple, Union
from torch import device, Tensor

# from captum._utils.gradient import (
#     compute_gradients,
# )

from captum._utils.common import (
    _reduce_list,
    _run_forward,
    _sort_key_list,
    _verify_select_neuron,
)

from captum._utils.typing import (
    Literal,
    ModuleOrModuleList,
    TargetType,
    TensorOrTupleOfTensorsGeneric,
)
import copy
from methods.plotting_functions import VisualizeImageGrayscale

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# # device = torch.device('cpu')

# criterion = torch.nn.CrossEntropyLoss()
# model = m.Model()

# # model.load_state_dict(torch.load("model_weights/model_baseline.dth", map_location=device))
# model.load_state_dict(torch.load('./model_weights/model_baseline.dth', map_location=device))
# model.eval()
# model.to(device)

def returnGrad(img, model, criterion, device = device):
    model.to(device)
    img = img.to(device)
    img.requires_grad_(True)
    pred = model(img)
    loss = criterion(pred, torch.tensor([int(torch.max(pred[0], 0)[1])]).to(device))
    loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    
    return Sc_dx

def compute_gradients(
    forward_fn: Callable,
    inputs: Union[Tensor, Tuple[Tensor, ...]],
    target_ind: TargetType = None,
    additional_forward_args: Any = None,
) -> Tuple[Tensor, ...]:
    r"""
    Computes gradients of the output with respect to inputs for an
    arbitrary forward function.

    Args:

        forward_fn: forward function. This can be for example model's
                    forward function.
        input:      Input at which gradients are evaluated,
                    will be passed to forward_fn.
        target_ind: Index of the target class for which gradients
                    must be computed (classification only).
        additional_forward_args: Additional input arguments that forward
                    function requires. It takes an empty tuple (no additional
                    arguments) if no additional arguments are required
    """
    with torch.autograd.set_grad_enabled(True):
        # runs forward pass
        outputs = _run_forward(forward_fn, inputs, target_ind, additional_forward_args)
        assert outputs[0].numel() == 1, (
            "Target not provided when necessary, cannot"
            " take gradient with respect to multiple outputs."
        )
        # torch.unbind(forward_out) is a list of scalar tensor tuples and
        # contains batch_size * #steps elements
        grads = torch.autograd.grad(torch.unbind(outputs), inputs)
    return grads

class grad_calc():
    def __init__(self, forward_func: Callable) -> None:
        r"""
        Args:

            forward_func (Callable): The forward function of the model or
                        any modification of it.
        """
        self.forward_func = forward_func
        self.gradient_func = compute_gradients

    def returnHigherOrderDeriv(self, 
                               img, n, index=None, device = device):
        # n = derivative order
        # self.forward_func.zero_grad()
        model = self.forward_func
        # model = copy.deepcopy(self.forward_func)
        model.zero_grad()
        model.to(device)
        img = img.to(device)
        img = img.requires_grad_(True)
        
        pred = model(img)
        if index == None:
            index = int(torch.max(pred[0], 0)[1])
        out = pred[0, index]
        # loss = criterion(pred, torch.tensor([index]).to(device))
        # loss.backward()

        # out = model(img)[0, index]
        
        
        # if n==0:
        deriv = (torch.ones_like(img) * torch.max(pred[0], 0)[0]).requires_grad_(True)
        out_derivs = [deriv.clone().detach().numpy()]
        
        ######################################################################
        # model.zero_grad()
        deriv = torch.autograd.grad(out, img, 
                                        create_graph=True,
                                        # retain_graph=True,
                                       materialize_grads=False,
                                    #    is_grads_batched=True,
                                     )[0]
        
        
        # print(deriv
        derivList = [deriv.requires_grad_(requires_grad=True)]
        out_derivs.append(deriv.clone().detach().numpy())
        ######################################################################

        # n == 1 -> 1st derivative
        n_ = int(max(n-1, 0))
        for i in range(n_):
            # model.zero_grad()
            out = derivList[i]#.clone().detach()
            # out = torch.tensor(out_derivs[i]).requires_grad_(requires_grad=True)
            deriv_ = torch.tensor(deriv.clone().detach().numpy()).requires_grad_(True)
            # if i == 0:
            #     print("deriv = ", deriv)
            deriv = torch.autograd.grad((deriv), (img), 
                                        grad_outputs=torch.ones_like(deriv),
                                        create_graph=True,
                                        # retain_graph=(i<n_-1),
                                        materialize_grads=False,
                                        # is_grads_batched=True,
                                        )[0]
            # deriv = out
            derivList.append(deriv.requires_grad_(requires_grad=True))
            out_derivs.append(deriv.clone().detach().numpy())
        
        deriv_out = torch.tensor(out_derivs[n])
        
        return deriv_out



def returnHigherOrderDeriv(img, 
                           model, 
                           n, index=None, device = device):
    # n = derivative order
    # self.forward_func.zero_grad()
    # model = self.forward_func
    # model = copy.deepcopy(self.forward_func)
    model.zero_grad()
    model.to(device)
    img = img.to(device)
    img.requires_grad_(True)
    pred = model(img)
    if index == None:
        index = int(torch.max(pred[0], 0)[1])
    # loss = criterion(pred, torch.tensor([index]).to(device))
    # loss.backward()
    
    # deriv1 = img.grad.requires_grad_(True)
    out = pred[0, index]
    # out = torch.max(pred[0], 0)[0]
    
    # if n==0:
    deriv = (torch.ones_like(img) * torch.max(pred[0], 0)[0]).requires_grad_(True)
    out_derivs = [deriv.clone().detach().numpy()]
    
    ######################################################################
    # model.zero_grad()
    deriv = torch.autograd.grad(out, img, 
                                    # grad_outputs=out,
                                    create_graph=True,
                                    # retain_graph=False,
                                    # retain_graph=True,
                                   materialize_grads=True,
                                 )[0]
    # print(deriv)
    # derivList = [torch.empty_like(deriv1).copy_(deriv1).requires_grad_(requires_grad=True),]
    derivList = [deriv.requires_grad_(requires_grad=True)]
    out_derivs.append(deriv.clone().detach().numpy())
    ######################################################################

    
    for i in range(n):
        # model.zero_grad()
        out = derivList[i]#.clone().detach()
        # out = torch.tensor(out_derivs[i]).requires_grad_(requires_grad=True)
        deriv = torch.autograd.grad((deriv), (img), 
                                    # grad_outputs=torch.ones_like(deriv),
                                    grad_outputs=(deriv),
                                    # create_graph=True,
                                    # retain_graph=(i<n), 
                                    # retain_graph=True,
                                    materialize_grads=True,
                                    )[0]#.requires_grad_(requires_grad=True)
        # deriv = out
        derivList.append(deriv.requires_grad_(requires_grad=True))
        out_derivs.append(deriv.clone().detach().numpy())
    
    deriv_out = torch.tensor(out_derivs[n])
    
    return deriv_out
    # return deriv

def returnGradPred(img, model, criterion):
    
    if (torch.cuda.is_available()):
        img = img.cuda()
    
    img.requires_grad_(True)
    pred = model(img)
    
    label = torch.tensor([int(torch.max(pred[0], 0)[1])])
    
    if (torch.cuda.is_available()):
        label = label.cuda()
    
    loss = criterion(pred, label)
    loss.backward()
    
#    S_c = torch.max(pred[0].data, 0)[0]
    Sc_dx = img.grad
    
    return Sc_dx, pred

def returnGradPredIndex(img, index, model, criterion):
    
    index = torch.tensor([int(index)])
    
    if (torch.cuda.is_available()):
        img = img.cuda()
        index = index.cuda()
    
    img.requires_grad_(True)
    pred = model(img)
    loss = criterion(pred, index)
    loss.backward()
    Sc_dx = img.grad
    
    return Sc_dx, pred


def random_baseline_deeplift(
    inp, dl,
    target_label_index,
    magnitude = True,
    num_random_trials=10):
  all_dlgrads = []
#  dt = np.dtype('d')  # double-precision floating-point number
  base = torch.tensor(np.float32((0.01*np.random.random(np.array(inp.shape)))))
  for i in range(num_random_trials):
    dlgrads = dl.attribute(inp,
                           base,
                           target=target_label_index,
                           return_convergence_delta=False).detach()
    if magnitude:
        dlgrads = dlgrads * dlgrads
    all_dlgrads.append(np.array(dlgrads))
  avg_dlgrads = torch.tensor(np.average(np.array(all_dlgrads), axis=0))
  return avg_dlgrads

def imgAccuracy(img, label, model):
    correct = 0
    total = 0
    pred = model(img)
    for i, p in enumerate(pred):
        if label[i] == torch.max(p.data, 0)[1]:
            correct = correct + 1
        total = total + 1
    test_accuracy = correct / total
    # print(test_accuracy)
    return test_accuracy