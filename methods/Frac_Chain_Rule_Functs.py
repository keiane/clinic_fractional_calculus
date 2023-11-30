# -*- coding: utf-8 -*-

"""
Created on Wed Sept 20 15:21:22 2023

@author: ianni
"""

import torch
import numpy as np
# import model as m
import math
from methods.Gradients import *
from methods.plotting_functions import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def ChainFracGrad(inp, alpha, model, k):
    a = torch.ones(k)
    for i in range(len(a)):
        a[i] = 1 / (i + 1)
    

def GetLyrActivation(model, inp_, layer_num):
    layer_names, layers = nested_children(model)
    lyrs = [layers[l] for l in layer_names]
    # Saving Activations
    model_activations = []
    for i in range(len(lyrs)):
        hk = lyrs[i].register_forward_hook(getActivation())
        pred = model(inp_)
        model_activations.append(hook_activations.cpu().detach())
        hk.remove()
    
    params = []
    for parameter in model.parameters():
        params.append(parameter)#.data)
        
    lyr_grad = (torch.autograd.grad(params[ik], inp_, # inp_ = results[img_num][0] 
                                         retain_graph=True)[0].requires_grad_(requires_grad=True))
    
    return lyr_grad

def LayerFracGradGL(inp_, alpha, h, model, lyr_num = 0, index = None, rightHand = True, N = 3):
    # Actual value should be N = (t-a)/h
    # print('GL')
    
    summation = 0.0
    attrGL = (inp_).clone().detach()
    inp = (inp_).clone().detach()
    for channel in range(len(inp[0])):
        for i_x in range(len(inp[0][0])):
            for i_y in range(len(inp[0][0][0])):
                if i_y % 5 == 0:
                    print('GL alpha =',str(alpha),'   Calculating pixel: (', 
                          str(channel), ',', str(i_x), ',', str(i_y), ')', 
                          "\t", 'Calculating... ', 
                          str(((i_x+1)*(i_y+1)*(channel+1))/((len(inp[0]))*(len(inp[0][0]))*(len(inp[0][0][0])))), 
                          '%\t\t\t|', end="\r")
                summation = 0
                t = inp#[0][0][i_x][i_y]
                # t = i * sample_step
                # N = int((t-a)/h)#math.ceil(alpha)#4
                for j in range(N):
                    torch.cuda.empty_cache()
                    model.zero_grad()
                    t_hj = inp.clone().detach()
                    
                    if rightHand == False:
                        t_hj[0][channel][i_x][i_y] -= ((h * (j - ((((alpha) + 1))/2))))
                    else:
                        t_hj[0][channel][i_x][i_y] -= (h * j)
                    with torch.no_grad():
                        # model.eval()
                        layer_names, layers = nested_children(model)
                        f_ = model(t_hj)#f(t - (h * j))
                        
                    
                    # if index == None:
                    #     f_ = torch.max(f_[0], 0)[0]
                    # else:
                    #     f_ = f_[0][index]
                    # fact_j = (treefactorial(j))#(math.factorial(j)) #(treefactorial(j))#
                    fact_j = (math.factorial(j))
                    gamma_bot = (math.gamma(alpha - j + 1))
                    gamma_top = (math.gamma(alpha + 1))
                    try:
                        alpha_j = ((gamma_top)) / (fact_j * gamma_bot)
                    except:
                        if j % (N-1) == 0 and i % (len(inp[0][0][0]) - 1) == 0:
                            print("Factorial too high")
                    
                    summation += (((-1) ** j) * float(alpha_j) * f_)
                
                attrGL[0][channel][i_x][i_y] = summation * (1 / (h ** alpha))
                # calculating (∂^α)a(x) / ∂x^α
    return attrGL

def FracGradChainRule(inp, alpha,)
