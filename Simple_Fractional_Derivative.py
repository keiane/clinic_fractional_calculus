# -*- coding: utf-8 -*-
from model import Model
import matplotlib.pyplot as plt
import torch, os
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
# from Integrated_Gradients import integrated_gradients
# import SmoothGrad as sg
# import Smooth_Integrated_Gradients as sig
# import Guided_Backprop_Integrated_Gradients as gbig
# import DeepLIFT_Integrated_Gradients as dlig
# from tensorflow.examples.tutorials.mnist import input_data
import math
import pandas as pd
from scipy.stats import spearmanr
from scipy.linalg import toeplitz
# import plotly.graph_objects as go
from scipy.integrate import quad
import differint.differint as df
from sympy import *
import sympy
from Simple_Frac_Functions import FracGradApproxGL, f, f_sym, returnDerivApprox
#import smoothed_functions as sf
#%matplotlib inline
# from captum.attr import (
#     InputXGradient,
#     Saliency,
#     GradientShap,
#     DeepLift,
#     DeepLiftShap,
#     IntegratedGradients,
#     GuidedGradCam,
#     LayerConductance,
#     NeuronConductance,
#     NoiseTunnel,
#     GuidedBackprop,
# )

#def f(x):
#    return x ** 2


def derivativeApprox(t, funct, h, n_deriv):
    t = np.array(t)
    if n_deriv == 0:
        deriv = funct(t)
    if n_deriv == 1:
        deriv = (funct(t + h) - funct(t - h)) / (2 * h)
    if n_deriv == 2:
        deriv = ((funct(t + h) - (2 * funct(t)) + funct(t - h)) / (h ** 2))
    if n_deriv == 3:
        deriv = ((funct(t + (2*h)) - (3 * funct(t + h)) + (3 * funct(t)) - funct(t - h)) / (h ** 3))
    return deriv

########## RL Method from https://arxiv.org/pdf/1208.2588.pdf ##########

def FracGradApproxRL(inp_, alpha, h , sample_step = 0.01, f_sympy = None):
    # N = math.ceil(alpha) + 3 # The approximation gets more accuracte as N -> inf
    N = 10
    if f_sympy == None:
        f_sympy = f
    x_ = symbols('x')
    f_ = f_sympy(x_)
    summation = 0.0
    attr = torch.tensor(inp_).clone().detach()
    attr = np.array(attr)
    inp = torch.tensor(inp_).clone().detach()
    inp = np.array(inp)
    for i in range(len(inp)):
            a = -h#inp_[i] - 0.00001
            summation = 0
            t = i * sample_step#inp[i]
#            print("inp_[", i, "] = ",inp_[i])
            for n in range(N):
#                print(n)
                C_ = C(n, alpha)
                dx = diff(f_, x_, n)
                x_deriv = dx.subs(x_,t)# * 0.000001
                
#                print("n = ", n, "  x_deriv = ", x_deriv)
                
                summation += C_ * ((t - a) ** (n - alpha)) * (x_deriv)
            
            attr[i] = summation
    return attr.tolist()

def C(n, alpha):
    C_ = (((((-1) ** (n - 1)) * alpha * math.gamma(n - alpha))/((math.gamma(1 - alpha) * math.gamma(n + 1)))) * (1 / math.gamma(n + 1 - alpha)))
    return C_

x_ = symbols('x')



#DF = df.RL(0.5, f)
#print(DF)
#plt.plot(DF)

sample_step = 0.01
xlim = [0, 200]
# num_sample_points = int((xlim[1] - xlim[0]) / sample_step)

alpha = 0.1

alpha_list0 = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.999]
#alpha_list0 = [0.01,0.2,0.4,0.6,0.8,0.999]
# alpha_list0 = [0.000001, 0.5, 0.99999]
alpha_list1 = [alpha + 1 for alpha in alpha_list0]
alpha_list2 = [alpha + 2 for alpha in alpha_list0]
alpha_list3 = [alpha + 3 for alpha in alpha_list0]

alpha_list = alpha_list0 #+ alpha_list1 #+ alpha_list2 + alpha_list3
# alpha_list = [0.01, 1.5]

h = 0.01
a = 0.0

num_sample_points = 300
f__, x__ = list(), list()
for i_f in range(num_sample_points):
    x = i_f * sample_step
    x__.append(x)
    f__.append(f(x))

fig, ax = plt.subplots()
plt.xlim([0, 300])
plt.ylim([0, 13])
# plt.plot(f__)
plt.plot(f__, label='f(x)')

for i in range(1, 2):
    dx = returnDerivApprox(x__, f, h, i, sample_step)
    # plt.plot(dx, label=str('f'+ str("'"*(i+1)) +'(x)'))
    plt.plot(dx, label=str('f^('+ str(i) +')(x)'))

plt.title("Integer Order Derivative Approximations of f(x) = x ** 5")
leg = plt.legend()
plt.show()

# plt.plot(f__, label='f(x)')
for i, alpha in enumerate(alpha_list):
    # DF = df.RL(alpha, f)
    # plt.plot(DF)    
    print('Iteration: ', i)
    #DF = df.RL(0.9999, f)
    #plt.plot(DF)
    # attr1 = FracGradApproxRL(f__, alpha, h) # 
    attr1 = FracGradApproxGL(f__, alpha, h, sample_step, a, rightHand = True)
    #attr = returnFracGrad(f_, alpha, h)

    plt.xlim([0, 300])
    plt.ylim([0, 13])
    plt.plot(attr1, label=str('f^('+ str(alpha) +')(x)'))

plt.title("GL Fractional Derivative Approximations of f(x) = x ** 2")
leg = plt.legend()
plt.show()

h = 0.0001
a = 0.0

# plt.plot(f__, label='f(x)')
for i, alpha in enumerate(alpha_list):
    
    print('Iteration: ', i)
    attr1 = FracGradApproxRL(f__, alpha, h)#, f_sympy = f_sym) # 
    
    plt.xlim([0, 300])
    plt.ylim([0, 13])
    plt.plot(attr1, label=str('f^('+ str(alpha) +')(x)'))

plt.title("RL Fractional Derivative Approximations of f(x) = x ** 2")
leg = plt.legend()
plt.show()

