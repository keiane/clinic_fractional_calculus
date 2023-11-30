# -*- coding: utf-8 -*-
"""
Created on Sun Nov 27 18:23:04 2022

@author: ianni
"""
import torch
import numpy as np
import math
from sympy import symbols, diff
from sympy import *
from decimal import Decimal
import sympy

########## GL Method from https://www.youtube.com/watch?v=opJVMMPb-EI ##########

def FracGradApproxGL(inp_, alpha, h, sample_step, a, rightHand = True):
    # N = math.ceil(alpha) + 1
    # Actual value should be N = (t-a)/h
    # N = 100#math.ceil(alpha)#4 # Actual value should be (t-a)/h
    # x_ = symbols('x')
    summation = 0.0
    attr = torch.tensor(inp_).clone().detach()
    attr = np.array(attr)
    inp = torch.tensor(inp_).clone().detach()
    inp = np.array(inp)
    for i in range(len(inp)):
            summation = 0
            t = i * sample_step
            N = int((t-a)/h)#math.ceil(alpha)#4
#            print("inp_[", i, "] = ",inp_[i])
            for j in range(N):
#                print(j)
                # f_ = f(t - (h * j))
                if rightHand == False:
                    f_ = f(t - (h * (j - int((((alpha) + 1))/2))))
                else:
                    f_ = f(t - (h * j))
                # print("j = ", j, "  f(x) = ", f_)
                # try:
                fact_j = (treefactorial(j))#(math.factorial(j))
                # if fact_j > 10000000000:
                #     fact_j = int(fact_j)
                #     alpha_j = (math.gamma(alpha + 1)) / (fact_j * int(math.ceil((math.gamma(alpha - j + 1)))))
                # else:
                gamma_bot = (math.gamma(alpha - j + 1))
                gamma_top = (math.gamma(alpha + 1))
                try:
                    alpha_j = ((gamma_top)) / (fact_j * gamma_bot)
                    # gamma_mult = Decimal(gamma_bot ** -1)
                    # alpha_j = (Decimal(gamma_top) * Decimal(gamma_mult)) / Decimal(fact_j)
                except:
                    if j % (N-1) == 0 and i % (len(inp) - 1) == 0:
                        print("Factorial too high")
                    # gamma_mult = int(1 / gamma_bot)
                    # gamma_bot = int(gamma_bot)
                    # fact_j = int(fact_j)
                    # alpha_j = int(gamma_top * gamma_mult) / int(fact_j)
                                   
                
                
                # alpha_j = gamma_top / (fact_j * gamma_bot)
                
                summation += (((-1) ** j) * float(alpha_j) * f_)
            
            attr[i] = summation * (1 / (h ** alpha))
    return attr.tolist()

def returnDerivApprox(inp_, funct, h, n_deriv, sample_step = 0.01, rightHand = True):
    # n: integer order for derivative
    # attr = img.clone().detach()
    # img_ = img.clone().detach()
    N = n_deriv+1
    summation = 0.0
    attr = torch.tensor(inp_).clone().detach()
    attr = np.array(attr)
    inp = torch.tensor(inp_).clone().detach()
    inp = np.array(inp)
    for i in range(len(inp)):
        summation = 0
        t = i * sample_step
        # attr[0][0][i][j] = 0
        for j_ in range(N):
            if rightHand == False:
                f_ = funct(t - (h * (j_ - int((((n_deriv) + 1))/2))))
            else:
                f_ = funct(t - (h * j_))
            
            fact_j = (math.factorial(j_)) #(treefactorial(j))
            gamma_bot = (math.factorial(n_deriv - j_))
            gamma_top = (math.factorial(n_deriv))
            alpha_j = ((gamma_top)) / (fact_j * gamma_bot)
            
            summation += (((-1) ** j_) * float(alpha_j) * f_)
            
        attr[i] = summation / (h ** n_deriv)
        
    return attr.tolist()

def range_prod(lo,hi):
    if lo+1 < hi:
        mid = (hi+lo)//2
        return range_prod(lo,mid) * range_prod(mid+1,hi)
    if lo == hi:
        return lo
    return lo*hi

def treefactorial(n):
    if n < 2:
        return 1
    return range_prod(1,n)





## Method to Approximate Integer Order Derivatives ##
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



def f(x):
    return x ** 2#x ** (x * 3)#math.exp(1) ** x#
   # return 0.25 * x * math.exp(1) ** (3 * x)#x ** 4#math.exp(1) ** (0.5 * x)#np.cos(x)#np.sin(x)#math.exp(1) ** (2 * x)#np.sin(x)#x ** 3#math.exp(1) ** (2 * x)#1 * x ** 3#math.exp(1) ** x#x**2

def f_sym(x):
    return sympy.cos(x)

# def f(x):
#    return x**3#math.exp(1) ** x#x**2

####################### UNUSED BELOW #######################

######### GL Discretization Method from https://arxiv.org/pdf/1608.03240.pdf #########

def returnFracGrad(inp_, alpha, h):
    #h = 0.001
    #alpha = 0.5
    k_ = 1.000001 #math.pi# * math.sqrt(2) #alpha 
    attr = torch.tensor(inp_).clone().detach()
    attr = np.array(attr)
    inp = torch.tensor(inp_).clone().detach()
    inp = np.array(inp)
    for i in range(len(inp)):
        #for j in range(len(img[0][0][0])):
#            print(i)
            start = 0
            a = inp[i] - (k_ * h)
            stop = h#((inp[i] - a) / h)
            #stop = k_
            attr[i] = summation(inp, alpha, h, start, stop, i)
            attr[i] = (attr[i] * (1 / (h ** alpha)))# * (1 / (k_ - stop + 1))
            #print(attr[0][0][i][j])
    return attr.tolist()

def summation(inp, alpha, h, start, stop, i_s):
    total = 0.00
    n = ((stop - start)) #/ h
    #img_ = img.clone().detach()
#    print("Summation ", i_s)
#    print("n ", n)
    inp2 = torch.tensor(inp).clone().detach()
    for j_ in range(int(n)):
        j_f = float(j_)
        inp2[i_s] = inp2[i_s] - (j_f * h)
        pred = f(inp2[i_s])#model_ft(img2)
        #pred = pred[0][labels[itr]]#torch.max(pred[0], 0)[0]
        w_val = w(j_f, alpha)
#        print("w ", w_val)
        
        total += w_val * pred
    return total / n

def w(j_w, alpha):
#    print("w ", j_w)
    if j_w >= 0:
        for k in range(int(j_w) + 1):
            if k == 0:
                w_out = 1.0
            else:
                w_out += ((1.0 - ((alpha + 1.0) / float(k))) * w_out)
    else:
        w_out = 0.0
#    print(w_out)
    return w_out



######### RL Method from ??? #########

def integralApproxRL(inp, begin, end, alpha, h, i_i, delta = 0):
    integral_total = 0.0
    n = math.ceil(alpha) ## this value is alpha rounded up
    begin_int = int((begin / h))
    end_int = int((end / h))
    for t in range(begin_int, end_int, 1):
        t_ = t * h
#        t_ = inp[t]
        pred = f(t_)
        if delta == 1:
            pred = f(t_ + h)
        if delta == 2:
            pred = f(t_ - h)
#        d_f = derivativeApprox(t_, f, h, n)
        integral_total += pred * ((end - t_) ** (n - alpha - 1))
    
    integral_total = integral_total / ((end - begin) / h)
    return integral_total

def FracGradRL(inp_, alpha, h):
    n = math.ceil(alpha)
    attr = torch.tensor(inp_).clone().detach()
    attr = np.array(attr)
    inp = torch.tensor(inp_).clone().detach()
    inp = np.array(inp)
    for i in range(len(inp)):
#            print(i)
            a = 0#(inp[i]) - (k_ * h)#inp[i] - (k_ * h)
            start = a
            t = inp[i]
            stop = t
#            attr[i] = integralApproxRL(inp, start, stop, alpha, h, i)
            deriv = t
            for j in range(n):
                inp1 = inp
                inp1[i] = inp1[i] + h
                int1 = integralApproxRL(inp1, start, stop, alpha, h, i, delta = 1)
                inp2 = inp
                inp2[i] = inp2[i] - h
                int2 = integralApproxRL(inp2, start, stop, alpha, h, i, delta = 2)
                deriv = (int1 - int2) / (2 * h)
#            print(attr[i])
            attr[i] = deriv
            attr[i] = (attr[i] * (1 / math.gamma(n - alpha)))
    return attr.tolist()

######### Caputo Method #########

def integralApproxCaputo(inp, begin, end, alpha, h, i_i):
    integral_total = 0.0
    n = math.ceil(alpha) ## this value is alpha rounded up
    begin_int = int((begin / h))
    end_int = int((end / h))
    for t in range(begin_int, end_int, 1):
        t_ = t * h
        d_f = derivativeApprox(t_, f, h, n)
        integral_total += (d_f / ((end - t_) ** (alpha - n + 1)))
    
    integral_total = integral_total / ((end - begin) / h)
    return integral_total

def returnFracGradCaputo(inp_, alpha, h):
    #h = 0.001
    #alpha = 0.5
    n = math.ceil(alpha)
    attr = torch.tensor(inp_).clone().detach()
    attr = np.array(attr)
    inp = torch.tensor(inp_).clone().detach()
    inp = np.array(inp)
    for i in range(len(inp)):
#            print(i)
            a = -0.0 #(inp[i]) - (k_ * h)#inp[i] - (k_ * h)
            start = a
            t = inp[i]
            stop = t
            attr[i] = integralApproxCaputo(inp, start, stop, alpha, h, i)
#            print(attr[i])
            attr[i] = (attr[i] * (1 / math.gamma(n - alpha)))
    return attr.tolist()
