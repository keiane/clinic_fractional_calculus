# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 10:24:38 2021

@author: ianni
"""

import torch
import numpy as np
# import model as m
import math
from methods.Gradients import *
from methods.plotting_functions import *
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# criterion = torch.nn.CrossEntropyLoss()
# model_ft = m.Model()

# model_ft.load_state_dict(torch.load("model_weights/model_ROAR_baseline.dth", map_location=device))
# model_ft.eval()
# model_ft.to(device)

def f(x):
    return ( x ** 2 )

########## GL Method from https://www.youtube.com/watch?v=opJVMMPb-EI ##########

def FracGradApproxGL(inp_, alpha, h, model, index = None, rightHand = True, N = 3):
    # N = math.ceil(alpha) + 1
    # Actual value should be N = (t-a)/h
    #math.ceil(alpha)#4 # Actual value should be (t-a)/h
    # x_ = symbols('x')
    # print('GL')
    summation = 0.0
    attrGL = (inp_).clone().detach()
    inp = (inp_).clone().detach()
    p_comp, percent_complete = 0.0, 0.0
    for channel in range(len(inp[0])):
        for i_x in range(len(inp[0][0])):
            for i_y in range(len(inp[0][0][0])):
                percent_complete = (((i_x+1)*(i_y+1)+((channel)*len(inp[0][0])*len(inp[0][0][0]))) / ((len(inp[0]))*(len(inp[0][0]))*(len(inp[0][0][0]))))
                if percent_complete > p_comp:
                        p_comp = percent_complete
                if i_y % 5 == 0:
                    # print('\033[FGL alpha =',str(alpha),'   Calculating pixel: (', str(channel), ',', str(i_x), ',', str(i_y), ')', "\t", 
                    #     'Calculating... ', str(((i_x+1)*(i_y+1)*(channel+1)) / ((len(inp[0]))*(len(inp[0][0]))*(len(inp[0][0][0])))), '%\t\t\t|', end="\r")
                    # sys.stdout.flush()
                    print('\033[FGL alpha =',str(alpha),'   Calculating pixel: (', str(channel), ',', str(i_x), ',', str(i_y), ')', "\t", 
                            'Calculating... ', str(p_comp), 
                            '%                                     |', end="\r")
                    sys.stdout.flush()
                summation = 0.0
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
                        f_ = model(t_hj)#f(t - (h * j))
                    if index == None:
                        f_ = torch.max(f_[0], 0)[0]
                    else:
                        f_ = f_[0][index]
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


########## RL Method from https://arxiv.org/pdf/1208.2588.pdf ##########
def FracGradApproxRL(inp_, alpha, h, grad_list):# d0x, dx, d2x, d3x):
    #n = math.ceil(alpha)
    # print('RL')
    N = len(grad_list)#4
#    x_ = symbols('x')
#    f_ = f(x_)
    summation = 0.0
    attr = (inp_).clone().detach()
    inp = (inp_).clone().detach()
    for channel in range(len(inp[0])):
        for i in range(len(inp[0][0])):
            # print(i, end = "\r")
            for j in range(len(inp[0][0][0])):
                #a = 0#-h
                summation = 0
                a, t = -(h * (alpha * 1)), h * (alpha * 1)#inp[0][0][i][j] * h#h# * 2#N#alpha * 2
                #t = torch.mean(inp_[0][0])#+h#((i * len(inp[0][0][0])) + j) * h#pred1#inp_[0][0][i][j]#(i * len(inp[0][0][0])) + j#inp[i]
                for n, d_x in enumerate(grad_list):#[d0x, dx, d2x, d3x]):
    #                d_x = VisualizeImageGrayscale(d_x)
                    C_ = C(n, alpha)
                    x_deriv = d_x[0][channel][i][j]
                    
    #                print("n = ", n, "  x_deriv = ", x_deriv)
                    
                    summation += C_ * ((t - a) ** (n - alpha)) * float(x_deriv)
                
                attr[0][channel][i][j] = summation
    return attr

def FracOnePixelGradApproxRL(inp_, alpha, h, grad_list, channel = 0, i = 0, j = 0):# d0x, dx, d2x, d3x):
    #n = math.ceil(alpha)
    # print('RL')
    N = len(grad_list)#4
#    x_ = symbols('x')
#    f_ = f(x_)
    summation = 0.0
    attr = 0
    inp = (inp_).clone().detach()
    # for channel in range(len(inp[0])):
    #     for i in range(len(inp[0][0])):
    #         # print(i, end = "\r")
    #         for j in range(len(inp[0][0][0])):
    #a = 0#-h
    summation = 0
    a, t = -(h * (alpha * 1)), h * (alpha * 1)#inp[0][0][i][j] * h#h# * 2#N#alpha * 2
    #t = torch.mean(inp_[0][0])#+h#((i * len(inp[0][0][0])) + j) * h#pred1#inp_[0][0][i][j]#(i * len(inp[0][0][0])) + j#inp[i]
    for n, d_x in enumerate(grad_list):#[d0x, dx, d2x, d3x]):
#                d_x = VisualizeImageGrayscale(d_x)
        C_ = C(n, alpha)
        # print(d_x)
        x_deriv = d_x#[0][channel][i][j]
        
#                print("n = ", n, "  x_deriv = ", x_deriv)
        
        summation += C_ * ((t - a) ** (n - alpha)) * float(x_deriv)
    
    attr = summation
    return attr

def C(n, alpha):
    C_ = (((((-1) ** (n - 1)) * alpha * math.gamma(n - alpha))/((math.gamma(1 - alpha) * math.gamma(n + 1)))) * (1 / math.gamma(n + 1 - alpha)))
    return C_



def returnZeroDerivGrad(img, model_ft):
#    h = 0.00001
    attr = img.clone().detach()
    img_ = img.clone().detach()
#    img_[0][0][i][j] -= (h)
    pred = model_ft(img_)
    pred = torch.max(pred[0], 0)[0]
    for i in range(len(img[0][0])):
        for j in range (len(img[0][0][0])):    
            attr[0][0][i][j] = (pred)
            
    return attr

class Frac_Grads:
# To make faster have the f_ values save so that they do not need to be calculated multiple times
    def __init__(self, img, model, h, index=None, rightHand = True):
        self.img = img
        self.model = model
        self.h = h
        self.index = index
        self.rightHand = rightHand
        pred = self.model(img)
        if self.index == None:
            pred = torch.max(pred[0], 0)[0]
        else:
            pred = pred[0][self.index]
        self.zero_t = torch.ones_like(img.clone().detach())#/img.clone().detach()# * 0.0
        # max_n = 5
        # print(self.zero_t)
        self.f_saved = [[torch.add(self.zero_t, float(pred - 1))],]# * int(max_n + 1)
        self.f_ticked = [[(self.zero_t)],]# * int(max_n + 1)
        # print(self.f_saved)
    
    def incLenFList(self, N):
        N += 4
        # itr = 1
        # while N >= len(self.f_saved):
        for itr in range(N + 1):
            self.f_saved.append([self.zero_t,])
            self.f_ticked.append([torch.zeros_like(self.zero_t),])
            # while N >= len(self.f_saved[itr]):
            for itr_ in range(N + 1):
                self.f_saved[itr].append(self.zero_t)
                self.f_ticked[itr].append(torch.zeros_like(self.zero_t))
            # itr += 1
        return
    
    def returnDerivApprox(self, n):
        # n: integer order for derivative
        img = self.img
        h = self.h
        N = n + 1
        attr = torch.zeros_like(img.clone().detach())
        img_ = img.clone().detach()
        # img_ = img_ + (torch.ones_like(img_) * 0.01)
        # print(self.f_ticked)
        self.incLenFList(N = N)
        p_comp, percent_complete = 0.0, 0.0
        
        for channel in range(len(img[0])):
            for i in range(len(img[0][0])):
                for j in range(len(img[0][0][0])):
                    percent_complete = ((i+1)*(j+1)+((channel)*len(img[0][0])*len(img[0][0][0]))) / ((len(img[0]))*(len(img[0][0]))*(len(img[0][0][0])))
                    if percent_complete > p_comp:
                        p_comp = percent_complete
                    if j % 5 == 0:
                        # print('n =',str(n),'   Calculating pixel: (', str(channel), ',', str(i), ',', str(j), ')', "\t", 
                        #     'Calculating... ', str(p_comp), 
                        #     '%                                     |', end="\r")
                        print('\033[FInteger order n =',str(n),'   Calculating pixel: (', str(channel), ',', str(i), ',', str(j), ')', "\t", 
                            'Calculating... ', str(p_comp), 
                            '%                                     |', end="\r")
                        sys.stdout.flush()
                    attr[0][channel][i][j] = 0.0
                    for j_ in range(N):
                        torch.cuda.empty_cache()
                        self.model.zero_grad()
                        img_ = img.clone().detach()
                        # print(np.shape(self.f_ticked[0][0]))
                        
                        if self.f_ticked[n][j_][0][channel][i][j] == 0:
                            if self.rightHand:
                                img_[0][channel][i][j] -= (h * j_)
                            else:
                                img_[0][channel][i][j] -= ((h * (j_ - int(N/2))))
                            ###### Get Model Predictions ######
                            with torch.no_grad():
                                f_ = self.model(img_).clone().detach()                
                            if self.index == None:
                                f_ = torch.max(f_[0], 0)[0]
                            else:
                                f_ = f_[0][self.index]
                            ###################################
                            self.f_saved[n][j_][0][channel][i][j] = float(0.0)
                            self.f_saved[n][j_][0][channel][i][j] = f_.clone().detach()
                            self.f_ticked[n][j_][0][channel][i][j] = 1
                            # print(f_)
                            # print(self.f_saved[n][j_][0][channel][i][j])
                        else:
                            # print(self.f_saved[n][j_][0][channel][i][j])
                            f_ = self.f_saved[n][j_][0][channel][i][j].clone().detach()
                        
                        
                        # fact_j = (treefactorial(j))#(math.factorial(j_)) #(treefactorial(j))
                        # gamma_bot = (treefactorial(n - j_))#(math.factorial(n - j_))
                        # gamma_top = (treefactorial(n))#(math.factorial(n))
                        ############
                        fact_j = float(math.factorial(j_))
                        gamma_bot = float(math.factorial(n - j_))
                        gamma_top = float(math.factorial(n))

                        alpha_j = ((gamma_top)) / (fact_j * gamma_bot)
                        
                        attr[0][channel][i][j] += (((-1) ** float(j_)) * float(alpha_j) * f_)
                            
                    # print(self.f_saved)
        print('')
        # print(self.f_saved)
        attr = attr / (h ** n)
                
        return attr
    
    def returnDerivOnePixelApprox(self, n, channel = 0, i = 0, j = 0):
        # n: integer order for derivative
        img = self.img
        h = self.h
        N = n + 1
        # attr = torch.zeros_like(img.clone().detach())
        img_ = img.clone().detach()
        # print(self.f_ticked)
        self.incLenFList(N = N)
        p_comp, percent_complete = 0.0, 0.0
        # for channel in range(len(img[0])):
        #     for i in range(len(img[0][0])):
        #         for j in range(len(img[0][0][0])):
        percent_complete = ((i+1)*(j+1)+((channel)*len(img[0][0])*len(img[0][0][0]))) / ((len(img[0]))*(len(img[0][0]))*(len(img[0][0][0])))
        if percent_complete > p_comp:
            p_comp = percent_complete
        if j % 5 == 0:
            # print('n =',str(n),'   Calculating pixel: (', str(channel), ',', str(i), ',', str(j), ')', "\t", 
            #     'Calculating... ', str(p_comp), 
            #     '%                                     |', end="\r")
            print('\033[FInteger order n =',str(n),'   Calculating pixel: (', str(channel), ',', str(i_x), ',', str(i_y), ')', "\t", 
                            'Calculating... ', str(p_comp), 
                            '%                                     |', end="\r")
            sys.stdout.flush()
        attr = 0
        for j_ in range(N):
            torch.cuda.empty_cache()
            self.model.zero_grad()
            img_ = img.clone().detach()
            # print(np.shape(self.f_ticked[0][0]))

            if self.f_ticked[n][j_][0][channel][i][j] == 0:
                if self.rightHand:
                    img_[0][channel][i][j] -= (h * j_)
                else:
                    img_[0][channel][i][j] -= ((h * (j_ - int(N/2))))
                ###### Get Model Predictions ######
                with torch.no_grad():
                    f_ = self.model(img_).clone().detach()                
                if self.index == None:
                    f_ = torch.max(f_[0], 0)[0]
                else:
                    f_ = f_[0][self.index]
                ###################################
                self.f_saved[n][j_][0][channel][i][j] = float(0.0)
                self.f_saved[n][j_][0][channel][i][j] = f_.clone().detach()
                self.f_ticked[n][j_][0][channel][i][j] = 1
                # print(f_)
                # print(self.f_saved[n][j_][0][channel][i][j])
            else:
                # print(self.f_saved[n][j_][0][channel][i][j])
                f_ = self.f_saved[n][j_][0][channel][i][j].clone().detach()


            # fact_j = (treefactorial(j))#(math.factorial(j_)) #(treefactorial(j))
            # gamma_bot = (treefactorial(n - j_))#(math.factorial(n - j_))
            # gamma_top = (treefactorial(n))#(math.factorial(n))
            ############
            fact_j = (math.factorial(j_))
            gamma_bot = (math.factorial(n - j_))
            gamma_top = (math.factorial(n))

            alpha_j = ((gamma_top)) / (fact_j * gamma_bot)
            
            attr += (((-1) ** j_) * float(alpha_j) * f_)
        # print(self.f_saved)
        print('')
        # print(self.f_saved)
        attr = attr / (h ** n)
                
        return attr

    ########## GL Method from https://www.youtube.com/watch?v=opJVMMPb-EI ##########

    def FracGradApproxGL(self, alpha, N = 4):
        # Actual value should be N = (t-a)/h
        inp_ = self.img
        h = self.h
        n = N - 1
        self.incLenFList(N = N)
        summation = 0.0
        attrGL = (inp_).clone().detach()
        inp = (inp_).clone().detach()
        p_comp, percent_complete = 0.0, 0.0
        for channel in range(len(inp[0])):
            for i_x in range(len(inp[0][0])):
                for i_y in range(len(inp[0][0][0])):
                    percent_complete = (((i_x+1)*(i_y+1)+((channel)*len(inp[0][0])*len(inp[0][0][0]))) / ((len(inp[0]))*(len(inp[0][0]))*(len(inp[0][0][0]))))
                    if percent_complete > p_comp:
                        p_comp = percent_complete
                    if i_y % 5 == 0:
                        print('\033[FGL alpha =',str(alpha),'   Calculating pixel: (', str(channel), ',', str(i_x), ',', str(i_y), ')', "\t", 
                            'Calculating... ', str(p_comp), 
                            '%                                     |', end="\r")
                        sys.stdout.flush()
                    summation = 0
                    t = inp#[0][0][i_x][i_y]
                    # t = i * sample_step
                    # N = int((t-a)/h)#math.ceil(alpha)#4
                    for j in range(N):
                        torch.cuda.empty_cache()
                        self.model.zero_grad()
                        t_hj = inp.clone().detach()
                        
                        if self.f_ticked[n][j][0][channel][i_x][i_y] == 0:
                            if self.rightHand == False:
                                t_hj[0][channel][i_x][i_y] -= ((h * (j - ((((alpha) + 1))/2))))
                            else:
                                t_hj[0][channel][i_x][i_y] -= (h * j)
                            with torch.no_grad():
                                f_ = self.model(t_hj).clone().detach()#f(t - (h * j))
                            if self.index == None:
                                f_ = torch.max(f_[0], 0)[0]
                            else:
                                f_ = f_[0][self.index]
                            self.f_saved[n][j][0][channel][i_x][i_y] = float(0.0)
                            self.f_saved[n][j][0][channel][i_x][i_y] = f_.clone().detach()
                            self.f_ticked[n][j][0][channel][i_x][i_y] = 1
                        else:
                            # print(self.f_saved[n][j_][0][channel][i][j])
                            f_ = self.f_saved[n][j][0][channel][i_x][i_y].clone().detach()
                        ######################################

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
    
    def FracOnePixelGradApproxGL(self, alpha, N = 4, channel = 0, i_x = 0, i_y = 0):
        # Actual value should be N = (t-a)/h
        inp_ = self.img
        h = self.h
        n = N - 1
        summation = 0.0
        attrGL = 0
        inp = (inp_).clone().detach()
        p_comp, percent_complete = 0.0, 0.0
        # for channel in range(len(inp[0])):
        #     for i_x in range(len(inp[0][0])):
        #         for i_y in range(len(inp[0][0][0])):
        percent_complete = (((i_x+1)*(i_y+1)+((channel)*len(inp[0][0])*len(inp[0][0][0]))) / ((len(inp[0]))*(len(inp[0][0]))*(len(inp[0][0][0]))))
        if percent_complete > p_comp:
            p_comp = percent_complete
        if i_y % 5 == 0:
            # print('\033[FGL alpha =',str(alpha),'   Calculating pixel: (', str(channel), ',', str(i_x), ',', str(i_y), ')', "\t", 
            #     'Calculating... ', str(p_comp), 
            #     '%                                     |', end="\r")
            print('\033[FGL alpha =',str(alpha),'   Calculating pixel: (', str(channel), ',', str(i_x), ',', str(i_y), ')', "\t", 
                            'Calculating... ', str(p_comp), 
                            '%                                     |', end="\r")
            sys.stdout.flush()
        summation = 0
        t = inp#[0][0][i_x][i_y]
        # t = i * sample_step
        # N = int((t-a)/h)#math.ceil(alpha)#4
        for j in range(N):
            torch.cuda.empty_cache()
            self.model.zero_grad()
            t_hj = inp.clone().detach()
            if self.rightHand == False:
                t_hj[0][channel][i_x][i_y] -= ((h * (j - ((((alpha) + 1))/2))))
            else:
                t_hj[0][channel][i_x][i_y] -= (h * j)
            with torch.no_grad():
                f_ = self.model(t_hj).clone().detach()#f(t - (h * j))
            if self.index == None:
                f_ = torch.max(f_[0], 0)[0]
            else:
                f_ = f_[0][self.index]
            ######################################

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
        
        attrGL = summation * (1 / (h ** alpha))
        # calculating (∂^α)a(x) / ∂x^α
        return attrGL

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



######################################

def returnDerivApprox_Efficient(img, model, n, h, index=None, rightHand = True):
    if n == 0:
        attr = returnZeroDerivGrad(img, model)
    else:
        if rightHand:
            if n == 1:
                attr = returnFirstDerivGradRightHand(img, h, model)
            if n == 2:
                attr = returnSecondDerivGradRightHand(img, h, model)
            if n == 3:
                attr = returnThirdDerivGradRightHand(img, h, model)
        else:
            if n == 1:
                attr = returnFirstDerivGrad(img, h, model)
            if n == 2:
                attr = returnSecondDerivGrad(img, h, model)
            if n == 3:
                attr = returnThirdDerivGrad(img, h, model)

    return attr

def getModelPredictions(img_, model, index=None):
    ###### Get Model Predictions ######
    torch.cuda.empty_cache()
    model.zero_grad()
    with torch.no_grad():
        f_ = model(img_.clone().detach())                
    if index == None:
        f_ = torch.max(f_[0], 0)[0]
    else:
        f_ = f_[0][index]
    ###################################
    return f_

def returnThirdDerivGrad(img, h, model_ft, index=None):
    #    h = 0.00001
    attr = img.clone().detach()
    pred = model_ft(img)
    # pred = torch.max(pred[0], 0)[0]
    pred = getModelPredictions(img, model_ft, index)
    for i in range(len(img[0][0])):
        for j in range (len(img[0][0][0])):    
            
            img_ = img.clone().detach()
            img_[0][0][i][j] += (h)
            pred1 = model_ft(img_)
            # pred1 = torch.max(pred1[0], 0)[0]
            pred1 = getModelPredictions(img_, model_ft, index)
            img_ = img.clone().detach()
            img_[0][0][i][j] -= (h)
            pred2 = model_ft(img_)
            # pred2 = torch.max(pred2[0], 0)[0]
            pred2 = getModelPredictions(img_, model_ft, index)
            img_ = img.clone().detach()
            img_[0][0][i][j] += (2 * h)
            pred3 = model_ft(img_)
            # pred3 = torch.max(pred3[0], 0)[0]
            pred3 = getModelPredictions(img_, model_ft, index)
            
            attr[0][0][i][j] = ((pred3 - (3 * pred1) + (3 * pred) - pred2) / (h ** 3))
            #attr[0][0][i][j] = ((pred1 - pred2 - (2 * pred3)) / (4 * (h ** 2)))
            
    return attr

def returnThirdDerivGradRightHand(img, h, model_ft):
    #    h = 0.00001
    attr = img.clone().detach()
    pred = model_ft(img)
    # pred = torch.max(pred[0], 0)[0]

    for i in range(len(img[0][0])):
        for j in range (len(img[0][0][0])):    
            
            img_ = img.clone().detach()
            img_[0][0][i][j] += (h)
            pred1 = model_ft(img_)
            pred1 = torch.max(pred1[0], 0)[0]
            img_ = img.clone().detach()
            img_[0][0][i][j] -= (h)
            pred2 = model_ft(img_)
            pred2 = torch.max(pred2[0], 0)[0]
            img_ = img.clone().detach()
            img_[0][0][i][j] -= (2 * h)
            pred3 = model_ft(img_)
            pred3 = torch.max(pred3[0], 0)[0]
            
            attr[0][0][i][j] = ((pred1 - (3 * pred) + (3 * pred2) - pred3) / (h ** 3))
            #attr[0][0][i][j] = ((pred1 - pred2 - (2 * pred3)) / (4 * (h ** 2)))
            
    return attr

def returnSecondDerivGrad(img, h, model_ft):
#    h = 0.00001
    attr = img.clone().detach()
    pred3 = model_ft(img)
    pred3 = torch.max(pred3[0], 0)[0]
    for i in range(len(img[0][0])):
        for j in range (len(img[0][0][0])):    
            
            img_ = img.clone().detach()
            img_[0][0][i][j] += (h)
            pred1 = model_ft(img_)
            pred1 = torch.max(pred1[0], 0)[0]
            img_ = img.clone().detach()
            img_[0][0][i][j] -= (h)
            pred2 = model_ft(img_)
            pred2 = torch.max(pred2[0], 0)[0]
            
            attr[0][0][i][j] = ((pred1 - (2 * pred3) + pred2) / (h ** 2))
            #attr[0][0][i][j] = ((pred1 - pred2 - (2 * pred3)) / (4 * (h ** 2)))
            
    return attr

def returnSecondDerivGradRightHand(img, h, model_ft):
#    h = 0.00001
    attr = img.clone().detach()
    pred3 = model_ft(img)
    pred3 = torch.max(pred3[0], 0)[0]
    for i in range(len(img[0][0])):
        for j in range (len(img[0][0][0])):    
            
            img_ = img.clone().detach()
            img_[0][0][i][j] -= (h)
            pred1 = model_ft(img_)
            pred1 = torch.max(pred1[0], 0)[0]
            img_ = img.clone().detach()
            img_[0][0][i][j] -= (2*h)
            pred2 = model_ft(img_)
            pred2 = torch.max(pred2[0], 0)[0]
            
            attr[0][0][i][j] = ((pred3 - (2 * pred1) + pred2) / (h ** 2))
            #attr[0][0][i][j] = ((pred1 - pred2 - (2 * pred3)) / (4 * (h ** 2)))
            
    return attr

def returnFirstDerivGrad(img, h, model_ft):
#    h = 0.00001
    attr = img.clone().detach()
    img_ = img.clone().detach()
#    img_[0][0][i][j] -= (h)
    pred2 = model_ft(img_)
    pred2 = torch.max(pred2[0], 0)[0]
    for i in range(len(img[0][0])):
        for j in range (len(img[0][0][0])):    
            
            img_ = img.clone().detach()
            img_[0][0][i][j] += (h)
            pred1 = model_ft(img_)
            pred1 = torch.max(pred1[0], 0)[0]
            
            
            attr[0][0][i][j] = ((pred1 - pred2) / h)
            
    return attr

def returnFirstDerivGradRightHand(img, h, model):
#    h = 0.00001
    attr = img.clone().detach()
    img_ = img.clone().detach()
#    img_[0][0][i][j] -= (h)
    pred2 = model_ft(img_)
    pred2 = torch.max(pred2[0], 0)[0]
    for i in range(len(img[0][0])):
        for j in range (len(img[0][0][0])):    
            
            img_ = img.clone().detach()
            img_[0][0][i][j] -= (h)
            pred1 = model_ft(img_)
            pred1 = torch.max(pred1[0], 0)[0]
            
            attr[0][0][i][j] = ((pred2 - pred1) / h)
            
    return attr