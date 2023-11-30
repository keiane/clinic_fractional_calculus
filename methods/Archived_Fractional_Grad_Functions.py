# -*- coding: utf-8 -*-
import math
from Fractional_Gradients import f
from methods.Gradients import *
from methods.plotting_functions import *


def returnFracGrad(img, alpha):
    h = 0.0001
    #alpha = 0.5
    k_ = 15 #alpha 
    attr = img.clone().detach()
    img_ = img.clone().detach()
    img2 = img.clone().detach()
    for i in range(len(img[0][0])):
        for j in range(len(img[0][0][0])):
            start = 0
            a = img[0][0][i][j] - (h * k_)
            stop = ((img_[0][0][i][j] - a) / h)
            stop = k_
            attr[0][0][i][j] = summation(img2, alpha, h, start, stop, i, j) * (1 / (h ** alpha))# * (1 / (k_ - stop + 1))
            #print(attr[0][0][i][j])
    return attr

def returnFracGradDF(img, alpha):
    h = 0.0001
    #alpha = 0.5
    k_ = 15 #alpha 
    attr = img.clone().detach()
    img_ = img.clone().detach()
    img2 = img.clone().detach()
    for i in range(len(img[0][0])):
        for j in range(len(img[0][0][0])):
            
            # print(torch.tensor(df.RL(alpha, model_ft, attr, i, j))[i][j])
            attr[0][0][i][j] = torch.tensor(df.GL(alpha, model_ft, attr, i, j))[i][j]#summation(img2, alpha, h, start, stop, i, j) * (1 / (h ** alpha))# * (1 / (k_ - stop + 1))
            #print(attr[0][0][i][j])
    return attr
    
def returnFG(img, alpha, first_grad_est):
    if alpha < 1.0:
        attr_out = (img * (1 - alpha)) + (first_grad_est * alpha)
    if alpha >= 1.0:
        attr_out = ((VisualizeImageGrayscale(first_grad_est) * (2 - alpha)) + 
                    (VisualizeImageGrayscale(second_grad_est) * (alpha - 1)))
    return attr_out

def derivativeApprox(t, funct, h, n_deriv):
    deriv = t
    for j in range(n_deriv):
        deriv = (funct(deriv + h) - funct(deriv - h)) / (2 * h)
    return deriv

def integralApproxCaputo(inp, labels, begin, end_, alpha, h, i_i, j_i, itr):
    integral_total = 0.0
    n = math.ceil(alpha) ## this value is alpha rounded up
    end = end_# + 1.0
    begin_int = int((begin / h))
    end_int = int(math.ceil(end_ / h))# + 1)
    print("\rstart: ", begin_int, "    stop: ", end_int, end = "")
#    inp_ = inp.clone().detach()
    for t in range(begin_int, end_int, 1):
        t_ = t * h
        ## Approximate Derivative ##
#        d_f = derivativeApprox(t_, model_ft.forward, h, n)
        
        inp1 = inp.clone().detach()
        inp1[0][0][i_i][j_i] = inp1[0][0][i_i][j_i] + (t_ * h)
        inp2 = inp.clone().detach()
        inp2[0][0][i_i][j_i] = inp2[0][0][i_i][j_i] - (t_ * h)
#        print("line 182: ", inp1[0][0][i_i][j_i])
        pred1 = model_ft(inp1)
        pred1 = pred1[0][labels[itr]]#torch.max(pred[0], 0)[0]
        pred2 = model_ft(inp2)
        pred2 = pred2[0][labels[itr]]#torch.max(pred[0], 0)[0]
#        print("line 187: ", pred1)
        d_f = (pred2 - pred1) / (2 * h)
#        print("line 189: ", d_f)
        integral_total += (d_f / ((end - t_) ** (alpha - n + 1)))
#        print("line 191: ", integral_total)
#    print("line 192: ", ((float(end) - float(begin)) / h))
    integral_total = integral_total / ((float(end) - float(begin)) / h)
#    print("line 194: ", integral_total)
#    if torch.is_nan(integral_total):
#        integral_total = torch.tensor([0.0])
    return integral_total

def returnFracGradCaputo(inp_, labels, alpha, h):
    #alpha = 0.5
    n = math.ceil(alpha)
    attr = (inp_).clone().detach()
#    attr = np.array(attr)
    inp = (inp_).clone().detach()
#    inp = np.array(inp)
    attr = inp_.clone().detach()
    for i in range(len(inp[0][0])):
        # print("\n", i)
        for j in range(len(inp[0][0][0])):
            a = -0.7#-0.7 #(inp[i]) - (k_ * h)#inp[i] - (k_ * h)
            start = float(a)
            t = inp[0][0][i][j] + h#0.01
            stop = float(t)
            attr[0][0][i][j] = integralApproxCaputo(inp, labels, start, stop, alpha, h, i, j)
#            print("line 214: ", attr[0][0][i][j])
            attr[0][0][i][j] = (attr[0][0][i][j] * (1 / math.gamma(n - alpha)))
    return attr#.tolist()

def integralApprox(img, labels, begin, end, alpha, h, i_i, j_i, itr, model_ft):
    integral_total = 0.0
    for t in range(begin, end, h):
        img[0][0][i_i][j_i] = t
        pred = model_ft(img)
        pred = pred[0][labels[itr]]
        integral_total += ((end - t) ** (math.ceil(alpha) - alpha - 1)) * pred
    integral_total = integral_total / ((end - begin) / h)
    

def summation(img, alpha, h, start, stop, i_s, j_s):
    total = 0.00
    n = stop - start
    #img_ = img.clone().detach()
    img2 = img.clone().detach()
    for j_ in range(int(n)):
        j_f = float(j_)
        img2[0][0][i_s][j_s] = img[0][0][i_s][j_s] - (j_f * h)
        pred = f(img2[0][0][i_s][j_s])
        #pred = model_ft(img2)
        #pred = pred[0][labels[itr]]#torch.max(pred[0], 0)[0]
        w_val = w(j_f, alpha)
        
        total += w_val * pred
    return total

def w(j_w, alpha):
    if j_w >= 0:
        for k in range(int(j_w) + 1):
            if k == 0:
                w_out = 1.0
            else:
                w_out += ((1.0 - ((alpha + 1.0) / float(k))) * w_out)
    else:
        w_out = 0.0
    return w_out

def returnThirdDerivGrad(img, h):
    #    h = 0.00001
    attr = img.clone().detach()
    pred = model_ft(img)
    pred = torch.max(pred[0], 0)[0]
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
            img_[0][0][i][j] += (2 * h)
            pred3 = model_ft(img_)
            pred3 = torch.max(pred3[0], 0)[0]
            
            attr[0][0][i][j] = ((pred3 - (3 * pred1) + (3 * pred) - pred2) / (h ** 3))
            #attr[0][0][i][j] = ((pred1 - pred2 - (2 * pred3)) / (4 * (h ** 2)))
            
    return attr

def returnSecondDerivGrad(img, h):
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

def returnSecondDerivGradRightHand(img, h):
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

def returnFirstDerivGrad(img, h):
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

def returnFirstDerivGradRightHand(img, h):
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


