# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:14:31 2023

@author: ianni
"""
### CODE FOR Tiny ImageNet ATTRIBUTION - Repurposed for Toy Problem ###

import matplotlib
# matplotlib.use("Agg")

# from model import Model
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
from methods.Integrated_Gradients import integrated_gradients
import methods.SmoothGrad as sg
import math
import pandas as pd
from scipy.stats import spearmanr
from scipy.linalg import toeplitz
# import differint.differint as df
from methods.plotting_functions import *
from methods.Gradients import *
from methods.Fractional_Gradients import *

from captum.attr import (
    InputXGradient,
    Saliency,
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    GuidedGradCam,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    GuidedBackprop,
)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# # For mutliple devices (GPUs: 4, 5, 6, 7)
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

# print('Using CUDA device: ', str(os.environ["CUDA_VISIBLE_DEVICES"]))

def attribute_image_features(algorithm, input, **kwargs):
    model_ft.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=int(predicted_out[num]),
                                              **kwargs
                                             )
    return tensor_attributions


    
class Args():
    def __init__(self):
        # self.loadModel1 = './models/Non-Robust/model_val_0.6547000000000001_epoch_30_nonrobust.pt'
        # self.loadModel2 = './models/Robust-GN-std-0.05/model_val_0.6356_epoch_34_robust_gn.pt'
        # self.loadModel3 = './models/Robust-PGD-alpha-1-steps-1/model_val_0.6421_epoch_28_robust_atk.pt'
        
        self.loadModel1 = './models/Non-Robust-no-crop/model_val_0.6503_epoch_22_nonrobust.pt'
        self.loadModel2 = './models/Robust-PGD-alpha-2-steps-1-no-crop/model_val_0.6331_epoch_17_robust_atk.pt'
        self.loadModel3 = './models/Robust-GN-std-0.05-mean-0-no-crop/model_val_0.6394000000000001_epoch_24_robust_gn.pt' 
        self.loadModel4 = './models/Robust-GN-std-0.05-mean-0.5-no-crop/model_val_0.6377_epoch_31_robust_gn.pt'
        
        self.model_path_list = [self.loadModel1, self.loadModel2, self.loadModel3, self.loadModel4]
        # self.model_path_list = [self.loadModel1, self.loadModel2, self.loadModel3]
        # self.model_path_list = [self.loadModel2, self.loadModel3]
        # self.model_path_list = [self.loadModel3]
        # self.loadModel = ''
        self.cuda = True
        self.epochs     = 30
        self.batch_size = 1
        self.lr         = 0.001
        self.num_labels = 10
        self.nsamples = 1
        self.sample_index = [3]#[12]#[4, 10, 12, 13, 14, 15, 16]#[1, 4, 9, 10, 11, 12, 13, 14, 15, 16]
        ## 11: bird, 10: lizard (for test)
        self.gray_maps = False
        self.squared_maps = False
        self.set_size = 24
        self.set_start = 100
        self.h = 0.00001
        self.h_range = 10
        self.alpha = 0.00000001
        self.do_GL = True
        self.do_RL = True
        
if __name__ == '__main__':
    args = Args()
    lr = args.lr
    cuda = args.cuda
    epochs = args.epochs
    # model_path = args.loadModel1
    model_path_list = args.model_path_list
    batch_size = args.batch_size
    num_labels = args.num_labels
    nsamples = args.nsamples
    gray_maps = args.gray_maps
    squared_maps = args.squared_maps
    # baseline = args.baseline
    set_size = args.set_size
    set_start = args.set_start
    sample_index = args.sample_index
    h = args.h
    h_range = args.h_range
    alpha = args.alpha
    do_GL = args.do_GL
    do_RL = args.do_RL
    
    h_list = [h * (inx + 1) for inx in range(h_range)]
    
    for model_path in model_path_list:
        frac_gradRL, frac_gradGL = list(), list()
        attr_frac_methods = []
        for h in h_list:
            start_alpha = 0
            for start_alpha in range(1):
                # N = 9
                N = 3
                # alpha_values = [0.000005, 0.00001, 0.00005, 0.0001, 0.001, 0.002, 0.0025, 0.0035, 0.004, 0.0045, 0.005, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.99999, 1.00000001]
                # alpha_values = [0.99999, 1.99999, 2.99999]
                # alpha_values = [1.000001, 2.000001, 3.000001]
                # alpha_values = [0.25, 0.5, 0.75]
                # alpha_values = [0.0001, 0.001, 0.01, 0.1]
                # alpha_values = [1.1, 1.5, 1.75, 2.1, 2.4, 2.45, 2.5, 2.75, 2.9]
                # alpha_values = [0.33, 0.66, 1.01, 1.33, 1.66, 2.01, 2.33, 2.66, 2.99]
                # alpha_values = [0.2, 0.4, 0.6, 0.8, 1.01, 1.2, 1.4, 1.6, 1.8, 2.01, 2.2, 2.4, 2.6, 2.8, 2.99]
                # alpha_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01, 1.1, 1.2, 1.3, 1.4, 1.5, 
                #                 1.6, 1.7, 1.8, 1.9, 2.01]#, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 2.99]
                alpha_values = [1.5]#,1.01]
                # CONSIDER ENSEMBLING FRACTIONAL DERIVATIVES FOR EXPERIMENTS TO SEE IF IT # 
                # CREATES BETTER AND MORE PLAUSIBLE EXPLANATIONS #
                
                num_alpha_vals = len(alpha_values)
                alpha_short = [alpha_values[0], alpha_values[int(num_alpha_vals/2)], alpha_values[num_alpha_vals-1]]
                # alpha_values = [2.3, 2.4, 2.5, 2.55, 2.6, 2.65, 2.7]
                # alpha_values = [2.1, 2.5, 2.75, 2.9]
                # alpha_values = [1.01]
                # alpha_values = [1.1, 1.25, 1.5, 1.75, 2.1, 2.25, 2.5, 2.55, 2.6, 2.65, 2.7, 2.75]
                # alpha_values = [2.5, 2.501, 2.51, 2.52, 2.53, 2.54]
                # alpha_values = [2.75, 2.9, 2.99, 2.999]
                # alpha_values = [1.1, 1.25, 1.5, 1.75, 2.1, 2.25, 2.5, 2.75]
                # alpha_values = [1.01 , 2.01]
                
                alpha_values = [alpha_values[inx] + start_alpha for inx in range(len(alpha_values))]
                
                if cuda:
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                else:
                    device = torch.device('cpu')
                
                #Load Resnet18
                model_ft = models.resnet18(pretrained=True)
                
                #Finetune Final few layers to adjust for tiny imagenet input
                model_ft.avgpool = nn.AdaptiveAvgPool2d(1)
                model_ft.fc.out_features = 200
        
                model_ft = model_ft.to(device)
                
                #Loss Function
                criterion = torch.nn.CrossEntropyLoss()
                optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
                
                classes = np.loadtxt("data/tiny-imagenet-200/words200.txt", delimiter="\n", dtype='str')
                classes = [i[10:] for i in classes]
                # data_dir = '/data/tiny-imagenet-200/'
                data_dir = 'C:/Users/ianni/OneDrive/Desktop/HW and Research/Research/Fractional_Attribution/TinyImageNet-ResNet/data/TinyImageNet/'
                
                num_workers = {'train' : 0,'val'   : 0,'test'  : 0}
                data_transforms = {
                    'train': transforms.Compose([
                        transforms.RandomRotation(20),
                        transforms.RandomHorizontalFlip(0.5),
            #            transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    ]),
                    'val': transforms.Compose([
            #            transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    ]),
                    'test': transforms.Compose([
            #            transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
                    ])
                }
                image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) 
                                  for x in ['train', 'val','test']}
                dataloaders = {x: data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=num_workers[x])
                                  for x in ['train', 'val', 'test']}
                dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
                
                # transform_test = transforms.Compose([
                #     transforms.ToTensor()
                # ])
                
                # mnist_testset = datasets.MNIST(root='./data', 
                #                            train=False, 
                #                            download=True, 
                #                            transform=transform_test
                #                            )
                
                image_datasets['val'] = torch.utils.data.Subset(image_datasets['val'], range(set_start, set_size + set_start))
                                                                        
                dataloaders['val'] = torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=True)
                
                print("Testing dataset size: ", len(dataloaders['val']))
                
                model_ft.load_state_dict(torch.load(model_path, map_location=device))
                model_ft.eval()
                    
                gbp = GuidedBackprop(model_ft)
                sa = Saliency(model_ft)
                dl = DeepLift(model_ft)
                gxi = InputXGradient(model_ft)
                i_g = IntegratedGradients(model_ft)
                
                nt = NoiseTunnel(sa)
            
                #Test
                result_images, predicted_out = list(), list()
                labels = list()
                for itr, (image, label) in enumerate(dataloaders['val']):
                    
                    if itr == int(sample_index[0]):
                        if h == h_list[0]:
                            imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(image.cpu())))
                            plt.show()
                    
                    image, label = image.to(device), label.to(device)
                    
                    image.requires_grad_(True)
                    pred = model_ft(image)
                    
                    loss = criterion(pred, label)
                    loss.backward()
                    
                    for i, p in enumerate(pred):
                        predicted_out.append(torch.max(p.data, 0)[1].detach().cpu().numpy())
                        result_images.append(image)
                        labels.append(label)
                
                # print(predicted_out)
                
                attribution_names = [
                                    "Zero Grad",
                                    "Grad Captum",
                                    # "First Grad Estimate",
                                    # "Second Grad Estimate",
                                    # "Third Grad Estimate",
                                    ]
                if N > 0:
                    attribution_names.append("First Grad Estimate")
                if N > 1:
                    attribution_names.append("Second Grad Estimate")
                if N > 2:
                    attribution_names.append("Third Grad Estimate")
                
                grad_list_h = list()
                images, preds = list(), list()
                # attribution_mapsRL, attribution_mapsGL = list(), list()
                # for x in range(len(attribution_names) + len(alpha_values)):
                #     attribution_mapsRL.append(list())
                #     attribution_mapsGL.append(list())
                
                for itr in range(nsamples):
                    print("Calculating saliency maps for sample ", int(itr + 1))
                    num = sample_index[itr]
                    img = result_images[num].clone().detach().to(device)
                    
                    fig_name_start = str('./figures/individuals/' + str(sample_index) + model_path.replace('/', "_").replace('.', "_"))
                    #grad_x_image = attribute_image_features(gxi, img)
                    rightHand = False
                    if h == h_list[0]:
                        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(img.clone().detach().cpu())), title=str(fig_name_start+'img'))
        
                    print("Calculating: ", "vg_captum")
                    # vanilla_grad = returnGrad(img, model_ft, criterion)
                    vg_captum = attribute_image_features(sa, img, abs = squared_maps)
                    if h == h_list[0]:
                        imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(vg_captum.clone().detach().cpu())), title=str(fig_name_start+'vg_captum'))
                    # FracGrads = Frac_Grads(img, model_ft, h=h, index=predicted_out[num], rightHand = rightHand)
                    FracGrads = Frac_Grads(img, model_ft, h=h, index=labels[num][0].cpu().numpy(), rightHand = rightHand)
                    
                    # zero_grad = returnZeroDerivGrad(img)
                    print("Calculating: ", "zero_grad") # returnDerivApprox_Efficient
                    zero_grad = FracGrads.returnDerivOnePixelApprox(n=0)
                    # zero_grad = returnDerivApprox_Efficient(img, model_ft, n=0, h=h, index=predicted_out[num], rightHand = rightHand)
                    # print("Calculating: ", "first_grad_est")
                    # first_grad_est = returnDerivApprox(img, model_ft, n=1, h=h, index=predicted_out[num], rightHand = rightHand)
                    # # first_grad_est = returnDerivApprox_Efficient(img, model_ft, n=1, h=h, index=predicted_out[num], rightHand = rightHand)
                    # print("Calculating: ", "second_grad_est")
                    # second_grad_est = returnDerivApprox(img, model_ft, n=2, h=h, index=predicted_out[num], rightHand = rightHand)
                    # second_grad_est = returnDerivApprox_Efficient(img, model_ft, n=2, h=h, index=predicted_out[num], rightHand = rightHand)
                    # print("Calculating: ", "third_grad_est")
                    # third_grad_est = returnDerivApprox(img, model_ft, n=3, h=h, index=predicted_out[num], rightHand = rightHand)
                    # third_grad_est = returnDerivApprox_Efficient(img, model_ft, n=3, h=h, index=predicted_out[num], rightHand = rightHand)
                    
                    grad_est_list = [zero_grad]#, first_grad_est, second_grad_est, third_grad_est]
                    # imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(grad_est_list[0].clone().detach().cpu())), title=str(fig_name_start+'zero_grad'))
                    if N > 0:
                        for i_n in range(1, N + 1):
                            print('Estimating Grad Order: ', str(i_n))
                            grad_est_list.append(FracGrads.returnDerivOnePixelApprox(n=i_n))
                            # imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(grad_est_list[i_n].clone().detach().cpu())), title=str(fig_name_start+str('Est_n_' + str(i_n))))
        
                    
                    
                    attrs = [grad_est_list[0], #zero_grad,
                             vg_captum, 
                             # grad_est_list[1], #first_grad_est, 
                             # grad_est_list[2], #second_grad_est, 
                             # grad_est_list[3], #third_grad_est, 
                             ]
                    for ip in range(1, N + 1):
                        attrs.append(grad_est_list[ip])
                    
                    
                    deriv_method_name = []
                    if do_RL:
                        deriv_method_name.append('RL')
                    if do_GL:
                        deriv_method_name.append('GL')
                    
                    ############### Make Single Pixel Versions of RL and GL ###############
                    for its, alpha in enumerate(alpha_values):
                        if do_RL:
                            print('Calculating', 'RL', str("Frac Grad a=" + str(alpha)))
                            frac_gradRL.append(FracOnePixelGradApproxRL(img.clone().detach(), alpha, h, grad_list = grad_est_list, i=img.size(2)-1, j=img.size(3)-1))# zero_grad, first_grad_est, second_grad_est, third_grad_est))
                            # imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(frac_gradRL[its].clone().detach().cpu())), title=str(fig_name_start+str('Frac_RL_a_' + str(alpha))))
                        if do_GL:
                            print('Calculating', 'GL', str("Frac Grad a=" + str(alpha)))
                            # frac_gradGL.append(FracGradApproxGL(img.clone().detach(), alpha, h, model_ft, index=predicted_out[num], rightHand = rightHand))#=None))#
                            frac_gradGL.append(FracGrads.FracOnePixelGradApproxGL(alpha = alpha, i_x = img.size(2)-1, i_y = img.size(3)-1))#
                            # imshow(torchvision.utils.make_grid(VisualizeImageGrayscale(frac_gradGL[its].clone().detach().cpu())), title=str(fig_name_start+str('Frac_GL_a_' + str(alpha))))
                        
                        attribution_names.append(str("Frac Grad a=" + str(alpha)))
                        
                    #gbp_attributions = gbp.attribute(img, target=int(predicted_out[num]))
                    #gbp_attributions = gbp_attributions.cpu()
                    
                    #integrated_grad = attribute_image_features(i_g, img, n_steps=50)
                    #integrated_grad, _ = integrated_gradients(img.cpu(), int(predicted_out[num]), baseline, steps=50, magnitude=squared_maps)
                    
                    #smoothgrad = nt.attribute(img, nt_type='smoothgrad', n_samples=150, target=int(predicted_out[num]))
                    #smoothgrad = sg.GetSmoothedMask(img.cpu(), nsamples=15, magnitude=squared_maps)
                    
                    #if squared_maps:
                        #vanilla_grad = vanilla_grad * vanilla_grad
                        #grad_x_image = grad_x_image * grad_x_image
                        #gbp_attributions = gbp_attributions * gbp_attributions
                        #dl_attributions = dl_attributions * dl_attributions
                    
                    ## setup to compare methods ##
                    # for itx in range(len(attrs)):
                    #     if do_RL:
                    #         attribution_mapsRL[itx].append(attrs[itx].clone().detach().cpu())
                    #     if do_GL:
                    #         attribution_mapsGL[itx].append(attrs[itx].clone().detach().cpu())
                    
                    # for p in range(len(alpha_values)):
                    #     if do_RL:
                    #         attribution_mapsRL[p + itx + 1].append(frac_gradRL[p])
                    #     if do_GL:
                    #         attribution_mapsGL[p + itx + 1].append(frac_gradGL[p])
                    
                    
                    
                    images.append(img)
                    preds.append(int(predicted_out[num]))
                
            
            # if do_RL:
            #     attr_frac_methods.append(attribution_mapsRL)
            # if do_GL:
            #     attr_frac_methods.append(attribution_mapsGL)
        
        plt.figure()

        # h_list_sort = sorted(h_list * len(alpha_values))
        
        if len(alpha_values) > 1:
            for refi, frac_grad in enumerate([frac_gradRL, frac_gradGL]):
                odd_i, even_i = [], []
                for i in range(0, len(frac_grad)):
                    if i % 2:
                        even_i.append(frac_grad[i])
                    else :
                        odd_i.append(frac_grad[i])
                res = odd_i + even_i
                if refi == 0:
                    frac_gradRL_sort = res
                if refi == 1:
                    frac_gradGL_sort = res
            plt.plot(torch.tensor(h_list),torch.tensor(frac_gradRL_sort[:int(len(frac_gradRL_sort)/2)]),'x', 
                     torch.tensor(h_list),torch.tensor(frac_gradRL_sort[int(len(frac_gradRL_sort)/2):]),'x', 
                     torch.tensor(h_list),torch.tensor(frac_gradGL_sort[:int(len(frac_gradGL_sort)/2)]),'x',
                     torch.tensor(h_list),torch.tensor(frac_gradGL_sort[int(len(frac_gradGL_sort)/2):]),'x')
        else:
            plt.plot(torch.tensor(h_list),torch.tensor(frac_gradRL),'x',
                     torch.tensor(h_list),torch.tensor(frac_gradGL),'x')
        # plt.plot(x1, -(x1*wb[0,1] + wb[0,0])/wb[0,2] , linestyle = 'solid') # plot decision boundary
        # plt.axis('equal')
        plt.xlabel('h')
        plt.ylabel('Fractional Gradient')
        plt.title(str('RL and GL One Pixel Plot (alpha = '+str(alpha)+")"))
        
        #str('RL alpha ='+str(alpha_values[i]))
        # for i in range(0, len(frac_grad)):
        #     if i % 2:
        leg = [str(deriv_method_name[i]+' alpha = '+str(alpha_values[j])) for i in range(2) for j in range(2)]
        plt.legend(leg)
        plt.show()
        
        # plt.figure()
        # plt.plot(torch.tensor(alpha_values),torch.tensor(frac_gradGL),'x')#, torch.tensor(frac_gradGL), alpha_values,'x') # plot random data points
        # # plt.plot(x1, -(x1*wb[0,1] + wb[0,0])/wb[0,2] , linestyle = 'solid') # plot decision boundary
        # # plt.axis('equal')
        # plt.xlabel('alpha')
        # plt.ylabel('output')
        # plt.title('GL One Pixel Plot')
        

