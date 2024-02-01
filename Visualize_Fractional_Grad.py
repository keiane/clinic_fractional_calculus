
### CODE FOR MNIST ATTRIBUTION ###


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
import random

import gradio as gr
from styling import theme

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

    
class Args():
    def __init__(self):
        # self.loadModel = './model_weights/model_baseline.dth' # './model_ROAR_baseline.dth'
        self.cifar_model = './model_weights/model_cifar_baseline.dth'
        self.mnist_model = './model_weights/model_mnist_baseline.dth'
        # TO RUN RL AND GL, UNCOMMENT THE LINE BELOW, OTHERWISE ONLY INCLUDE 'RL' OR 'GL' IN LIST
        # self.deriv_method_names = ['RL', 'GL']
        self.deriv_method_names = ['RL',]
        self.deriv_approx_integ = False # If true, calulates the numerical approximation of the integer order derivatives
#        self.loadModel = ''
        self.cuda = False
        self.epochs     = 30
        self.batch_size = 1
        self.lr         = 0.001
        self.num_labels = 10
        self.nsamples = 3 # Overridden by sample_index if randomize_samples is false
        # Sample indexes: [13, 16]#[4, 10, 12, 13, 14, 15, 16, 17, 18, 19]#[1, 4, 9, 10, 11, 12, 13, 14, 15, 16]
        self.sample_index = [17, 18, 19,]
        self.randomize_samples = True # Overrides sample_index if true and uses nsamples
        self.gray_maps = False
        self.squared_maps = False
        self.set_size = 50
        self.h = 0.00001
        self.alpha = 0.00000001
        self.N = 7
        self.dataset_names = ['MNIST', 'CIFAR10']
        # self.dataset = 0 # 'MNIST'
        # self.dataset_val = 0 # 'CIFAR10'

def test(database, alpha_slider):

    def attribute_image_features(algorithm, input, **kwargs):
        model_ft.zero_grad()
        tensor_attributions = algorithm.attribute(input,
                                                target=int(predicted_out[num]),
                                                **kwargs
                                                )
        return tensor_attributions

    args = Args()
    ##############################################################################
    if database == "MNIST":
        model_path = args.mnist_model
    if database == "CIFAR10":
        model_path = args.cifar_model
    # model_path = args.loadModel
    deriv_method_names = args.deriv_method_names
    deriv_approx_integ = args.deriv_approx_integ
    cuda = args.cuda
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    num_labels = args.num_labels
    sample_index = args.sample_index
    gray_maps = args.gray_maps
    squared_maps = args.squared_maps
    set_size = args.set_size
    h = args.h
    alpha = args.alpha
    randomize_samples = args.randomize_samples
    if not randomize_samples:
        nsamples = len(sample_index)
    else:
        nsamples = args.nsamples
    sample_index = random.sample(range(set_size), nsamples)
    N = args.N
    dataset_names = args.dataset_names
    dataset_val = database
    ##############################################################################

    start_alpha = 0
    for start_alpha in range(1):
        
        print("N: ", N)
        ################ Auto-Generated Alpha Values for Range ################
        # # RL CHANGE: Between 1.9998 and 1.99999 ; 2.00001 and 2.001 ; 2.9999998 and 2.99999999
        start_ends = [
                        # [0.1, 0.6],
                        [0.0001, 0.9999],
                        [1.0001, 1.9999],
                        [2.0001, 2.9999],
                        # [1.9998, 1.99999], 
                        # [2.00001, 2.001], 
                        # [2.9999995, 2.99999999],
                        ]
        num_steps = 3
        alphas = [[start + (((end-start)/float(num_steps)) * i) for i in range(num_steps + 1)] for start, end in start_ends]
        alpha_values = []
        # alpha_values = [0.999, 1.5, 2.001, 2.5, 3.001] # Slider
        ## alpha_values = sorted([sample[0] for sample in samples]) # Slider
        alpha_values.append(alpha_slider)
        # for i in range(len(alphas)):
        #     alpha_values += alphas[i] 
        ########################################################################
        
        # alpha_values = [alpha_values[inx] + start_alpha for inx in range(len(alpha_values))]
        
        if cuda:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device('cpu')
        
        if dataset_val == "MNIST": # if dataset_val is MNIST
            model_ft = Model()
        if dataset_val == "CIFAR10": # if dataset_val is CIFAR10
            model_ft = torchvision.models.resnet18(pretrained=True)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, 10)

        
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model_ft.parameters(), lr=0.000005)

        model_ft = model_ft.to(device)
        
        
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
        
        if dataset_val == "MNIST": # if dataset_val is MNIST
            testset = datasets.MNIST(root='./data', 
                                    train=False, 
                                    download=True, 
                                    transform=transform_test
                                    )
        if dataset_val == "CIFAR10": # if dataset_val is CIFAR10
            testset = datasets.CIFAR10(root='./data', 
                                    train=False, 
                                    download=True, 
                                    transform=transform_test
                                    )
        
        testset = torch.utils.data.Subset(testset, range(set_size))
                                                                
        test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
        
        print("Testing dataset size: ", len(testset))
        
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
        for itr, (image, label) in enumerate(test_dataloader):
            
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
        
        
        images, preds = list(), list()
        frac_gradRL, frac_gradGL = list(), list()
        attribution_mapsRL, attribution_mapsGL = list(), list()
        
        for itr in range(nsamples):
            print("Calculating saliency maps for sample ", int(itr + 1))
            num = sample_index[itr]
            img = result_images[num].clone().detach().to(device)
            
            #grad_x_image = attribute_image_features(gxi, img)
            rightHand = False
            
            
            vanilla_grad = returnGrad(img, model_ft, criterion, device)
            vg_captum = attribute_image_features(sa, img, abs = squared_maps)
            
            FracGrads = Frac_Grads(img, model_ft, h=h, index=labels[num][0].cpu().numpy(), rightHand = rightHand)
            grads = grad_calc(model_ft)
            
            grad_list = [
                        # zero_grad_autograd, 
                        # first_grad_autograd, 
                        # second_grad_autograd, 
                        # third_grad_autograd, 
                        ]
            
            for i_n in range(len(grad_list), N + 1):
                print('Estimating Grad Order: ', str(i_n), end="\r")
                # grad_list.append(FracGrads.returnDerivApprox(n=i_n))
                # grad_list.append(returnHigherOrderDeriv(img, model_ft, n=i_n, index=labels[num][0], device = device))
                grad_list.append(grads.returnHigherOrderDeriv(img, n=i_n, index=labels[num][0], device = device))
            
            if deriv_approx_integ:
                zero_grad = FracGrads.returnDerivApprox(n=0)
                first_grad_est = FracGrads.returnDerivApprox(n=1)
                second_grad_est = FracGrads.returnDerivApprox(n=2)
                third_grad_est = FracGrads.returnDerivApprox(n=3)
                grad_est_list = [zero_grad, 
                                first_grad_est, 
                                second_grad_est, 
                                third_grad_est, 
                                ]

            attrs = [
                    # zero_grad_autograd, #zero_grad,
                    # vg_captum, 
                    ]
            attribution_names = [
                            # "Zero Grad",
                            # "Grad Captum",
                            ]
            
            for i in range(1, min((3 + 1), len(grad_list))):
                attrs.append(grad_list[i])
                attribution_names.append(f"Deriv {i}")
            if deriv_approx_integ:
                for i in range(1, len(grad_est_list)):
                    attrs.append(grad_est_list[i])
                    attribution_names.append(f"Deriv {i} Approx")
            
            # frac_gradGL_class = list()
            frac_gradRL, frac_gradGL = list(), list()
            attribution_mapsRL_, attribution_mapsGL_ = list(), list()
            
            for _, alpha in enumerate(alpha_values):
                print('Calculating ', str("a=" + str(alpha)))
                if 'RL' in deriv_method_names:
                    frac_gradRL.append(FracGradApproxRL(img.clone().detach(), alpha, h, grad_list = grad_list))# zero_grad, first_grad_est, second_grad_est, third_grad_est))
                if 'GL' in deriv_method_names:
                    frac_gradGL.append(FracGradApproxGL(img.clone().detach(), alpha, h, model_ft, index=predicted_out[num], rightHand = rightHand, N = N))#=None))#
                # frac_gradGL.append(FracGradApproxGL(alpha = alpha))#, N = N))
                # frac_gradGL.append(FracGrads.FracGradApproxGL(alpha = alpha))#, N = N))
                
                attribution_names.append(str("a=" + str(round_repeating_decimals(alpha, precision=8))))
            
            if squared_maps:
                for attr in attrs:
                    attr = attr * attr
            
            # attribution_mapsGL_class = list()
            for x in range(len(attribution_names)):
                attribution_mapsRL_.append(list())
                attribution_mapsGL_.append(list())
                # attribution_mapsGL_class.append(list())
            
            ## setup to compare methods ##
            # for att_maps_ in att_maps:
            for itx in range(len(attrs)):
                attribution_mapsRL_[itx].append(attrs[itx].clone().detach()) if 'RL' in deriv_method_names else None
                attribution_mapsGL_[itx].append(attrs[itx].clone().detach()) if 'GL' in deriv_method_names else None
                # att_maps_[itx].append(attrs[itx].clone().detach())
            
            # for att_maps_ in att_maps:
            for p in range(len(alpha_values)):
                attribution_mapsRL_[p + itx + 1].append(frac_gradRL[p].clone().detach()) if 'RL' in deriv_method_names else None
                attribution_mapsGL_[p + itx + 1].append(frac_gradGL[p].clone().detach()) if 'GL' in deriv_method_names else None
                
            
            attribution_mapsRL.append(attribution_mapsRL_)
            attribution_mapsGL.append(attribution_mapsGL_)
            images.append(img)
            preds.append(int(predicted_out[num]))
        
        plt.ioff()
        att_maps = []
        if 'RL' in deriv_method_names:
            att_maps.append(attribution_mapsRL)
        if 'GL' in deriv_method_names:
            att_maps.append(attribution_mapsGL)
        for i_fig, attribution_maps in enumerate(att_maps):
            # print(i_fig)
            m = 1
            scale = 1.5
            length, width = int(nsamples), int((len(attrs)+1) + len(alpha_values))#(len(attribution_maps) + 1)
            fig=plt.figure(figsize=(width * scale, length * scale))
            plt.title(deriv_method_names[i_fig])
        
            for i in range(nsamples): # len(attribution_maps) = nsamples
                
                fig.add_subplot(length, width, (i * width) + 1).set_title(str("pred:" + str(preds[i])))
                imshow((VisualizeImageGrayscale(images[i][0].clone().detach().cpu())))
                plt.axis('off')
        
                for j in range(len(attribution_maps[0])):
                    
                    attribution_map = attribution_maps[i][j][0].clone().detach().cpu()
                    attribution_map *= 3.0
                    if gray_maps:
                        attribution_map = attribution_map[0][0] + attribution_map[0][1] + attribution_map[0][2]
                    
                    # df = pd.DataFrame({'A': np.array(torch.flatten(attribution_maps[0][i].clone().detach().cpu())).tolist(),
                    #                        'B': np.array(torch.flatten(attribution_map)).tolist()})
                    # rho1, p1 = spearmanr(df['A'], df['B'])
                    
                    fig.add_subplot(length, width, (i * width) + j + 2).set_title(str(attribution_names[j]), loc='center')# + " RC: "+ str(round(rho1, 4))))
                    imshow((VisualizeImageGrayscale(attribution_map))[0], cmap='RdBu')
                    plt.axis('off')
            
            fig_name = str("figures/saliency_MNIST_" + str(deriv_method_names[i_fig]) + 'start-alpha-' + str(start_alpha) + '_samples-' + str(sample_index))
            if squared_maps:
                fig_name = fig_name + "_sq"
            plt.savefig(fig_name + ".png")
            # plt.show()
            # plt.ioff()
        return fig

with gr.Blocks() as functionApp:

    with gr.Row():
        gr.Markdown("# Fractional Calculus Web App")
    with gr.Row():
            gr.Markdown("## Inputs")
    with gr.Column():
        with gr.Row():
            with gr.Column():
                alpha_slider = gr.Slider(label="Alpha", minimum=0.001, maximum=4, step=0.001)
                database = gr.Radio(choices=["MNIST", "CIFAR10"], label="Choose a Database", value="MNIST")

    with gr.Row():
        gr.Markdown("## Results")
    with gr.Row():
        with gr.Column():
            plot1 = gr.Plot(label="RL Attribution Map")
        
    alpha_slider.change(fn=test, inputs=[database, alpha_slider], outputs=[plot1])


markdown_file_path = 'documentation.md'
with open(markdown_file_path, 'r') as file:
    markdown_content = file.read()
with gr.Blocks() as documentationApp:
    with gr.Row():
        gr.Markdown(markdown_content)

### LAUNCH APP
demo = gr.TabbedInterface([functionApp, documentationApp], ["Run Model", "Documentation"], theme=theme)
demo.queue().launch()
