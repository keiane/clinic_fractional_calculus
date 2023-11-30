
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model import Model
import numpy as np
import torchvision.models as models
import torch.nn as nn
import torchvision
from methods.Gradients import *

# Function passed into the forward hook
hook_activations = None # Variable to store the activations
def getActivation():
    # the hook signature
    def hook(model, input, output):
        global hook_activations
        hook_activations = output#.detach() # REMOVE DETACH IF U WANT TO TURN THIS INTO A LOSS AND BACKPROP
        # return activation
    return hook

def nested_children(m: torch.nn.Module):
    children = dict(m.named_children())
    output = {}
    names = []
    if children == {}:
        # if the module is on the last child (m)
        return m
    else:
        # look for children in children list
        for name, child in children.items():
            names.append(name)
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return names, output

def imshow(img, transpose = True):
    img = torchvision.utils.make_grid(img)
    # img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    npimg_tp = np.transpose(npimg, (1, 2, 0))
    plt.imshow(npimg_tp)
    # plt.show()
    
def normalize_image(image_3d):
    vmin = torch.min(image_3d)
    image_2d = image_3d - vmin
    vmax = torch.max(image_2d)
    return (image_2d / vmax)


cuda = False

if cuda:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')

mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

mnist_valset, mnist_testset = torch.utils.data.random_split(mnist_testset, [int(0.9 * len(mnist_testset)), int(0.1 * len(mnist_testset))])

test_batch = 1
train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=test_batch, shuffle=False)

classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

print("Training dataset size: ", len(mnist_trainset))
print("Validation dataset size: ", len(mnist_valset))
print("Testing dataset size: ", len(mnist_testset))

# visualize data
fig=plt.figure(figsize=(20, 10))
for i in range(1, 6):
    img = transforms.ToPILImage(mode='L')(mnist_trainset[i][0])
    fig.add_subplot(1, 6, i)
    plt.title(mnist_trainset[i][1])
    plt.imshow(img)
plt.show()

model = Model()

# model = models.resnet18(pretrained=True)
# model.avgpool = torch.nn.AdaptiveAvgPool2d(1)
# model.fc.out_features = len(classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# if (torch.cuda.is_available()):
#     model.cuda()

model.to(device)

TRAIN_MODEL = False

if TRAIN_MODEL:
    no_epochs = 30
    train_loss = list()
    val_loss = list()
    best_val_loss = 10
    for epoch in range(no_epochs):
        total_train_loss = 0
        total_val_loss = 0
    
        model.train()
        # training
        for itr, (image, label) in enumerate(train_dataloader):
    
            # if (torch.cuda.is_available()):
            #     image = image.cuda()
            #     label = label.cuda()
            image = image.to(device)
            label = label.to(device)
            
            
            optimizer.zero_grad()
    
            pred = model(image)
            pred = torch.nn.functional.softmax(pred, dim=1)
    
            loss = criterion(pred, label)
            total_train_loss += loss.item()
    
            loss.backward()
            optimizer.step()
    
        total_train_loss = total_train_loss / (itr + 1)
        train_loss.append(total_train_loss)
    
        # validation
        model.eval()
        total = 0
        for itr, (image, label) in enumerate(val_dataloader):
    
            # if (torch.cuda.is_available()):
            #     image = image.cuda()
            #     label = label.cuda()
            image = image.to(device)
            label = label.to(device)
    
            pred = model(image)
    
            loss = criterion(pred, label)
            total_val_loss += loss.item()
    
            pred = torch.nn.functional.softmax(pred, dim=1)
            for i, p in enumerate(pred):
                if label[i] == torch.max(p.data, 0)[1]:
                    total = total + 1
    
        accuracy = total / len(mnist_valset)
    
        total_val_loss = total_val_loss / (itr + 1)
        val_loss.append(total_val_loss)
    
        print('\nEpoch: {}/{}, Train Loss: {:.8f}, Val Loss: {:.8f}, Val Accuracy: {:.8f}'.format(epoch + 1, no_epochs, total_train_loss, total_val_loss, accuracy))
    
        if total_val_loss < best_val_loss:
            best_val_loss = total_val_loss
            print("Saving the model state dictionary for Epoch: {} with Validation loss: {:.8f}".format(epoch + 1, total_val_loss))
            torch.save(model.state_dict(), "model_weights/model_cifar.dth")
    
    fig=plt.figure(figsize=(20, 10))
    plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
    plt.plot(np.arange(1, no_epochs+1), val_loss, label="Validation loss")
    plt.xlabel('Loss')
    plt.ylabel('Epochs')
    plt.title("Loss Plots")
    plt.legend(loc='upper right')
    plt.show()

# test model
model.load_state_dict(torch.load("model_weights/model_cifar.dth"))
model.eval()


layer_names, layers = nested_children(model)
lyrs = [layers[l] for l in layer_names]

results = list()
total = 0
for itr, (image, label) in enumerate(test_dataloader):

    # if (torch.cuda.is_available()):
    #     image = image.cuda()
    #     label = label.cuda()
    image = image.to(device)
    label = label.to(device)
    
    image = image.requires_grad_(requires_grad=True)
    pred = model(image)

    for i, p in enumerate(pred):
        if label[i] == torch.max(p.data, 0)[1]:
            total = total + 1
            results.append((image, torch.max(p.data, 0)[1], torch.max(p)))

test_accuracy = total / (itr + 1)
print('Test accuracy {:.8f}'.format(test_accuracy))

img_num = 0

# Saving Activations
model_activations = []
for i in range(len(lyrs)):
    hk = lyrs[i].register_forward_hook(getActivation())
    pred = model(results[img_num][0])
    model_activations.append(hook_activations.cpu().detach())
    hk.remove()
# model_activations = torch.tensor(model_activations)

params = []
for parameter in model.parameters():
    params.append(parameter)#.data)
    

results[img_num][2].backward(retain_graph=True, create_graph=True) # 2 is the full max pred

pred_max = results[img_num][2]
lyr_grads = [torch.autograd.grad(pred_max, results[img_num][0], 
                                 retain_graph=True, create_graph=True)[0].requires_grad_(requires_grad=True)]
for ik in range(1, len(params)):
    lyr_grads.append(torch.autograd.grad(pred_max, params[ik], 
                                         retain_graph=True)[0].requires_grad_(requires_grad=True))

f = results[img_num][2].requires_grad_(requires_grad=True)
wrt = lyr_grads[0].requires_grad_(requires_grad=True)
n = 2

# deriv2 = nth_derivative(f, wrt, 2)
# print(deriv2)

deriv1_pred_wrt_l1 = lyr_grads[0] # first derivative wrt first layer (d pred_max / d layer1)
deriv2_pred_wrt_l1 = torch.autograd.grad(deriv1_pred_wrt_l1.sum().requires_grad_(requires_grad=True), results[img_num][0].requires_grad_(requires_grad=True), 
                                  retain_graph=True, create_graph=True)#[0]
# second derivative of first layer (d^2 pred_max / d layer1^2)

# print(params[0].shape)
# deriv1_l1_wrt_img = torch.autograd.grad(params[0], results[img_num][0] ,retain_graph=True, create_graph=True)
# lyr_frac_grad = 
# frac_grad = (D^a l1 wrt img) * deriv1_pred_wrt_l1

# print(deriv2_pred_wrt_l1)
deriv1_img = results[img_num][0].grad # first derivative of first layer (d pred_max / d img)
# deriv2_img = torch.autograd.grad(deriv1_img.sum(), results[img_num][0], 
#                                   retain_graph=True, create_graph=True)[0].requires_grad_(requires_grad=True)
# second derivative of first layer (d^2 pred_max / d img^2)
    
deriv2_img = returnHigherOrderDeriv(img=results[img_num][0], n=2, device = device)
deriv3_img = returnHigherOrderDeriv(img=results[img_num][0], n=3, device = device)

# deriv1_img_chain = lyr_grads[0]
# for i in range(len(lyr_grads)-1):
#     deriv1_img_chain = torch.dot(deriv1_img_chain, lyr_grads[i+1])


# visualize results
fig=plt.figure(figsize=(20, 10))
for i in range(1, test_batch):
    img = transforms.ToPILImage(mode='L')(results[0][0][i].squeeze(0).detach().cpu())
    fig.add_subplot(2, 5, i)
    plt.title(results[i][1].item())
    plt.imshow(img)
plt.show()

# visualize results
fig=plt.figure(figsize=(10, 10))
fig.add_subplot(1, 4, 1)
imshow((results[img_num][0].detach().cpu())) # plot image
fig.add_subplot(1, 4, 2)
imshow(normalize_image(results[img_num][0].grad.detach().cpu())) # plot first derive of image
fig.add_subplot(1, 4, 3)
imshow(normalize_image(deriv2_img.detach().cpu())) # plot second derive of image
fig.add_subplot(1, 4, 4)
imshow(normalize_image(deriv3_img.detach().cpu())) # plot second derive of image
plt.show()
# for i in range(1):
#     # img = transforms.ToPILImage(mode='L')(results[i][0].grad.squeeze(0).detach().cpu())
#     # fig.add_subplot(2, 5, i)
#     img = results[0][0].grad
#     plt.title('Grad: ', results[i][1].item())
#     plt.imshow(img)
# plt.show()
