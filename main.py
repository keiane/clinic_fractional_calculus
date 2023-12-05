import torch
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from model import Model
import numpy as np
import gradio as gr

from styling import theme

def main(no_epochs):
    print('test')

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))

    mnist_valset, mnist_testset = torch.utils.data.random_split(mnist_testset, [int(0.9 * len(mnist_testset)), int(0.1 * len(mnist_testset))])

    train_dataloader = torch.utils.data.DataLoader(mnist_trainset, batch_size=64, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(mnist_valset, batch_size=32, shuffle=False)
    test_dataloader = torch.utils.data.DataLoader(mnist_testset, batch_size=32, shuffle=False)

    print("Training dataset size: ", len(mnist_trainset))
    print("Validation dataset size: ", len(mnist_valset))
    print("Testing dataset size: ", len(mnist_testset))

    print("Plotting Figure 1 =>")
    # visualize data
    fig1=plt.figure(figsize=(20, 10))
    for i in range(1, 6):
        img = transforms.ToPILImage(mode='L')(mnist_trainset[i][0])
        fig1.add_subplot(1, 6, i)
        plt.title(mnist_trainset[i][1])
        plt.imshow(img)
    plt.show()

    model = Model()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if (torch.cuda.is_available()):
        model.cuda()

    TRAIN_MODEL = False

    if TRAIN_MODEL:
        no_epochs = no_epochs
        train_loss = list()
        val_loss = list()
        best_val_loss = 10
        for epoch in range(no_epochs):
            total_train_loss = 0
            total_val_loss = 0
        
            model.train()
            # training
            for itr, (image, label) in enumerate(train_dataloader):
        
                if (torch.cuda.is_available()):
                    image = image.cuda()
                    label = label.cuda()
                
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
        
                if (torch.cuda.is_available()):
                    image = image.cuda()
                    label = label.cuda()
        
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
        print("Plotting Figure 2 =>")
        fig=plt.figure(figsize=(20, 10))
        plt.plot(np.arange(1, no_epochs+1), train_loss, label="Train loss")
        plt.plot(np.arange(1, no_epochs+1), val_loss, label="Validation loss")
        plt.xlabel('Loss')
        plt.ylabel('Epochs')
        plt.title("Loss Plots")
        plt.legend(loc='upper right')
        plt.show()

    # test model
    model.load_state_dict(torch.load("model_weights/model_cifar.dth", map_location=torch.device(('cuda' if torch.cuda.is_available() else 'cpu'))))
    model.eval()

    results = list()
    total = 0
    for itr, (image, label) in enumerate(test_dataloader):

        if (torch.cuda.is_available()):
            image = image.cuda()
            label = label.cuda()

        pred = model(image)

        for i, p in enumerate(pred):
            if label[i] == torch.max(p.data, 0)[1]:
                total = total + 1
                results.append((image, torch.max(p.data, 0)[1]))

    test_accuracy = total / (itr + 1)
    print('Test accuracy {:.8f}'.format(test_accuracy))

    print("Plotting Figure 3 =>")
    # visualize results
    fig2=plt.figure(figsize=(20, 10))
    for i in range(1, 11):
        img = transforms.ToPILImage(mode='L')(results[0][0][i].squeeze(0).detach().cpu())
        fig2.add_subplot(2, 5, i)
        plt.title(results[i][1].item())
        plt.imshow(img)
    plt.show()

    return test_accuracy, fig1, fig2

with gr.Blocks(theme=theme) as demo:
    with gr.Row():
        no_epochs = gr.Slider(minimum=1, maximum=30, step=1, default=1, label="Number of epochs")
        run_btn = gr.Button(text="Run")
    with gr.Row():        
        initial_plot = gr.Plot(title="Loss Plots", labels=["Epochs", "Loss"])
        results_plot = gr.Plot(title="Results", labels=["Image", "Label"])
        # image = gr.inputs.Image(shape=(28, 28))
        # label = gr.outputs.Label(num_top_classes=3)
        accuracy = gr.Textbox(label="Accuracy")
    run_btn.click(fn=main, inputs=[no_epochs], outputs=[accuracy, initial_plot, results_plot])

demo.launch()

main()
