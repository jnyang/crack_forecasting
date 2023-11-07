import os
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn

import matplotlib.pyplot as plt

# load from scripts folder
import sys
sys.path.append('../scripts/')
from image_folder import ImageFolder, ImageDataset
from loss_functions import SSIMLoss, PhaseFieldLoss
from model import GeoTorchConvLSTM

# To-do: move these to a config file
# Set parameters
# model parameters
input_dim = 1
input_width = 128
input_height = 128
hidden_layer_sizes = [256, 256, 1]
num_layers = len(hidden_layer_sizes)

# sequence lengths
len_history = 5
len_predict = 1

# training parameters
epoch_nums = 40
learning_rate = 0.002
batch_size = 4
params = {'batch_size': batch_size, 'shuffle': False, 'drop_last':False, 'num_workers': 2}


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def compute_errors(preds, y_true):
    pred_mean = preds[:, 0:2]
    diff = y_true - pred_mean

    mse = np.mean(diff ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(diff))

    return mse, mae, rmse             


def get_validation_loss(model, val_generator, criterion, device, len_history):
    model.eval()
    mean_loss = []
    for i, sample in enumerate(val_generator):
        X_batch = sample["X_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["Y_data"].type(torch.FloatTensor).to(device)

        outputs = model(X_batch)
        # might need to set this to just outputs if we are using just mseLoss
        loss=criterion(outputs[:, len_history - 1:len_history, :, :, :], Y_batch).item()
        mean_loss.append(loss)

    mean_loss = np.mean(mean_loss)
    return mean_loss


# Create function that assigns model name to global variable
def set_model_name(name):
    global model_name, model_dir, initial_checkpoint
    model_name = name

    # Sets the model output directory
    checkpoint_dir = '../models'
    model_name = name
    model_dir = checkpoint_dir + "/" + model_name
    model_dir_plots = model_dir + "/plots"
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(model_dir_plots, exist_ok=True)

    # For loading pretrained model if available
    initial_checkpoint = model_dir + '/model.best.pth'

def load_data():
    # Set data paths
    data_root = '../data/cracks_s_nb'
    train_path = os.path.join(data_root, "train")
    test_path = os.path.join(data_root, "test")
    val_path = os.path.join(data_root, "val")

    train = ImageFolder(root=train_path, transform=transforms.ToTensor())
    test = ImageFolder(root=test_path, transform=transforms.ToTensor())
    val = ImageFolder(root=val_path, transform=transforms.ToTensor())

    train.set_sequential_representation(history_length=len_history, predict_length=len_predict)
    test.set_sequential_representation(history_length=len_history, predict_length=len_predict)
    val.set_sequential_representation(history_length=len_history, predict_length=len_predict)

    train_generator = DataLoader(ImageDataset(train), **params)
    test_generator = DataLoader(ImageDataset(test), **params)
    val_generator = DataLoader(ImageDataset(val), **params)
    
    return train_generator, test_generator, val_generator


# Write a function to use plt to save the model predictions
def save_pred_plots(outputs, Y_batch, epoch, num_samples=batch_size):
    for i in range(num_samples):
        try:
            plt.figure(figsize=(8, 4))
            plt.subplot(1, 2, 1)
            plt.imshow(outputs[i, len_history - 1, 0, :, :].cpu().data.numpy(), cmap='gray')
            plt.axis('off')
            plt.title("Predicted")
            plt.subplot(1, 2, 2)
            plt.imshow(np.squeeze(Y_batch[i, 0, :, :].cpu().data.numpy()), cmap='gray')
            plt.axis('off')
            plt.title("Ground Truth")

            # save figure to model_dir
            plt.savefig(model_dir + f"/sample_{i}_{epoch}.png")
        except:
            print(f"sample {i} failed to plot")
            continue
        
        
def createModelAndTrain(loss_fn, LOAD_INITIAL=False):
    device = get_device()

    model = GeoTorchConvLSTM(input_dim, hidden_layer_sizes, num_layers)

    if LOAD_INITIAL:
        model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))

    # calculate step size for saving prediction plots
    step_size = epoch_nums // 10

    loss_fn = loss_fn
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.to(device)
    loss_fn.to(device)
    train_generator, test_generator, val_generator = load_data()

    min_val_loss = None
    for e in range(epoch_nums):
        for i, sample in enumerate(train_generator):
            X_batch = sample["X_data"].type(torch.FloatTensor).to(device)
            Y_batch = sample["Y_data"].type(torch.FloatTensor).to(device)

            # Forward pass
            outputs = model(X_batch)
            loss = loss_fn(outputs[:, len_history - 1:len_history, :, :, :], Y_batch)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss = get_validation_loss(model, val_generator, loss_fn, device, len_history)
        
        print('Epoch [{}/{}], Train Loss: {:.4f}'.format(e + 1, epoch_nums, loss.item()), 'Mean Val Loss:', val_loss)

        if min_val_loss == None or val_loss < min_val_loss:
            min_val_loss = val_loss
            torch.save(model.state_dict(), initial_checkpoint)
            print('best model saved!')

    model.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    model.eval()
    rmse_list = []
    mse_list = []
    mae_list = []
    for i, sample in enumerate(test_generator):
        X_batch = sample["X_data"].type(torch.FloatTensor).to(device)
        Y_batch = sample["Y_data"].type(torch.FloatTensor).to(device)

        outputs = model(X_batch)
        mse, mae, rmse = compute_errors(outputs[:, len_history - 1:len_history, :, :, :].cpu().data.numpy(),
                                        Y_batch.cpu().data.numpy())

        rmse_list.append(rmse)
        mse_list.append(mse)
        mae_list.append(mae)

        if (e + 1) % step_size == 0:
            # save outputs to plot later
            save_pred_plots(outputs, Y_batch, epoch=e+1, num_samples=batch_size)

    rmse = np.mean(rmse_list)
    mse = np.mean(mse_list)
    mae = np.mean(mae_list)

    print("\n************************")
    print("Test ConvLSTM model with Crack Dataset:")
    print('Test mse: %.6f mae: %.6f rmse (norm): %.6f' % (
    mse, mae, rmse))
    

def main():
    loss_dict = {'mse': nn.MSELoss(), 'ssim': SSIMLoss(), 'phasefield': PhaseFieldLoss()}
    
    if len(sys.argv) > 1:
        loss_arg = sys.argv[1]
        loss_fn = loss_dict[loss_arg]
        print(f"Training ConvLSTM with loss {loss_arg}!")

    try:
        epoch_nums = 100
        learning_rate = 0.002
        set_model_name(f'convlstm_{loss_arg}_epoch{epoch_nums}_lr{learning_rate}')
        
        start_time = time.time()
        createModelAndTrain(loss_fn=loss_fn)
        end_time = time.time()
        print(f"Time elapsed: {end_time - start_time}")
    finally:
        # Clear the memory
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()