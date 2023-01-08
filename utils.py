from os.path import exists

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import time
from torch import optim
from enum import Enum
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Device(Enum):
    CPU = "cpu"
    GPU = "cuda"

device = Device.CPU.value

def print_stats(waveform, sample_rate=None, src=None):
    if src:
        print("-" * 10)
        print("Source:", src)
        print("-" * 10)
    if sample_rate:
        print("Sample Rate:", sample_rate)
    print("Shape:", tuple(waveform.shape))
    print("Dtype:", waveform.dtype)
    print(f" - Max:     {waveform.max().item():6.3f}")
    print(f" - Min:     {waveform.min().item():6.3f}")
    print(f" - Mean:    {waveform.mean().item():6.3f}")
    print(f" - Std Dev: {waveform.std().item():6.3f}")
    print()
    print(waveform)
    print()


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate
    
    figure, axes = plt.subplots(num_channels, 1, figsize=(10, 5))
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)


def plot_specgram(waveform, sample_rate, title="Spectrogram", xlim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    print("Spectrogram Shape, channels, frames : ",
          waveform.shape, num_channels, num_frames)

    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1, figsize=(10, 5))
    if num_channels == 1:
        axes = [axes]

    for c in range(num_channels):
        Pxx, freqs, bins, im = axes[c].specgram(waveform[c], Fs=sample_rate)

        print(Pxx)

        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
    figure.suptitle(title)
    plt.show(block=False)


def save_specgram(waveform, sample_rate, name):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    #time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]

    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)

        '''if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)'''

    # figure.suptitle(title)
    figure.savefig(name)
    # plt.show(block=False)
    
    plt.close(figure)
    
    return

def save_waveform(waveform, sample_rate, name):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    
    time_axis = torch.arange(0, num_frames) / sample_rate
    
    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        '''axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)'''
    #figure.suptitle(title)
    figure.savefig(name)
    #plt.show(block=False)
    
    del time_axis
    
    plt.close(figure)
    
    return



def generate_all_spectrograms(audio_dataset, sampling_rate):
    i = 0
    for data in audio_dataset.id_train:
        if exists("./spectrogram/"+data+".png"):
            print("[" + int(100 * i / audio_dataset.id_train.size) * '-' + int(100 * (1 - i / audio_dataset.id_train.size))
                  * ' ' + "]", "{} %".format(100 * i / audio_dataset.id_train.size) + 200 * ' ', end='\r')
            i += 1
            continue

        waveform = audio_dataset.getaudio(i)

        save_specgram(waveform, sampling_rate, "./spectrogram/"+data+".png")

        #plt.close()

        print("[" + int(100 * i / audio_dataset.id_train.size) * '-' + int(100 * (1 - i / audio_dataset.id_train.size))
              * ' ' + "]", "{} %".format(100 * i / audio_dataset.id_train.size) + 200 * ' ', end='\r')

        i += 1
        
def generate_all_waveforms(audio_dataset, sampling_rate):
    i = 0
    for data in audio_dataset.id_train:
        if exists("./waveform/"+data+".png"):
            print("[" + int(100 * i / audio_dataset.id_train.size) * '-' + int(100 * (1 - i / audio_dataset.id_train.size))
                  * ' ' + "]", "{} %".format(100 * i / audio_dataset.id_train.size) + 200 * ' ', end='\r')
            i += 1
            continue

        waveform = audio_dataset.getaudio(i)

        save_waveform(waveform, sampling_rate, "./waveform/"+data+".png")

        del waveform
        #plt.close()

        print("[" + int(100 * i / audio_dataset.id_train.size) * '-' + int(100 * (1 - i / audio_dataset.id_train.size))
              * ' ' + "]", "{} %".format(100 * i / audio_dataset.id_train.size) + 200 * ' ', end='\r')

        i += 1

##################################################### TRAINING / TESTING #####################################################


def get_accuracy(output, target):
    y_true = target.detach().numpy()

    y_prob = output.detach().numpy()

    y_prob = np.where(y_prob <= 0.5, 0, y_prob)
    y_prob = np.where(y_prob > 0.5, 1, y_prob)

    accuracy = metrics.accuracy_score(y_true, y_prob)

    return accuracy


def train(model, train_dataset, val_dataset, epochs=30, lr=0.01, augmentation_prob=0.5):

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=128, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model = model.train()

        start_time = time.process_time()

        running_loss = 0

        running_accuracy = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            p = torch.rand(1).item()

            if p > augmentation_prob:
                data = add_noise(data)

            output = model(data).view(-1)

            loss = criterion(output, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            accuracy = get_accuracy(output.cpu(), target.cpu())

            running_accuracy += accuracy * len(data)

            print("Epoch: {}/{} -- [{}/{} ({:.1f}%)]\tLoss: {}".format(
                epoch+1, epochs, (batch_idx + 1) * len(data), len(train_loader.dataset), 100 * (batch_idx + 1) / len(train_loader), running_loss / (batch_idx + 1)), end='\r')

        end_time = time.process_time()

        val_running_loss = 0

        val_running_accuracy = 0

        with torch.no_grad():

            model = model.eval()

            for val_batch_idx, (val_data, val_target) in enumerate(val_loader):
                val_data, val_target = val_data.to(
                    device), val_target.to(device)

                val_output = model(val_data).view(-1)

                val_loss = criterion(val_output, val_target)

                val_running_loss += val_loss

                accuracy = get_accuracy(val_output.cpu(), val_target.cpu())

                val_running_accuracy += accuracy * len(val_data)

        print("Epoch: {}/{} -- [{}/{} ({:.1f}%)]\tLoss: {}\tAccuracy: {:.3f}\tTime taken: {}".format(
            epoch+1, epochs, (batch_idx + 1) * len(data), len(train_loader.dataset), 100 * (batch_idx + 1) / len(train_loader), running_loss / (batch_idx + 1), running_accuracy / len(train_loader.dataset), end_time - start_time), end='\t')

        print("Validation Loss: {} || Validation Accuracy: {:.3f}".format(
            val_running_loss / (val_batch_idx + 1), val_running_accuracy / len(val_loader.dataset)))


def train_lstm(model, train_dataset, val_dataset, epochs=30, lr=0.01, batch_size=128, num_layers=3, hidden_size=100, augmentation_prob=0.5):

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model = model.train()

        start_time = time.process_time()

        running_loss = 0

        running_accuracy = 0

        for batch_idx, (data, target) in enumerate(train_loader):

            data, target = data.to(device), target.to(device)

            # print(target.shape)
            
            p = torch.rand(1).item()

            if p > augmentation_prob:
                data = add_noise(data)

            data = data.unsqueeze(-1)

            hidden = (torch.zeros(num_layers, data.shape[0], hidden_size).to(device), torch.zeros(
                num_layers, data.shape[0], hidden_size).to(device))  # Hidden state and cell state
            #data.shape[0] : batch_size

            #print(data.shape, hidden[0].shape)

            output, _ = model(data, hidden)

            #print("Output 1", output.shape)

            #output = output.view(-1)

            output = output.select(1, output.size(1) - 1).view(-1)

            #print("Output 2", output.shape)

            loss = criterion(output, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

            accuracy = get_accuracy(output.cpu(), target.cpu())

            running_accuracy += accuracy * len(data)

            print("Epoch: {}/{} -- [{}/{} ({:.1f}%)]\tLoss: {}".format(
                epoch+1, epochs, (batch_idx + 1) * len(data), len(train_loader.dataset), 100 * (batch_idx + 1) / len(train_loader), running_loss / (batch_idx + 1)), end='\r')

        end_time = time.process_time()

        val_running_loss = 0

        val_running_accuracy = 0

        with torch.no_grad():

            model = model.eval()

            for val_batch_idx, (val_data, val_target) in enumerate(val_loader):
                val_data, val_target = val_data.to(
                    device), val_target.to(device)

                val_data = val_data.unsqueeze(-1)

                val_hidden = (torch.zeros(num_layers, val_data.shape[0], hidden_size).to(device), torch.zeros(
                    num_layers, val_data.shape[0], hidden_size).to(device))  # Hidden state and cell state

                #print(val_data.shape, val_hidden[0].shape)

                val_output, _ = model(val_data, val_hidden)

                val_output = val_output.select(
                    1, val_output.size(1) - 1).view(-1)

                val_loss = criterion(val_output, val_target)

                val_running_loss += val_loss

                accuracy = get_accuracy(val_output.cpu(), val_target.cpu())

                val_running_accuracy += accuracy * len(val_data)

        print("Epoch: {}/{} -- [{}/{} ({:.1f}%)]\tLoss: {}\tAccuracy: {:.3f}\tTime taken: {}".format(
            epoch+1, epochs, (batch_idx + 1) * len(data), len(train_loader.dataset), 100 * (batch_idx + 1) / len(train_loader), running_loss / (batch_idx + 1), running_accuracy / len(train_loader.dataset), end_time - start_time), end='\t')

        print("Validation Loss: {} || Validation Accuracy: {:.3f}".format(
            val_running_loss / (val_batch_idx + 1), val_running_accuracy / len(val_loader.dataset)))


####################################################################################### DATA AUGMENTATION #######################################################################################

def add_noise(audio):
    # Add noise to the audio
    audio += torch.randn_like(audio) * 0.1
    return audio
