import torch
import torch.nn as nn

class MLPModel(nn.Module):
    
    def __init__(self, sample_rate=44100) -> None:
        super().__init__()
        
        """ self.fc1 = nn.Linear(sample_rate, 2*sample_rate)
        
        self.fc2 = nn.Linear(2*sample_rate, 4*sample_rate)
        
        self.fc3 = nn.Linear(4*sample_rate, 4*sample_rate)
        
        self.fc4 = nn.Linear(4*sample_rate, 2*sample_rate)
        
        self.fc5 = nn.Linear(2*sample_rate, 1) """
        
        self.fc1 = nn.Linear(sample_rate, 2000)
        
        self.fc2 = nn.Linear(2000, 1000)
        
        self.fc3 = nn.Linear(1000, 500)
        
        self.fc4 = nn.Linear(500, 250)
        
        self.fc5 = nn.Linear(250, 1)
        
    def forward(self, x):
        x = self.fc1(x)
        
        x = torch.relu(x)
        
        x = self.fc2(x)
        
        x = torch.relu(x)
        
        x = self.fc3(x)
        
        x = torch.relu(x)
        
        x = self.fc4(x)
        
        x = torch.relu(x)
        
        x = self.fc5(x)
        
        return torch.sigmoid(x)
    
    
class ConvBlock1d(nn.Module):
    
    def __init__(self, input_channels=1, output_channels=1) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv1d(input_channels, output_channels, 5)
        
        self.max_pool = nn.MaxPool1d(5)
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.max_pool(x)
        
        return x
    
class ConvBlock2d(nn.Module):
    
    def __init__(self, input_channels=3, output_channels=3) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(input_channels, output_channels, (3,3))
        
        self.max_pool = nn.MaxPool2d((3,3))
        
        
    def forward(self, x):
        
        x = self.conv1(x)
        
        x = self.max_pool(x)
        
        return x

class CNNModel(nn.Module):
    
    def __init__(self, input_channels=1) -> None:
        super().__init__()
        
        self.input_channels = input_channels
        
        self.conv_block1 = ConvBlock1d(input_channels, 10)
        
        self.conv_block2 = ConvBlock1d(10, 20)
        
        self.conv_block3 = ConvBlock1d(20, 30)
        
        self.conv_block4 = ConvBlock1d(30, 60)
        
        self.classifier = nn.Sequential(
            nn.Linear(660, 1250), #8000 Hz
            #nn.Linear(1440, 1250), #16000 Hz            
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1250, 625),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(625, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        shape = x.shape
        
        #print("Input Shape : ", shape)
        
        x = torch.reshape(x, (shape[0], self.input_channels, shape[1]))
        
        x = self.conv_block1(x)
        
        x = self.conv_block2(x)
        
        x = self.conv_block3(x)
        
        x = self.conv_block4(x)
        
        #print("Shape After convolutions : ", x.shape)
        
        x = self.classifier(torch.flatten(x, start_dim=1))
        
        return x
    
class SpecCNNModel(nn.Module):
    
    def __init__(self, input_channels=3) -> None:
        super().__init__()
        
        self.input_channels = input_channels
        
        self.conv_block1 = ConvBlock2d(input_channels, 10)
        
        self.conv_block2 = ConvBlock2d(10, 20)
        
        self.conv_block3 = ConvBlock2d(20, 30)
        
        self.conv_block4 = ConvBlock2d(30, 60)
        
        self.classifier = nn.Sequential(
            nn.Linear(480, 1250),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(1250, 625),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(625, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        
        shape = x.shape
        
        #print("Input Shape : ", shape)
        
        #x = torch.reshape(x, (shape[0], self.input_channels, shape[1]))
        
        x = self.conv_block1(x)
        
        x = self.conv_block2(x)
        
        x = self.conv_block3(x)
        
        x = self.conv_block4(x)
        
        #print("Shape After convolutions : ", x.shape)
        
        x = self.classifier(torch.flatten(x, start_dim=1))
        
        return x

class RNNModel(nn.Module):
    
    def __init__(self) -> None:
        super().__init__()
        
    def forward(self, x):
        pass
    
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers) -> None:
        super().__init__()
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        self.classifier = nn.Linear(hidden_size, 1)
        
    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        
        x = self.classifier(x)
        
        return torch.sigmoid(x), hidden