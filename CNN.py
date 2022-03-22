import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CNN, self).__init__()

        self.conv1=nn.Sequential(
            nn.Conv1d(
            in_channels=1,
            out_channels=16,
            kernel_size=5,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=5,
            stride=1,
            padding=2
        ),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
        )
        self.out=nn.Linear(32*7*7, 10)

    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0), -1)
        output=self.out(x)
        return output