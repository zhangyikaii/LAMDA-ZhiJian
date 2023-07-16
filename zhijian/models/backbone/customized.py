import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, args, num_classes):
        super(MLP, self).__init__()
        self.args = args
        self.image_size = 224
        self.fc1 = nn.Linear(self.image_size * self.image_size * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc2(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.fc3(x)
        return x

MyModels = {
    'mlp': MLP
}

