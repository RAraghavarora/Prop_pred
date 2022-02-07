import torch
from torch import nn
from torchviz import make_dot


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(528, 16)
        self.lin2 = nn.Linear(56, 16)
        self.lin4 = nn.Linear(16, 1)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        slatm = x[:, :528]
        elec = x[:, 528:]
        layer1 = self.lin1(slatm)
        layer1 = nn.functional.elu(layer1)

        concat = torch.cat([layer1, elec], dim=1)
        # concat = nn.functional.elu(concat)

        layer2 = self.lin2(concat)
        layer2 = nn.functional.elu(layer2)
        layer4 = self.lin4(layer2)

        return layer4


model = torch.load('C:/Users/raghav/bob/16/20000/model.pt')
torch.onnx.export(model)
x = torch.randn(1, 568)
y = model(x)
# make_dot(y.mean(), params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True
#          ).render('Plots/plot')