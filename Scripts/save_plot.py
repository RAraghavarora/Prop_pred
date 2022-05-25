import torch
from torch import nn
from torchviz import make_dot
import hiddenlayer as hl


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.lin1 = nn.Linear(17895, 16)
        self.lin2 = nn.Linear(56, 2)
        self.lin4 = nn.Linear(2, 1)
        self.apply(init_weights)
        # self.flatten = nn.Flatten(-1,0)

    def forward(self, x):
        slatm = x[:, :17895]
        elec = x[:, 17895:]
        layer1 = self.lin1(slatm)
        layer1 = nn.functional.elu(layer1)

        concat = torch.cat([layer1, elec], dim=1)
        # concat = nn.functional.elu(concat)

        layer2 = self.lin2(concat)
        layer2 = nn.functional.elu(layer2)
        layer4 = self.lin4(layer2)

        return layer4


model = torch.load('withdft/slatm/20000/model.pt', map_location=torch.device('cpu'))
# torch.onnx.export(model)
x = torch.randn(1, 17935)
y = model(x)

# transforms = [hl.transforms.Prune('Constant'), hl.transforms.FoldDuplicates()]
graph = hl.build_graph(model, x)
graph.theme = hl.graph.THEMES['blue'].copy()
graph.save('rnn_hiddenlayer2', format='png')

# make_dot(y.mean(), params=dict(list(model.named_parameters())), show_attrs=True, show_saved=True
#          ).render('Plots/plot')
