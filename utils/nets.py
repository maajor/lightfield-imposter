import torch
from torch import nn
import torch.nn.functional as F

# Model
class ImposterNN(nn.Module):
    def __init__(self, D=6, W=256, RES=128):
        """ 
        """
        super(ImposterNN, self).__init__()
        self.D = D
        self.W = W
        self.RES = RES
        self.input_ch = 4 # phi, theta, x, y

        self.fc_in = nn.Linear(4, W)
        self.fcs = nn.ModuleList([nn.Linear(W, W) for i in range(D)])
        self.fc_out = nn.Linear(W, 4)

        x = torch.arange(self.RES)
        x_grid, y_grid = torch.meshgrid([x,x])
        self.meshgrid = torch.stack((x_grid, y_grid)).float().cuda()
        # renormalize to -1, 1
        self.meshgrid = (self.meshgrid / self.RES - 0.5) * 2
        self.meshgrid.requires_grad = False

    def forward(self, x):
        # concatenate meshgrid to angle input
        batch = x.shape[0]
        x = x.view(batch, -1, 1, 1).repeat(1, 1, self.RES, self.RES)
        meshgrid = self.meshgrid.unsqueeze(0).repeat(batch, 1, 1, 1)
        x = torch.cat([x, meshgrid],1).permute(0,2,3,1)

        x = self.fc_in(x)
        x = F.relu(x)
        for i in range(self.D):
            x = self.fcs[i](x)
            x = F.relu(x)
        x = self.fc_out(x)

        return x