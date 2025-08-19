import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

data = np.load("demos_circle.npy", allow_pickle=True)
obs = np.stack(data[:,0])
acts = np.stack(data[:,1])

ds = TensorDataset(torch.tensor(obs,dtype=torch.float32), torch.tensor(acts,dtype=torch.float32))
loader = DataLoader(ds, batch_size=128, shuffle=True)

class BCNet(nn.Module):
    def __init__(self, obs_dim=5, act_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim,128), nn.ReLU(),
            nn.Linear(128,128), nn.ReLU(),
            nn.Linear(128,act_dim), nn.Tanh()
        )
    def forward(self,x): return self.net(x)

model = BCNet()
opt = optim.Adam(model.parameters(), lr=1e-3)
for ep in range(30):
    loss_sum = 0
    for x,y in loader:
        pred = model(x)
        loss = ((pred-y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        loss_sum += loss.item()
    print(ep, loss_sum)
torch.save(model.state_dict(), "bc_model.pth")
