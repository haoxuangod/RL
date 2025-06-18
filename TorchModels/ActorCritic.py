import torch
import torch.nn as nn

from RL.TorchModels.BaseNet import BaseNet
from RL.TorchModels.ModelInput import MLPInput


class ActorOutput(BaseNet):
    def __init__(self,state_dim,action_dim,hidden_size,a_min=-1.0,a_max=1.0,log_dir=None,name=None):
        super().__init__(log_dir=log_dir,name=name)
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.a_min = a_min
        self.a_max = a_max
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU()
        )
        self.mu_head = nn.Linear(hidden_size, action_dim)
        self.log_std_head = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.fc(x)
        mu_raw = self.mu_head(h)
        mu = torch.tanh(mu_raw)
        a_min=torch.tensor(self.a_min,device=x.device)
        a_max = torch.tensor(self.a_max,device=x.device)
        mu = a_min + (mu + 1) * (a_max - a_min) / 2  # bound actions to [a_min,a_max]
        log_std = self.log_std_head.clamp(-20, 2)  # 限制在 e^{-20} ~ e^{2} 之间
        std = log_std.exp()
        return mu, std
class CriticOutput(BaseNet):
    def __init__(self,state_dim,hidden_size,log_dir=None,name=None):
        super().__init__(log_dir=log_dir,name=name)
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    pass
if __name__ == "__main__":
    main()

