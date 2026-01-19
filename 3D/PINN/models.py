from config import*

class PINN(nn.Module):
    def __init__(self, num_hidden_layers, hidden_neurons):

        super(PINN, self).__init__()

        self.activation = nn.Tanh()

        layers = []

        layers.append(nn.Linear(3, hidden_neurons))
        layers.append(self.activation)

        for _ in range(num_hidden_layers-1):
            layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            layers.append(self.activation)

        layers.append(nn.Linear(hidden_neurons, 1))
        self.network = nn.Sequential(*layers)

        for m in self.network:
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x, y, z):
        inputs = torch.cat([x, y, z], dim=1)  
        output = self.network(inputs)
        return output