from config import*



class WPINN(nn.Module):
    def __init__(self, input_size, num_hidden_layers1, num_hidden_layers2, hidden_neurons, family_size):
        
        super(WPINN, self).__init__()
        
        self.activation = nn.Tanh()
        
        # First network: processes each (x,t) point to create single feature
        first_stage_layers = []
        
        # Input layer
        first_stage_layers.append(nn.Linear(3, hidden_neurons)) 
        first_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers1-1):
            first_stage_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            first_stage_layers.append(self.activation)
        
        # Output of first stage: single feature per point
        first_stage_layers.append(nn.Linear(hidden_neurons, 1))
        self.first_stage = nn.Sequential(*first_stage_layers)
        
        # Second network: processes all point features to create global coefficients
        second_stage_layers = []
        
        # Input size is now just input_size (number of points) since each point has 1 feature
        second_stage_layers.append(nn.Linear(input_size, hidden_neurons))
        second_stage_layers.append(self.activation)
        
        for _ in range(num_hidden_layers2-1):  # Fewer layers in second stage
            second_stage_layers.append(nn.Linear(hidden_neurons, hidden_neurons))
            second_stage_layers.append(self.activation)
        
        # Final layer outputs the wavelet coefficients
        second_stage_layers.append(nn.Linear(hidden_neurons, family_size))
        self.second_stage = nn.Sequential(*second_stage_layers)
        
        # Initialize weights
        for network in [self.first_stage, self.second_stage]:
            for m in network:
                if isinstance(m, nn.Linear):
                    init.xavier_uniform_(m.weight)
                    init.constant_(m.bias, 0)
        
        self.bias = nn.Parameter(torch.tensor(0.5))


    def forward(self, x, y, z):
        inputs = torch.stack([x, y, z], dim=-1)  
        
        # First stage: process each point to get single feature
        point_features = self.first_stage(inputs)  
        point_features = point_features.squeeze(-1)  
        # Second stage: generate global coefficients from all point features
        coefficients = self.second_stage(point_features)  

        bias = self.bias
        
        return coefficients, bias
    



class AWPINN(nn.Module):
    def __init__(self, wx, bx, wy, by, coeff, bias):
        super(AWPINN, self).__init__()
        num_wavelets = len(wx)
        
        # Make these parameters trainable
        self.wx = nn.Parameter(wx.reshape(num_wavelets, 1))
        self.bx = nn.Parameter(bx)
        self.wy = nn.Parameter(wy.reshape(num_wavelets, 1))
        self.by = nn.Parameter(by)
    
        self.output_weight = nn.Parameter(coeff.reshape(1, -1))
        self.output_bias = nn.Parameter(torch.tensor(bias))

    def forward(self, x, y):

        # with torch.no_grad():
        x = x.view(-1, 1)
        y = y.view(-1, 1)
            
        x_transformed = x @ self.wx.t() + self.bx
        y_transformed = y @ self.wy.t() + self.by
        
        x_exp = torch.exp(-x_transformed**2 / 2)
        y_exp = torch.exp(-y_transformed**2 / 2)
        
        x_wavelets = -x_transformed * x_exp
        y_wavelets = -y_transformed * y_exp
        
        d2x_wavelets = (self.wx.t()**2) * x_transformed * (3 - x_transformed**2) * x_exp
        d2y_wavelets = (self.wy.t()**2) * y_transformed * (3 - y_transformed**2) * y_exp

        
        wavelets = x_wavelets * y_wavelets
        output = (wavelets @ self.output_weight.t()).squeeze() + self.output_bias
        
        # Derivatives
        d2u_dx2 = (d2x_wavelets * y_wavelets) @ self.output_weight.t()
        d2u_dy2 = (x_wavelets * d2y_wavelets) @ self.output_weight.t()

        
        return output, d2u_dx2.squeeze(), d2u_dy2.squeeze()
        