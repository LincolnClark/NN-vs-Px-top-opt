import torch
from torch import nn

class OffsetLayer(torch.nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        bias_value = torch.randn(shape)
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer
    
class RandInput(torch.nn.Module):
    def __init__(self, len) -> None:
        super().__init__()
        bias_value = torch.randn((len))
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer
    
class BiasLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bias_value = torch.randn((1))
        self.bias_layer = torch.nn.Parameter(bias_value)
    
    def forward(self, x):
        return x + self.bias_layer
    
class UpsampleBlock(torch.nn.Module):
    def __init__(self, upscale_factor, shape, offset = True) -> None:
        super().__init__()
        
        self.scale = upscale_factor
        self.norm = nn.BatchNorm2d(1, affine = False)
        
        self.offset = offset
        if offset == True:
            self.offset = OffsetLayer(shape)
        else:
            self.offset = BiasLayer()

    def forward(self, x):
        output = nn.functional.interpolate(x, mode = "bilinear", scale_factor = self.scale)
        return self.offset(self.norm(output))
    
class NeuralNetwork(nn.Module):
    def __init__(self, n, m,
                 scale = [1, 2, 2, 2, 2, 1],
                 offset = [True, True, True, True, True, True],
                 device = "cuda"):
        super().__init__()

        modules = [RandInput(n*m)]
        
        # Add appropriate number of upsampling blocks
        shape = [1, 1, n, m]
        for i in range(len(scale)):
            shape[1] = 1
            shape[2] = scale[i] * shape[2]
            shape[3] = scale[i] * shape[3]
            
            modules.append(UpsampleBlock(scale[i], tuple(shape), offset[i]))

        self.nn_modules = nn.ModuleList(modules)
        self.n_blocks = len(modules)

        self.n = n
        self.m = m


    def forward(self, x):
        x = torch.flatten(x)

        x = self.nn_modules[0](x) # bias to all pixels

        x = torch.reshape(x, (1, 1, self.n, self.m))

        for i in range(1, self.n_blocks): # Convolution blocks
            x = self.nn_modules[i](x)

        x = x[0, 0, :, :]

        return x
    

def train_loop_dual_angle(model, loss_fn, optimiser, x, beta, hist, c_hist, 
                          options, angles, layers, targets, targetp, geom, sim_dtype):

    # Set the model to training mode
    model.train()
    
    # Forward simulation
    gamma = model(x)

    if True in torch.isnan(gamma):
        print("NaN in gamma")
    # Binarisation
    kappa = torch.special.expit(beta * gamma)

    cost = loss_fn(kappa, options, angles, layers, targets, targetp, geom, sim_dtype)

    # Backpropagation
    cost.backward()
    optimiser.step()
    optimiser.zero_grad()

    hist.append(kappa.detach().cpu().numpy())
    c_hist.append(cost.detach().cpu().numpy())

def train_loop_single_angle(model, loss_fn, optimiser, x, beta, hist, c_hist, 
                            options, angles, layers, target, pol, geom, sim_dtype):

    # Set the model to training mode
    model.train()
    
    # Forward simulation
    gamma = model(x)

    if True in torch.isnan(gamma):
        print("NaN in gamma")
    # Binarisation
    kappa = torch.special.expit(beta * gamma)

    cost = loss_fn(kappa, options, angles, layers, target, pol, geom, sim_dtype)

    # Backpropagation
    cost.backward()
    optimiser.step()
    optimiser.zero_grad()

    hist.append(kappa.detach().cpu().numpy())
    c_hist.append(cost.detach().cpu().numpy())

def train_loop_dual_spectral(model, loss_fn, optimiser, x, beta, hist, c_hist, 
                             options, wavelengths, layers, targets, targetp, geom, sim_dtype):

    # Set the model to training mode
    model.train()
    
    # Forward simulation
    gamma = model(x)

    if True in torch.isnan(gamma):
        print("NaN in gamma")
    # Binarisation
    kappa = torch.special.expit(beta * gamma)

    cost = loss_fn(kappa, options, wavelengths, layers, targets, targetp, geom, sim_dtype)

    # Backpropagation
    cost.backward()
    optimiser.step()
    optimiser.zero_grad()

    hist.append(kappa.detach().cpu().numpy())
    c_hist.append(cost.detach().cpu().numpy())

def train_loop_single_spectral(model, loss_fn, optimiser, x, beta, hist, c_hist, 
                             options, wavelengths, layers, target, pol, geom, sim_dtype):

    # Set the model to training mode
    model.train()
    
    # Forward simulation
    gamma = model(x)

    if True in torch.isnan(gamma):
        print("NaN in gamma")
    # Binarisation
    kappa = torch.special.expit(beta * gamma)

    cost = loss_fn(kappa, options, wavelengths, layers, target, pol, geom, sim_dtype)

    # Backpropagation
    cost.backward()
    optimiser.step()
    optimiser.zero_grad()

    hist.append(kappa.detach().cpu().numpy())
    c_hist.append(cost.detach().cpu().numpy())