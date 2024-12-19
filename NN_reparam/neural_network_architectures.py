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
    
class UpsampleBlock(torch.nn.Module):
    def __init__(self, channel_in, channel_out, upscale_factor, kernel_size, shape, offset = True) -> None:
        super().__init__()
        
        self.non_lin = nn.Tanh()
        self.scale = upscale_factor
        self.norm = nn.BatchNorm2d(channel_in, affine = False)
        
        self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, bias = False, padding_mode = "circular", padding = "same")
        self.offset = offset
        if offset == True:
            self.offset = OffsetLayer(shape)
        else:
            self.conv = nn.Conv2d(channel_in, channel_out, kernel_size, bias = True, padding_mode = "circular", padding = "same")

    
    def forward(self, x):
        output = nn.functional.interpolate(self.non_lin(x), mode = "bilinear", scale_factor = self.scale)
        if self.offset == False:
            return self.conv(self.norm(output))
        else:
            return self.offset(self.conv(self.norm(output)))
    
class NeuralNetwork(nn.Module):
    def __init__(self, n, m, ker_size = 5,
                 scale = [1, 2, 2, 2, 2, 1], channels = [256, 128, 64, 32, 16, 1],
                 offset = [True, True, True, True, True, True], dense_channels = 1,
                 blur = None, device = "cuda"):
        super().__init__()

        modules = [RandInput(n*m),
                   nn.Linear(n * m, dense_channels * n * m)]
        
        # Add appropriate number of upsampling blocks
        shape = [1, dense_channels, n, m]
        for i in range(len(scale)):
            shape[1] = channels[i]
            shape[2] = scale[i] * shape[2]
            shape[3] = scale[i] * shape[3]

            if i == 0:
                chan = dense_channels
            else:
                chan = channels[i - 1]
            
            modules.append(UpsampleBlock(chan, channels[i], scale[i], ker_size, tuple(shape), offset[i]))

        self.blur = blur
        if blur is not None:
            # Create kernel
            x = torch.linspace(-1, 1, blur)
            y = torch.linspace(-1, 1, blur)
            xx, yy = torch.meshgrid(x, y, indexing = "xy")
            w = 1 - torch.sqrt(xx**2 + yy**2)
            w[w < 0] = 0
            w = torch.reshape(w, (1, 1, blur, blur))
            self.blur_kernel = w
            print(self.blur_kernel.device)

        self.nn_modules = nn.ModuleList(modules)
        self.n_blocks = len(modules)

        self.n = n
        self.m = m
        self.n_dense_chan = dense_channels


    def forward(self, x):
        x = torch.flatten(x)

        x = self.nn_modules[0](x) # bias to all pixels
        x = self.nn_modules[1](x) # Linear layer

        x = torch.reshape(x, (1, self.n_dense_chan, self.n, self.m))

        for i in range(2, self.n_blocks): # Convolution blocks
            x = self.nn_modules[i](x)

        if self.blur is not None:
            x = nn.functional.pad(x, (self.blur//2, self.blur//2, self.blur//2, self.blur//2), mode = "circular")
            x = nn.functional.conv2d(x, self.blur_kernel, bias = None)

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