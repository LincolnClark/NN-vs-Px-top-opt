import torch
from torch import nn


class BiasLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        bias_value = torch.randn((1))
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
    
class NeuralNetwork(nn.Module):
    def __init__(self, n, m, scale):
        super().__init__()

        self.flatten = nn.Flatten()

        self.nn_modules = nn.ModuleList([
            RandInput(n*m),
            nn.Linear(n * m, n * m),
            nn.Tanh(),
            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(1, 256, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(256, 128, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(128, 64, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(64, 32, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),

            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(32, 16, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer(),
            nn.Tanh(),
            nn.LazyBatchNorm2d(affine = False),
            nn.Conv2d(16, 1, 5, bias = False, padding_mode = "circular", padding = "same"),
            BiasLayer()]
        )

        self.normalise_loc = []#[3, 8, 13, 18, 22]
        self.upscale_loc = [7, 12, 17, 22]
        self.n_modules = len(self.nn_modules)
        self.n_functionals = len(self.normalise_loc) + len(self.upscale_loc)

        self.n = n
        self.m = m
        self.scale = scale


    def forward(self, x):
        x = self.flatten(x)
        x = self.nn_modules[0](x)
        x = self.nn_modules[1](x)
        x = torch.reshape(x, (1, 1, self.n, self.m))

        j = 2
        for i in range(2, self.n_modules + self.n_functionals):
            #if i in self.normalise_loc:
            #    x = nn.functional.normalize(x)
            if i in self.upscale_loc:
                x = nn.functional.interpolate(x, mode = "bilinear", scale_factor = self.scale)
            else:
                x = self.nn_modules[j](x)
                j = j + 1

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

def train_loop_spectral(model, loss_fn, optimiser, x, beta, hist, c_hist, 
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