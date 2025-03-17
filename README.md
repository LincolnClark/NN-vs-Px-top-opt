# Neural Reparameterisation for Metasurface Topology Optimisation

Comparing different parameterisation methods for metasurface topology optimisation. Specifically, we compare a constrained pixel parameterisation with an unconstrained pixel parameterisation, parameterisation with a convolutional neural network and a hybrid NN and pixel method. In structural mechanics, it was found that generating a candiate design with the weights and biases of a neural network gave simpler and better performing designs [[1]](https://arxiv.org/abs/1909.04240). We were inspired by this idea and applied it to the design of periodic optical metasurfaces for analogue optical computing.

This code performs optimisation for either the angular or spectral amplitude transmittance of metasurfaces using different methods. It evaluates the root mean square between a target response and the actual metasurface transmission, the minimum feature size of the designs produced and the time taken by the optimisation method.

## Requirements
- NumPy
- PyTorch
- matplotlib
- pandas
- [TORCWA](https://github.com/kch3782/torcwa)

## Paper
- [link to paper]()
