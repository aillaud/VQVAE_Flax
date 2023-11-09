# AutoEncoder models in Flax

This repository contains my implementation of various autoencoder models in Flax:
* [The first notebook](./Autoencoders_VAE_MNIST_JaxvPyTorch.ipynb) implements a vanilla autoencoder and a Variational AutoEncoder (VAE) on MNIST, with a comparison of the latent space representation in 2D. <br>
  The models are also implemented in PyTorch for comparison
* [The second notebook](./VQ-VAE_Flax.ipynb) implements two methods to train the codebook of a Vector Quantised-Variational AutoEncoder, also provided in two standalone python files


## References

1. [VAE](https://arxiv.org/abs/1312.6114) : Diederik P Kingma, Max Welling *Auto-Encoding Variational Bayes*,  	arXiv:1312.6114, December 2013
2. [VQ VAE](https://arxiv.org/pdf/1711.00937.pdf) : Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu *Neural Discrete Representation Learning*, arXiv:1711.00937, November 2017
