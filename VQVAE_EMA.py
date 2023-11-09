#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any
import jax
from jax import lax, random, numpy as jnp
from jax.typing import ArrayLike
import numpy as np
from flax import linen as nn
from flax.training import train_state, orbax_utils
import optax
import orbax.checkpoint as ocp
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm import tqdm


# Globales
batch_size = 256
workers = 8
learning_rate = 0.001
epochs = 100
seed = 654321
# Commitment cost
beta = 0.25
# 1 - smoothing factor for EMA
gamma = 0.9
num_embeddings = 512
latent_dim = 64
writer = SummaryWriter("./logs/EMA")
# Calculated in the jupyter notebook
cifar10_mean = (0.49139968, 0.48215842, 0.44653094)
cifar10_std = (0.24703224, 0.24348514, 0.26158783)


def numpy_normalize(x, mean, std):
    """ Scales values from [0, 255] to [0, 1] and normalizes images with dataset moments """
    x = np.array(x, dtype=jnp.float32) / 255.
    return (x - mean) / std


def numpy_collate(batch):
    """ Stack elements in batches of numpy arrays instead of Torch tensors """
    transposed_data = list(zip(*batch))
    imgs = np.stack(transposed_data[0])
    labels = np.array(transposed_data[1])
    return imgs, labels


class ResnetBlock(nn.Module):
    # Use of GroupNorm and swish activation function, as in Stable Diffusion without time embedding (more recent than VQVAE)
    # inspiration from https://github.com/huggingface/diffusers/blob/v0.21.0/src/diffusers/models/resnet_flax.py
    out_channels: int

    def setup(self):
        self.norm1 = nn.GroupNorm()
        self.conv1 = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=1)
        self.norm2 = nn.GroupNorm()
        self.conv2 = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=1)

    def __call__(self, x):
        h = nn.swish(self.norm1(x))
        h = self.conv1(h)
        h = nn.swish(self.norm2(h))
        h = self.conv2(h)
        return x + h


class Encoder(nn.Module):
    latent_dim: int
    num_embeddings: int

    def setup(self):
        # H_out = H_in / 2
        self.conv1 = nn.Conv(self.latent_dim // 2, kernel_size=(4, 4), strides=(2, 2), padding=1)
        # H_out = H_in / 2
        self.conv2 = nn.Conv(self.latent_dim, kernel_size=(4, 4), strides=(2, 2), padding=1)
        # H_out = H_in
        self.conv3 = nn.Conv(self.latent_dim, kernel_size=(3, 3), strides=(1, 1), padding=1)
        self.res_block1 = ResnetBlock(self.latent_dim)
        self.res_block2 = ResnetBlock(self.latent_dim)

    def __call__(self, x):
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        return x


class EMA(nn.Module):
    # inspired from https://github.com/deepmind/dm-haiku/blob/master/haiku/_src/moving_averages.py
    decay: float
    shape: list
    dtype: Any = jnp.float32

    def setup(self):
        self.hidden = self.variable('batch_stats', 'hidden',
                                    lambda shape: jnp.zeros(shape, dtype=self.dtype),
                                    self.shape)
        self.average = self.variable('batch_stats', 'average',
                                     lambda shape: jnp.zeros(shape, dtype=self.dtype),
                                     self.shape)
        self.counter = self.variable('batch_stats', 'counter',
                                     lambda shape: jnp.zeros(shape, dtype=jnp.int32),
                                     ())

    def __call__(self, value, update_stats):
        counter = self.counter.value + 1
        decay = jax.lax.convert_element_type(self.decay, value.dtype)
        one = jnp.ones([], value.dtype)
        hidden = self.hidden.value * decay + value * (one - decay)
        average = hidden
        # Apply zero-debiasing
        average /= (one - jnp.power(decay, counter))
        if update_stats:
            self.counter.value = counter
            self.hidden.value = hidden
            self.average.value = average
        return average


class VectorQuantizerEMA(nn.Module):
    # Method described in the appendix of the original article https://arxiv.org/pdf/1711.00937.pdf
    # Codebook loss replaced by EMA update of the codebook
    num_embeddings: int
    embedding_dim: int
    rng: ArrayLike
    beta: float
    gamma: float
    epsilon: float = 1e-5

    def setup(self):
        # This time the codebook is non-trainable, only updated with EMA
        self.codebook = self.variable('batch_stats', 'codebook',
                                      nn.initializers.lecun_uniform(), self.rng,
                                      (self.embedding_dim, self.num_embeddings))
        # Non-trainable parameter to store the number of elements in each cluster (around each embedding)
        self.ema_cluster_size = EMA(self.gamma, (self.num_embeddings))
        # Non-trainable parameter to store the sum of elements in each cluster
        # (cf. (7) in the annex of the article)
        self.ema_dw = EMA(self.gamma, (self.embedding_dim, self.num_embeddings))

    def __call__(self, inputs, training: bool):
        flat_inputs = jnp.reshape(inputs, (-1, self.embedding_dim))
        distances = (jnp.sum(jnp.square(flat_inputs), 1, keepdims=True) -
                     2 * jnp.matmul(flat_inputs, self.codebook.value) +
                     jnp.sum(jnp.square(self.codebook.value), 0, keepdims=True))
        encoding_indices = jnp.argmin(distances, 1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings, dtype=distances.dtype)
        flat_quantized = jnp.take(self.codebook.value, encoding_indices, axis=1).swapaxes(1, 0)
        quantized = jnp.reshape(flat_quantized, inputs.shape)
        loss = self.beta * jnp.mean(jnp.square(jax.lax.stop_gradient(quantized) - inputs))

        # Update the codebook with EMA only if the model is in training mode
        if training:
            # Number of closest inputs for each embedding in the codebook (size: num_embeddings)
            cluster_size = jnp.sum(encodings, axis=0)
            updated_ema_cluster_size = self.ema_cluster_size(cluster_size, update_stats=True)
            # Sum of inputs within clusters (equation (7))
            dw = jnp.matmul(flat_inputs.T, encodings)
            updated_ema_dw = self.ema_dw(dw, update_stats=True)
            # Laplace smoothing of cluster size / nb of elements
            n = jnp.sum(updated_ema_cluster_size)
            updated_ema_cluster_size = ((updated_ema_cluster_size + self.epsilon) /
                                        (n + self.num_embeddings * self.epsilon) * n)
            normalised_updated_ema_w = updated_ema_dw / jnp.reshape(updated_ema_cluster_size, [1, -1])

            self.codebook.value = normalised_updated_ema_w

        # Straight Through Estimator : returns the value of the quantized latent space
        # and multiplies gradient by 1 in chain rule, as input = output
        # - i.e. gradient from the decoder passed directly to the encoder in backprop phase
        ste = inputs + lax.stop_gradient(quantized - inputs)

        # Perplexity computation
        avg_probs = jnp.mean(encodings, 0)
        perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

        return ste, perplexity, loss


class Upsample(nn.Module):
    # Replace TransposeConv2d by Upsample block to avoid checkerboard artifacts
    # Upsample by factor upfactor, followed by convolution so that the model can learn parameters
    # cf. https://distill.pub/2016/deconv-checkerboard/
    out_channels: int
    upfactor: int

    def setup(self):
        # H_out = H_in
        self.conv = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(1, 1), padding=1)

    def __call__(self, inputs):
        batch, height, width, channels = inputs.shape
        hidden_states = jax.image.resize(inputs,
                                         shape=(batch, height * self.upfactor, width * self.upfactor, channels),
                                         method="bilinear")
        return self.conv(hidden_states)


class Decoder(nn.Module):
    latent_dim: int

    def setup(self):
        self.res_block1 = ResnetBlock(self.latent_dim)
        self.res_block2 = ResnetBlock(self.latent_dim)
        self.upsample1 = Upsample(out_channels=self.latent_dim // 2, upfactor=2)
        self.upsample2 = Upsample(out_channels=3, upfactor=2)

    def __call__(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = nn.relu(self.upsample1(x))
        return self.upsample2(x)


class VQVAE_EMA(nn.Module):
    latent_dim: int
    num_embeddings: int
    rng: ArrayLike
    beta: float
    gamma: float
    epsilon: float = 1e-5

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim, num_embeddings=self.num_embeddings)
        self.vector_quantizer = VectorQuantizerEMA(num_embeddings=self.num_embeddings,
                                                   embedding_dim=self.latent_dim, rng=self.rng,
                                                   beta=self.beta, gamma=self.gamma, epsilon=self.epsilon)
        self.decoder = Decoder(latent_dim=self.latent_dim)

    def __call__(self, x, training):
        ze = self.encoder(x)
        zq, perplexity, loss = self.vector_quantizer(ze, training)
        return self.decoder(zq), perplexity, loss


def create_train_state(model, rng, learning_rate):
    """ Instanciate the state outside of the training loop """
    params = model.init(rng, jnp.ones([1, 32, 32, 3]))['params']
    opti = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply,
                                         params=params, tx=opti)


class TrainStateEMA(train_state.TrainState):
    batch_stats: Any


def create_train_state_EMA(model, rng, learning_rate):
    """ Instanciate the state outside of the training loop """
    variables = model.init(rng, jnp.ones([1, 32, 32, 3]), training=True, mutable=True)
    opti = optax.adam(learning_rate)
    return TrainStateEMA.create(apply_fn=model.apply, params=variables['params'],
                                batch_stats=variables['batch_stats'], tx=opti)


@jax.jit
def train_step_EMA(state, batch):
    """ Train for a single step """
    def loss_fn(params, batch_stats):
        (x_recon, perplexity, commitment_loss), _ = state.apply_fn({'params': params,
                                                                    'batch_stats': batch_stats},
                                                                    batch[0], training=True,
                                                                    mutable=True)
        recon_loss = optax.squared_error(predictions=x_recon, targets=batch[0]).mean()
        metrics = {"perplexity": perplexity, "recon_loss": recon_loss,
                   "commitment_loss": commitment_loss}
        return recon_loss + commitment_loss, metrics

    # Update parameters with gradient descent
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, metrics), grads = grad_fn(state.params, state.batch_stats)
    state = state.apply_gradients(grads=grads)
    return state, loss, metrics


@jax.jit
def eval_step_EMA(state, batch):
    """ Computes the metric on the test batch (code already included in train_step for train batch) """
    x_recon, perplexity, commitment_loss = state.apply_fn({'params': state.params,
                                                           'batch_stats': state.batch_stats},
                                                          batch[0], training=False)
    recon_loss = optax.l2_loss(predictions=x_recon, targets=batch[0]).mean()
    metrics = {"perplexity": perplexity, "recon_loss": recon_loss,
               "commitment_loss": commitment_loss}
    return state, recon_loss + commitment_loss, metrics


def main():
    trainset = CIFAR10(root="./data", train=True, download=True,
                       transform=lambda x: numpy_normalize(x, cifar10_mean, cifar10_std))
    testset = CIFAR10(root="./data", train=False, download=True,
                      transform=lambda x: numpy_normalize(x, cifar10_mean, cifar10_std))
    trainloader = DataLoader(trainset, batch_size, shuffle=True, num_workers=workers,
                             collate_fn=numpy_collate, drop_last=True)
    testloader = DataLoader(testset, batch_size, shuffle=False, num_workers=workers,
                            collate_fn=numpy_collate, drop_last=True)

    # Model initialization
    init_rng = random.PRNGKey(seed)
    init_rng, codebook_rng = random.split(init_rng)
    model = VQVAE_EMA(num_embeddings=num_embeddings, latent_dim=latent_dim, rng=codebook_rng,
                      beta=beta, gamma=gamma)
    state = create_train_state_EMA(model, init_rng, learning_rate)
    del init_rng

    # Training loop
    epoch = tqdm(range(epochs))
    for e in epoch:
        loss_train, loss_test, perplexity_train, perplexity_test = [], [], [], []
        recon_loss, commitment_loss = [], []
        torch.manual_seed(seed)
        for batch in trainloader:
            state, loss, metrics = train_step_EMA(state, batch)
            loss_train.append(loss)
            perplexity_train.append(metrics["perplexity"].item())
            recon_loss.append(metrics["recon_loss"].item())
            commitment_loss.append(metrics["commitment_loss"].item())
        writer.add_scalars('losses_train', {'recon': np.mean(recon_loss),
                                            'commitment': np.mean(commitment_loss)}, e)

        # Compute metrics on the test set after each training epoch
        test_state = state
        for batch in testloader:
            test_state, loss, metrics = eval_step_EMA(test_state, batch)
            loss_test.append(loss)
            perplexity_test.append(metrics["perplexity"].item())
        writer.add_scalars('loss', {'train': np.mean(loss_train),
                                    'test': np.mean(loss_test)}, e)
        writer.add_scalars('perplexity', {'train': np.mean(perplexity_train),
                                          'test': np.mean(perplexity_test)}, e)
        epoch.set_description(f"Epoch: {e+1}/{epochs} - Train Loss: {np.mean(loss_train):.4f} - Test loss: {np.mean(loss_test):.4f}")

    # Save model
    ckpt = {'model': state}
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(ckpt)
    orbax_checkpointer.save(f"./vqvae_ema_gamma{gamma}_lr{learning_rate}_e{epochs}", ckpt, save_args=save_args)


if __name__ == "__main__":
    main()
