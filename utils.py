import os
import torch
import modules
import operator
import numpy as np
import torch.nn as nn
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import OlshausenDataset, MNISTVariant, \
    CUB2011Dataset, CIFAR10Dataset, InterpolationDataset


def is_iterable(x):
    """Source: https://stackoverflow.com/a/1952481."""
    try:
        iter(x)
        return True
    except TypeError:
        return False


def product(iterable):
    """Source: https://stackoverflow.com/a/595409."""
    return reduce(operator.mul, iterable, 1)


def to_img(x):
    if len(x.size()) < 4:
        h = w = int(np.sqrt(product(list(x.size())[1:])))
        x = x.view(x.size(0), 1, h, w)
    return x


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def zero_mask(x, zero_frac):
    """Apply zero-masking noise to a PyTorch tensor.
    Returns noisy X and a bitmask describing the affected locations."""
    bitmask = torch.rand_like(x) > zero_frac  # approx. ZERO_FRAC zeros
    return x * bitmask.float(), bitmask  # assumes the minimum value is 0


def add_gaussian(x, gaussian_stdev):
    """Apply isotropic additive Gaussian noise to a PyTorch tensor.
    Returns noisy X and a bitmask describing the affected locations."""
    noise = torch.empty_like(x).normal_(0, gaussian_stdev)
    return x + noise, torch.ones_like(x, dtype=torch.uint8)


def salt_and_pepper(x, sp_frac, minval=0.0, maxval=1.0):
    """Apply salt-and-pepper noise to a PyTorch tensor.
    Returns noisy X and a bitmask describing the affected locations."""
    rand = torch.rand_like(x)
    min_idxs = rand < (sp_frac / 2.0)
    max_idxs = rand > (1.0 - sp_frac / 2.0)
    x_sp = x.clone()
    x_sp[min_idxs] = minval
    x_sp[max_idxs] = maxval
    return x_sp, torch.clamp(min_idxs + max_idxs, 0, 1)


def plot_first_layer_weights(model, weight_h=None, weight_w=None, block_on_viz=False):
    weights = model.get_first_layer_weights()
    print('shape of first-layer weights: %r' % (weights.shape,))

    if len(weights.shape) == 4:
        # weights for convolutional layer
        n, c, h, w = weights.shape
        weights = np.reshape(weights, (n * c, h, w))

    if not block_on_viz:
        plt.ion()
        plt.show()

    n = weights.shape[0]
    if n < 50:
        nrows = int(np.sqrt(n))
        ncols = int(np.ceil(n // nrows))
    else:
        nrows, ncols = 5, 10
    fig, ax = plt.subplots(nrows, ncols)

    if nrows == 1 and ncols == 1:
        ax = [[ax]]
    elif nrows == 1:
        ax = [[col for col in ax]]
    elif ncols == 1:
        ax = [[row] for row in ax]

    i = 0
    for row in ax:
        for col in row:
            weight = weights[i]
            if len(weight.shape) == 1:
                if not weight_h or not weight_w:
                    # infer height and width of weight, assuming it is square
                    weight_h = weight_w = int(np.sqrt(weight.size))
                weight = np.reshape(weight, (weight_h, weight_w))
            col.imshow(weight, cmap='gray')
            col.axis('off')
            i += 1

    if not block_on_viz:
        plt.pause(10)
        plt.close()
    else:
        plt.show()


def save_image_wrapper(img, filepath):
    save_image(img, filepath)
    print('[o] saved image to %s' % filepath)


def init_model(model_class, restore_path, restore_required, **model_kwargs):
    # instantiate model
    model = getattr(modules, model_class)(**model_kwargs).cuda()
    print('instantiated a model of type %s' % model.__class__.__name__)
    # restore parameters
    if restore_required or restore_path:
        if restore_required or os.path.exists(restore_path):
            model.load_state_dict(torch.load(restore_path))
            print('restored "%s" model from %s' % (model_class, restore_path))
        else:
            print('warning: checkpoint %s not found, skipping...' % restore_path)
    return model


def init_loss(loss_type, **loss_kwargs):
    Loss = {
        'mse': nn.MSELoss,
        'bce': nn.BCELoss,
        'binary_cross_entropy': nn.BCELoss,
        'nll': nn.NLLLoss,
        'vae': modules.VAELoss,
    }[loss_type.lower()]
    print('using %r as the loss' % (Loss,))
    return Loss(**loss_kwargs)


def init_data_loader(dataset_key,
                     train_ver=True,
                     batch_size=128,
                     olshausen_path=None,
                     olshausen_step_size=1,
                     cub_folder=None):

    dataset_key = dataset_key.lower()
    if dataset_key.startswith('mnist') \
            or dataset_key in MNISTVariant.variant_options:
        # MNIST or MNIST variant
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize),
        ])
        variant = None if dataset_key == 'mnist' else dataset_key
        dataset = MNISTVariant('./data',
                               train=train_ver,
                               transform=img_transform,
                               download=True,
                               variant=variant)
        sample_c, sample_h, sample_w = 1, 28, 28
    elif dataset_key.startswith('olshausen'):
        # Olshausen natural scenes
        dataset = OlshausenDataset(olshausen_path,
                                   patch_size=12,
                                   step_size=olshausen_step_size,
                                   normalize=False)
        sample_c, sample_h, sample_w = 1, 12, 12
    elif dataset_key.startswith('cub'):
        # CUB birds
        dataset = CUB2011Dataset(cub_folder,
                                 train=train_ver,
                                 normalize=False)
        sample_c = 3
        sample_h = CUB2011Dataset.RESIZE_H
        sample_w = CUB2011Dataset.RESIZE_W
    elif dataset_key == 'cifar10':
        # CIFAR-10
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize),
        ])
        dataset = CIFAR10Dataset('./data',
                                 train=train_ver,
                                 transform=img_transform,
                                 download=True)
        sample_c, sample_h, sample_w = 3, 32, 32
    elif dataset_key.startswith('interp'):
        # Toy grayscale interpolation dataset
        dataset = InterpolationDataset('./data',
                                       normalize=True)
        sample_c = 1
        sample_h, sample_w = dataset.sample_h, dataset.sample_w
    else:
        raise ValueError('unrecognized dataset: %s' % dataset_key)
    data_minval = dataset.get_minval()
    data_maxval = dataset.get_maxval()
    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    return data_loader, sample_c, sample_h, sample_w, data_minval, data_maxval
