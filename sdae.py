"""
Stacked denoising autoencoder.
Code originally based on https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder.
"""

import os
import torch
import operator
import argparse
import numpy as np
from models import *
import torch.nn as nn
from functools import reduce
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from datasets import OlshausenDataset, MNISTVariant


def to_img(x):
    h = w = int(np.sqrt(
        reduce(operator.mul, list(x.size())[1:], 1)))
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


def plot_first_layer_weights(model, weight_h=None, weight_w=None):
    weights = model.get_first_layer_weights()
    print('shape of first-layer weights: %r' % (weights.shape,))
    fig, ax = plt.subplots(nrows=5, ncols=10)
    i = 0
    for row in ax:
        for col in row:
            weight = weights[i, :]
            if not weight_h or not weight_w:
                # Infer height and width of weight, assuming it is square
                weight_h = weight_w = int(np.sqrt(weight.size))
            col.imshow(np.reshape(weights[i, :], (weight_h, weight_w)), cmap='gray')
            col.axis('off')
            i += 1
    plt.show()


def save_image_wrapper(img, filepath):
    save_image(img, filepath)
    print('[o] saved image to %s' % filepath)


def train_sdae(batch_size=128, learning_rate=1e-2, num_epochs=100, model_key='olshausen_ae',
               dataset='olshausen', noise_type='gs', zero_frac=0.3, gaussian_stdev=0.4, sp_frac=0.1,
               restore_path=None, save_path='./stage1_sae.pth', log_freq=10, olshausen_path=None,
               olshausen_step_size=1, weight_decay=0, loss_type='mse', emph_wt_a=1, emph_wt_b=1):
    # set up log folders
    if not os.path.exists('./01_original'):
        os.makedirs('./01_original')
    if not os.path.exists('./02_noisy'):
        os.makedirs('./02_noisy')
    if not os.path.exists('./03_output'):
        os.makedirs('./03_output')
    if not os.path.exists('./04_filters'):
        os.makedirs('./04_filters')

    # set up model and optimizer
    Model = {
        'mnist_ae': MNISTAE,
        'mnist_sae2': MNISTSAE2,
        'olshausen_ae': OlshausenAE,
    }[model_key.lower()]
    print('using %r as the model' % (Model,))
    model = Model().cuda()
    if restore_path:
        if os.path.exists(restore_path):
            model.load_state_dict(torch.load(restore_path))
            print('restored model from %s' % restore_path)
        else:
            print('warning: checkpoint %s not found, skipping...' % restore_path)
    Loss = {
        'mse': nn.MSELoss,
        'binary_cross_entropy': nn.BCELoss,
    }[loss_type.lower()]
    print('using %r as the loss' % (Loss,))
    criterion = Loss()

    # load data
    data_minval, data_maxval = 0.0, 1.0
    if dataset.lower().startswith('mnist'):
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(normalize),
        ])
        dataset = MNISTVariant(
            './data', train=True, transform=img_transform, download=True, variant=dataset)
    elif dataset.lower() == 'olshausen':
        dataset = OlshausenDataset(
            olshausen_path, patch_size=12, step_size=olshausen_step_size, normalize=False)
        data_minval = dataset.get_minval()
        data_maxval = dataset.get_maxval()
    else:
        print('unrecognized dataset: %r' % (dataset,))
        print('error incoming...')
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    # training loop
    affected = None
    warning_displayed = False
    original, noisy, output = None, None, None
    for ae_idx in range(model.num_blocks):

        # train one block at a time
        print('--------------------')
        print('training block %d/%d' % (ae_idx + 1, model.num_blocks))
        print('--------------------')
        optimizer = torch.optim.Adam(
            model.get_block_parameters(ae_idx), lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            mean_loss = 0
            for batch_idx, data in enumerate(data_loader):
                original, _ = data
                original = original.float()
                original = original.view(original.size(0), -1)
                original = original.cuda()
                original = model.encode(original)
                original = original.detach()

                # apply noise
                if noise_type == 'mn':
                    noisy, affected = zero_mask(original, zero_frac)
                elif noise_type == 'gs':
                    noisy, affected = add_gaussian(original, gaussian_stdev)
                elif noise_type == 'sp':
                    noisy, affected = salt_and_pepper(original, sp_frac, data_minval, data_maxval)
                else:
                    if not warning_displayed:
                        print('unrecognized noise type: %r' % (noise_type,))
                        print('using clean image as input')
                        warning_displayed = True
                    noisy = original
                noisy = noisy.detach().cuda()

                # =============== forward ===============
                output = model(noisy, ae_idx)
                if (emph_wt_a != 1 or emph_wt_b != 1) and noise_type != 'gs':
                    # emphasize corrupted dimensions in the loss
                    loss = emph_wt_a * criterion(output[affected], original[affected]) + \
                           emph_wt_b * criterion(output[1 - affected], original[1 - affected])
                else:
                    loss = criterion(output, original)
                mean_loss += (loss - mean_loss) / (batch_idx + 1)

                # =============== backward ==============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # =================== log ===================
            print('epoch {}/{}, loss={:.6f}'.format(epoch + 1, num_epochs, mean_loss.item()))
            if epoch % log_freq == 0 or epoch == num_epochs - 1:
                if ae_idx == 0:
                    to_save = [
                        (to_img(original.data.cpu()), './01_original', 'original'),
                        (to_img(noisy.data.cpu()), './02_noisy', 'noisy'),
                        (to_img(output.data.cpu()), './03_output', 'output'),
                        (to_img(model.get_first_layer_weights(as_tensor=True)), './04_filters', 'filters'),
                    ]
                    for img, folder, desc in to_save:
                        save_image_wrapper(img, os.path.join(folder, '{}_{}.png'.format(desc, epoch + 1)))

                torch.save(model.state_dict(), save_path)
                print('[o] saved model to %s' % save_path)
        model.num_trained_blocks += 1

    plot_first_layer_weights(model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_key', type=str, default='olshausen_ae')
    parser.add_argument('--dataset', type=str, default='olshausen')
    parser.add_argument('--noise_type', type=str, default='gs')
    parser.add_argument('--zero_frac', type=float, default=0.3)
    parser.add_argument('--gaussian_stdev', type=float, default=0.4)
    parser.add_argument('--sp_frac', type=float, default=0.1)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./stage1_sae.pth')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--olshausen_path', type=str, default=None)
    parser.add_argument('--olshausen_step_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--emph_wt_a', type=float, default=1)
    parser.add_argument('--emph_wt_b', type=float, default=1)

    args = parser.parse_args()
    print(args)
    print('----------')

    train_sdae(
        args.batch_size, args.learning_rate, args.num_epochs, args.model_key, args.dataset, args.noise_type,
        args.zero_frac, args.gaussian_stdev, args.sp_frac, args.restore_path, args.save_path, args.log_freq,
        args.olshausen_path, args.olshausen_step_size, args.weight_decay, args.loss_type, args.emph_wt_a, args.emph_wt_b)
