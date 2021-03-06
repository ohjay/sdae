import torch
import modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import init_model, init_data_loader


def plot_samples(samples, fig_save_path=None):
    """Given a tensor of samples
    [of shape (num_originals, num_variations+1, sh, sw)]
    corresponding to Figure 15 from the 2010 SDAE paper,
    plots the samples in a grid of variations as per Figure 15."""
    nrows, ncols = samples.shape[:2]
    fig, ax = plt.subplots(nrows, ncols)
    if nrows == 1 and ncols == 1:
        ax = [[ax]]
    elif nrows == 1:
        ax = [[col for col in ax]]
    elif ncols == 1:
        ax = [[row] for row in ax]
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.imshow(samples[i, j, :, :], cmap='gray')
            col.axis('off')
    if fig_save_path is not None:
        fig.savefig(fig_save_path)
        print('[o] saved figure to %s' % fig_save_path)
    plt.show()


def generate_samples_ae(dataset_key,
                        olshausen_path=None,
                        olshausen_step_size=1,
                        cub_folder=None,
                        num_originals=1,
                        num_variations=5,
                        fig_save_path=None):

    # load data
    batch_size = 1
    data_loader, _, sample_h, sample_w, _, _ = init_data_loader(
        dataset_key, True, batch_size, olshausen_path, olshausen_step_size, cub_folder)
    img_shape = [sample_h, sample_w]

    # generate samples
    with torch.no_grad():
        samples = []
        for original_idx in range(num_originals):
            img, _ = next(iter(data_loader))
            img = img.float()
            if torch.cuda.is_available():
                img = img.cuda()
            img_shape = [d for d in img.size() if d != 1]
            if not model.is_convolutional:
                img = img.view(img.size(0), -1)
            z = model.encode(img)  # top-layer representation
            sample_variations = [img]  # leftmost entry is the original image
            for variation_idx in range(num_variations):
                sample = z
                for k in range(model.num_blocks):
                    # sample
                    sample = torch.bernoulli(sample)
                    # decode
                    sample = model.decode(sample, model.num_blocks - k - 1)
                sample_variations.append(sample)
            samples.append(torch.cat(sample_variations, dim=0))
        samples = torch.stack(samples, dim=0)  # shape: (num_originals, num_variations+1, sh*sw)
        samples = samples.view(num_originals, num_variations + 1, *img_shape)
        samples = samples.cpu().numpy()
        if len(img_shape) == 3:
            # reshape to (h, w, c)
            samples = np.transpose(samples, (0, 1, 3, 4, 2))
        plot_samples(samples, fig_save_path)


def generate_samples_vae(num,
                         sample_h,
                         sample_w,
                         fig_save_path=None,
                         lower=-3,
                         upper=3):

    assert not model.is_convolutional

    # generate samples
    with torch.no_grad():
        # vary first two dims over grid
        dim0_vals = np.linspace(lower, upper, num)
        dim1_vals = np.linspace(lower, upper, num)
        dim0_vals, dim1_vals = np.meshgrid(dim0_vals, dim1_vals)
        latent_vecs = np.stack((dim0_vals, dim1_vals), axis=-1).reshape(-1, 2)

        latent_dimensionality = model.get_enc_out_features(-1)
        if latent_dimensionality > 2:
            # keep values for other dimensions fixed
            other_vals = np.random.randn(latent_dimensionality - 2)
            other_vals = np.expand_dims(other_vals, axis=0)  # shape: (1, ld-2)
            other_vals = np.tile(other_vals, (num * num, 1))  # shape: (n*n, ld-2)
            latent_vecs = np.concatenate((latent_vecs, other_vals), axis=1)  # shape: (n*n, ld)

        latent_vecs = torch.from_numpy(latent_vecs).float()
        if torch.cuda.is_available():
            latent_vecs = latent_vecs.cuda()
        samples = model.decode(latent_vecs)  # shape: (n*n, sample_h*sample_w)
        samples = samples.view(num, num, sample_h, sample_w)
        plot_samples(samples.cpu().numpy(), fig_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', type=str, default='MNISTSAE2')
    parser.add_argument('--dataset_key', type=str, default='mnist')
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--olshausen_path', type=str, default=None)
    parser.add_argument('--olshausen_step_size', type=int, default=1)
    parser.add_argument('--cub_folder', type=str, default=None)
    parser.add_argument('--num_originals', type=int, default=1)
    parser.add_argument('--num_variations', type=int, default=5)
    parser.add_argument('--fig_save_path', type=str, default=None)
    parser.add_argument('--num', type=int, default=10)
    parser.add_argument('--sample_h', type=int, default=28)
    parser.add_argument('--sample_w', type=int, default=28)
    parser.add_argument('--lower', type=float, default=-3)
    parser.add_argument('--upper', type=float, default=3)

    args = parser.parse_args()
    print(args)
    print('----------')

    model = init_model(args.model_class, args.restore_path, restore_required=True)
    model.num_trained_blocks = model.num_blocks
    model.eval()

    if isinstance(model, modules.SVAE) or \
            args.dataset_key.lower().startswith('interp'):
        generate_samples_vae(args.num,
                             args.sample_h,
                             args.sample_w,
                             args.fig_save_path,
                             args.lower,
                             args.upper)
    else:
        generate_samples_ae(args.dataset_key,
                            args.olshausen_path,
                            args.olshausen_step_size,
                            args.cub_folder,
                            args.num_originals,
                            args.num_variations,
                            args.fig_save_path)
