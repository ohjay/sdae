import torch
import argparse
from utils import init_model
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets import OlshausenDataset, MNISTVariant


def plot_samples(samples, fig_save_path=None):
    """Given a tensor of samples
    [of shape (num_originals, num_variations+1, sh, sw)]
    corresponding to Figure 15 from the 2010 SDAE paper,
    plots the samples in a grid of variations as per Figure 15."""
    fig, ax = plt.subplots(
        nrows=samples.shape[0], ncols=samples.shape[1])
    for i, row in enumerate(ax):
        for j, col in enumerate(row):
            col.imshow(samples[i, j, :, :], cmap='gray')
            col.axis('off')
    if fig_save_path is not None:
        fig.savefig(fig_save_path)
        print('[o] saved figure to %s' % fig_save_path)
    plt.show()


def generate_samples(model_class,
                     dataset,
                     restore_path,
                     olshausen_path=None,
                     olshausen_step_size=1,
                     num_originals=1,
                     num_variations=5,
                     fig_save_path=None):

    model = init_model(model_class, restore_path, restore_required=True)
    model.num_trained_blocks = model.num_blocks
    model.eval()

    # load data
    sample_h, sample_w = None, None
    if dataset.lower().startswith('mnist') or dataset.lower() in MNISTVariant.variant_options:
        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min())),
        ])
        variant = None if dataset.lower() == 'mnist' else dataset
        dataset = MNISTVariant(
            './data', train=True, transform=img_transform, download=True, variant=variant)
        sample_h, sample_w = 28, 28
    elif dataset.lower() == 'olshausen':
        dataset = OlshausenDataset(
            olshausen_path, patch_size=12, step_size=olshausen_step_size, normalize=False)
        sample_h, sample_w = 12, 12
    else:
        print('unrecognized dataset: %r' % (dataset,))
        print('error incoming...')
    data_loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

    # generate samples
    with torch.no_grad():
        samples = []
        for original_idx in range(num_originals):
            img, _ = next(iter(data_loader))
            img = img.view(img.size(0), -1).cuda()
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
        samples = samples.view(num_originals, num_variations + 1, sample_h, sample_w)
        plot_samples(samples.cpu().numpy(), fig_save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', type=str, default='MNISTSAE2')
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--olshausen_path', type=str, default=None)
    parser.add_argument('--olshausen_step_size', type=int, default=1)
    parser.add_argument('--num_originals', type=int, default=1)
    parser.add_argument('--num_variations', type=int, default=5)
    parser.add_argument('--fig_save_path', type=str, default=None)

    args = parser.parse_args()
    print(args)
    print('----------')

    generate_samples(args.model_class,
                     args.dataset,
                     args.restore_path,
                     args.olshausen_path,
                     args.olshausen_step_size,
                     args.num_originals,
                     args.num_variations,
                     args.fig_save_path)
