import torch
import modules
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import init_model, init_data_loader

COLORS = [
    'firebrick', 'forestgreen', 'teal', 'gold', 'lightslategray',
    'rebeccapurple', 'darkorange', 'deepskyblue', 'fuchsia', 'navy'
]


def plot_tsne(model_class, restore_path, dataset_key, batch_size, cub_folder, fig_save_path):
    # model
    model = init_model(model_class, restore_path, restore_required=True)
    model.num_trained_blocks = model.num_blocks
    model.eval()

    # data
    data_loader, _, _, _, _, _ = init_data_loader(
        dataset_key, True, batch_size, None, None, cub_folder)

    with torch.no_grad():
        img, label = next(iter(data_loader))
        if not model.is_convolutional:
            img = img.view(img.size(0), -1)
        img, label = img.float().cuda(), label.data.cpu().numpy()

        z = model.encode(img)
        if isinstance(model, modules.SVAE):
            z = z[1]  # use mean as `z`
        z = z.view(z.size(0), -1)
        z = z.data.cpu().numpy()  # (batch_size, feature_dim)
        z_embedded = TSNE(n_components=2).fit_transform(z)  # (n, 2)

        fig, ax = plt.subplots()
        for i in range(10):
            idxs = np.where(label == i)
            ax.scatter(z_embedded[idxs, 0], z_embedded[idxs, 1], c=COLORS[i], label=i)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.9, box.height * 0.9])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xticks([])
        ax.set_yticks([])
        if fig_save_path:
            plt.savefig(fig_save_path)
            print('[o] saved figure to %s' % fig_save_path)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_class', type=str, default='MNISTSAE2')
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--dataset_key', type=str, default='mnist')
    parser.add_argument('--batch_size', type=int, default=1000)
    parser.add_argument('--cub_folder', type=str, default=None)
    parser.add_argument('--fig_save_path', type=str, default=None)

    args = parser.parse_args()
    print(args)
    print('----------')

    plot_tsne(args.model_class, args.restore_path, args.dataset_key,
              args.batch_size, args.cub_folder, args.fig_save_path)
