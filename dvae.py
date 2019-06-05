"""
Denoising variational autoencoder.
Code originally based on https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder.
"""

import os
import torch
import argparse
from utils import to_img, zero_mask, add_gaussian, salt_and_pepper, \
    save_image_wrapper, init_model, init_loss, init_data_loader


def train_dvae(batch_size, learning_rate, num_epochs, model_class,
               dataset_key, noise_type, zero_frac, gaussian_stdev, sp_frac,
               restore_path, save_path, log_freq, olshausen_path,
               olshausen_step_size, weight_decay, reconstruction_loss_type):
    # set up log folders
    if not os.path.exists('./01_original'):
        os.makedirs('./01_original')
    if not os.path.exists('./02_noisy'):
        os.makedirs('./02_noisy')
    if not os.path.exists('./03_output'):
        os.makedirs('./03_output')
    if not os.path.exists('./04_filters'):
        os.makedirs('./04_filters')

    # set up model and criterion
    model = init_model(model_class, restore_path, restore_required=False)
    criterion = init_loss('vae', reconstruction_loss_type=reconstruction_loss_type)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # load data
    data_loader, _, _, data_minval, data_maxval = init_data_loader(
        dataset_key, True, batch_size, olshausen_path, olshausen_step_size)

    # training loop
    warning_displayed = False
    original, noisy, output = None, None, None
    for epoch in range(num_epochs):
        mean_loss, total_num_examples = 0, 0
        for batch_idx, data in enumerate(data_loader):
            original, _ = data
            original = original.float()
            original = original.view(original.size(0), -1)
            original = original.cuda()

            # apply noise
            if noise_type == 'mn':
                noisy, _ = zero_mask(original, zero_frac)
            elif noise_type == 'gs':
                noisy, _ = add_gaussian(original, gaussian_stdev)
            elif noise_type == 'sp':
                noisy, _ = salt_and_pepper(original, sp_frac, data_minval, data_maxval)
            else:
                if not warning_displayed:
                    print('unrecognized noise type: %r' % (noise_type,))
                    print('using clean image as input')
                    warning_displayed = True
                noisy = original
            noisy = noisy.cuda()

            # =============== forward ===============
            output, mean, log_var = model(noisy)
            loss = criterion(output, original, mean, log_var)
            batch_size_ = original.size(0)  # might be undersized last batch
            total_num_examples += batch_size_
            # assumes `loss` is sum for batch
            mean_loss += (loss - mean_loss * batch_size_) / total_num_examples

            # =============== backward ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # =================== log ===================
        print('epoch {}/{}, loss={:.6f}'.format(epoch + 1, num_epochs, mean_loss.item()))
        if epoch % log_freq == 0 or epoch == num_epochs - 1:
            to_save = [
                (to_img(original.data.cpu()), './01_original', 'original'),
                (to_img(noisy.data.cpu()), './02_noisy', 'noisy'),
                (to_img(output.data.cpu()), './03_output', 'output'),
            ]
            for img, folder, desc in to_save:
                save_image_wrapper(img, os.path.join(folder, '{}_{}.png'.format(desc, epoch + 1)))

            torch.save(model.state_dict(), save_path)
            print('[o] saved model to %s' % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--model_class', type=str, default='MNISTVAE')
    parser.add_argument('--dataset_key', type=str, default='mnist')
    parser.add_argument('--noise_type', type=str, default='gs')
    parser.add_argument('--zero_frac', type=float, default=0.3)
    parser.add_argument('--gaussian_stdev', type=float, default=0.4)
    parser.add_argument('--sp_frac', type=float, default=0.1)
    parser.add_argument('--restore_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='./stage1_vae.pth')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--olshausen_path', type=str, default=None)
    parser.add_argument('--olshausen_step_size', type=int, default=1)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--reconstruction_loss_type', type=str, default='mse')

    args = parser.parse_args()
    print(args)
    print('----------')

    train_dvae(
        args.batch_size, args.learning_rate, args.num_epochs, args.model_class, args.dataset_key, args.noise_type,
        args.zero_frac, args.gaussian_stdev, args.sp_frac, args.restore_path, args.save_path, args.log_freq,
        args.olshausen_path, args.olshausen_step_size, args.weight_decay, args.reconstruction_loss_type)
