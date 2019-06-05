"""
Stacked denoising autoencoder.
Code originally based on https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder.
"""

import os
import torch
import argparse
from utils import to_img, zero_mask, add_gaussian, salt_and_pepper, \
    plot_first_layer_weights, save_image_wrapper, init_model, init_loss, init_data_loader


def train_sdae(batch_size=128, learning_rate=1e-2, num_epochs=100, model_class='OlshausenAE',
               dataset_key='olshausen', noise_type='gs', zero_frac=0.3, gaussian_stdev=0.4, sp_frac=0.1,
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

    # set up model and criterion
    model = init_model(model_class, restore_path, restore_required=False)
    criterion = init_loss(loss_type)

    # load data
    data_loader, _, _, data_minval, data_maxval = init_data_loader(
        dataset_key, True, batch_size, olshausen_path, olshausen_step_size)

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
    parser.add_argument('--model_class', type=str, default='OlshausenAE')
    parser.add_argument('--dataset_key', type=str, default='olshausen')
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
        args.batch_size, args.learning_rate, args.num_epochs, args.model_class, args.dataset_key, args.noise_type,
        args.zero_frac, args.gaussian_stdev, args.sp_frac, args.restore_path, args.save_path, args.log_freq,
        args.olshausen_path, args.olshausen_step_size, args.weight_decay, args.loss_type, args.emph_wt_a, args.emph_wt_b)
