"""
Stacked denoising autoencoder.
Code originally based on https://github.com/L1aoXingyu/pytorch-beginner/tree/master/08-AutoEncoder.
"""

import os
import torch
import modules
import argparse
import numpy as np
from utils import try_float, to_img, zero_mask, add_gaussian, salt_and_pepper, \
    plot_first_layer_weights, save_image_wrapper, init_model, init_loss, init_data_loader


def train_sdae(batch_size, learning_rate, num_epochs, model_class, dataset_key,
               noise_type, zero_frac, gaussian_stdev, sp_frac, restore_path,
               save_path, log_freq, olshausen_path, olshausen_step_size, weight_decay,
               loss_type, emph_wt_a, emph_wt_b, vae_reconstruction_loss_type,
               cub_folder, learned_noise_wt, nt_restore_prefix, nt_save_prefix):
    # set up log folders
    if not os.path.exists('./01_original'):
        os.makedirs('./01_original')
    if not os.path.exists('./02_noisy'):
        os.makedirs('./02_noisy')
    if not os.path.exists('./03_output'):
        os.makedirs('./03_output')
    if not os.path.exists('./04_filters'):
        os.makedirs('./04_filters')
    if not os.path.exists('./05_stdev'):
        os.makedirs('./05_stdev')

    # set up model and criterion
    model = init_model(model_class, restore_path, restore_required=False)
    if isinstance(model, modules.SVAE):
        criterion = init_loss('vae', reconstruction_loss_type=vae_reconstruction_loss_type)
    else:
        criterion = init_loss(loss_type)

    # load data
    data_loader, sample_c, sample_h, sample_w, data_minval, data_maxval = init_data_loader(
        dataset_key, True, batch_size, olshausen_path, olshausen_step_size, cub_folder)
    original_size = sample_c * sample_h * sample_w

    # training loop
    affected = None
    warning_displayed = False
    original, noisy, output = None, None, None
    for ae_idx in range(model.num_blocks):

        stdev = None
        noise_transformer = None
        noise_transformer_params = []
        if learned_noise_wt:
            noise_transformer = modules.NoiseTransformer(original_size).cuda()
            if nt_restore_prefix is not None:
                nt_restore_path = '%s_%d.pth' % (nt_restore_prefix, ae_idx)
                if os.path.exists(nt_restore_path):
                    noise_transformer.load_state_dict(torch.load(nt_restore_path))
                    print('restored noise transformer from %s' % nt_restore_path)
                else:
                    print('warning: checkpoint %s not found, skipping...' % nt_restore_path)
            noise_transformer_params = list(noise_transformer.parameters())

        # train one block at a time
        print('--------------------')
        print('training block %d/%d' % (ae_idx + 1, model.num_blocks))
        print('--------------------')
        params = model.get_block_parameters(ae_idx) + noise_transformer_params
        optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)

        for epoch in range(num_epochs):
            mean_loss, total_num_examples = 0, 0
            for batch_idx, data in enumerate(data_loader):
                original, _ = data
                original = original.float()
                if not model.is_convolutional:
                    original = original.view(original.size(0), -1)
                original = original.cuda()
                original = model.encode(original)
                if isinstance(model, modules.SVAE):
                    original = original[1]  # (sampled latent vector, mean, log_var)
                original = original.detach()

                # apply noise
                if learned_noise_wt:
                    stdev = noise_transformer.compute_stdev(original)
                    noisy = noise_transformer.apply_noise(original, stdev)
                else:
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
                if isinstance(model, modules.SVAE):
                    output, mean, log_var = model(noisy, ae_idx)
                    loss = criterion(output, original, mean, log_var)
                    batch_size_ = original.size(0)  # might be undersized last batch
                    total_num_examples += batch_size_
                    # assumes `loss` is sum for batch
                    mean_loss += (loss - mean_loss * batch_size_) / total_num_examples
                else:
                    output = model(noisy, ae_idx)
                    if (emph_wt_a != 1 or emph_wt_b != 1) and noise_type != 'gs':
                        # emphasize corrupted dimensions in the loss
                        loss = emph_wt_a * criterion(output[affected], original[affected]) + \
                               emph_wt_b * criterion(output[1 - affected], original[1 - affected])
                    else:
                        loss = criterion(output, original)
                    mean_loss += (loss - mean_loss) / (batch_idx + 1)  # assumes `loss` is mean for batch

                if learned_noise_wt:
                    # encourage large standard deviations
                    if type(learned_noise_wt) == str:
                        if learned_noise_wt.lower().startswith(('sqrt', 'mean')):
                            learned_noise_wt_ = np.sqrt(mean_loss.item())
                        else:
                            raise ValueError('invalid `learned_noise_wt` option %s' % learned_noise_wt)
                    else:
                        learned_noise_wt_ = learned_noise_wt
                    loss -= learned_noise_wt_ * torch.mean(stdev)

                # =============== backward ==============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # =================== log ===================
            print('epoch {}/{}, loss={:.6f}'.format(epoch + 1, num_epochs, mean_loss.item()))
            if epoch % log_freq == 0 or epoch == num_epochs - 1:
                # save images
                if ae_idx == 0:
                    to_save = [
                        (to_img(original.data.cpu()), './01_original', 'original'),
                        (to_img(noisy.data.cpu()), './02_noisy', 'noisy'),
                        (to_img(output.data.cpu()), './03_output', 'output'),
                        (to_img(model.get_first_layer_weights(as_tensor=True)), './04_filters', 'filters'),
                    ]
                    for img, folder, desc in to_save:
                        save_image_wrapper(img, os.path.join(folder, '{}_{}.png'.format(desc, epoch + 1)))

                # save learned stdev
                if learned_noise_wt:
                    stdev_path = os.path.join(
                        './05_stdev', 'stdev_{}_{}.txt'.format(ae_idx, epoch + 1))
                    np.savetxt(stdev_path, stdev.data.cpu().numpy(), fmt='%.18f')
                    print('[o] saved stdev to %s' % stdev_path)

                # save model(s)
                torch.save(model.state_dict(), save_path)
                print('[o] saved model to %s' % save_path)
                if learned_noise_wt and nt_save_prefix is not None:
                    nt_save_path = '%s_%d.pth' % (nt_save_prefix, ae_idx)
                    torch.save(noise_transformer.state_dict(), nt_save_path)
                    print('[o] saved lvl-%d noise transformer to %s' % (ae_idx, nt_save_path))
        model.num_trained_blocks += 1
        original_size = model.get_enc_out_features(ae_idx)

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
    parser.add_argument('--vae_reconstruction_loss_type', type=str, default='mse')
    parser.add_argument('--cub_folder', type=str, default=None)
    parser.add_argument('--learned_noise_wt', default=0)
    parser.add_argument('--nt_restore_prefix', type=str, default=None)
    parser.add_argument('--nt_save_prefix', type=str, default=None)

    args = parser.parse_args()
    args.learned_noise_wt = try_float(args.learned_noise_wt)
    print(args)
    print('----------')

    train_sdae(
        args.batch_size, args.learning_rate, args.num_epochs, args.model_class, args.dataset_key, args.noise_type,
        args.zero_frac, args.gaussian_stdev, args.sp_frac, args.restore_path, args.save_path, args.log_freq,
        args.olshausen_path, args.olshausen_step_size, args.weight_decay, args.loss_type, args.emph_wt_a, args.emph_wt_b,
        args.vae_reconstruction_loss_type, args.cub_folder, args.learned_noise_wt, args.nt_restore_prefix, args.nt_save_prefix)
