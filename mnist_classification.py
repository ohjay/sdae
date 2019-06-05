import torch
import argparse
import torch.nn as nn
from utils import init_model, init_data_loader


def init_sae_classifier(sae_model_class,
                        sae_restore_path,
                        classifier_model_class,
                        classifier_restore_path):

    # stacked autoencoder
    sae = init_model(sae_model_class, sae_restore_path, False)
    sae.num_trained_blocks = sae.num_blocks

    # obtain output dimensionality of final encoder
    enc_out_features = None
    for module in sae.encoders[-1]:
        if hasattr(module, 'out_features'):
            enc_out_features = module.out_features

    # classifier
    classifier = init_model(classifier_model_class,
                            classifier_restore_path,
                            restore_required=False,
                            enc_out_features=enc_out_features)

    return sae, classifier


def init_loss(loss_type, **loss_kwargs):
    Loss = {
        'mse': nn.MSELoss,
        'binary_cross_entropy': nn.BCELoss,
        'nll': nn.NLLLoss,
    }[loss_type.lower()]
    print('using %r as the loss' % (Loss,))
    return Loss(**loss_kwargs)


def mnist_train(batch_size=128,
                learning_rate=1e-2,
                num_epochs=100,
                sae_model_class='MNISTSAE2',
                sae_restore_path='./stage1_sae.pth',
                sae_save_path='./stage2_sae.pth',
                classifier_model_class='MNISTDenseClassifier2',
                classifier_restore_path=None,
                classifier_save_path='./stage2_classifier.pth',
                log_freq=10,
                weight_decay=0,
                loss_type='nll',
                no_sae=False,
                mnist_variant='mnist'):

    if no_sae:
        sae = None
        classifier = init_model(
            classifier_model_class, classifier_restore_path, False, enc_out_features=28*28)
        parameters = classifier.parameters()
    else:
        sae, classifier = init_sae_classifier(
            sae_model_class, sae_restore_path, classifier_model_class, classifier_restore_path)
        parameters = list(sae.parameters()) + list(classifier.parameters())
        sae.train()
    classifier.train()

    # loss and optimization
    criterion = init_loss(loss_type)
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    # load data
    data_loader, _, _, _, _ = init_data_loader(mnist_variant, True, batch_size)

    # training loop
    for epoch in range(num_epochs):
        mean_loss = 0
        for batch_idx, (img, label) in enumerate(data_loader):
            img = img.view(img.size(0), -1)
            img, label = img.cuda(), label.cuda()

            # =============== forward ===============
            if sae is not None:
                z = sae.encode(img)
                output = classifier(z)
            else:
                output = classifier(img)
            loss = criterion(output, label)
            mean_loss += (loss - mean_loss) / (batch_idx + 1)

            # =============== backward ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # =================== log ===================
        print('epoch {}/{}, loss={:.6f}'.format(epoch + 1, num_epochs, mean_loss.item()))
        if epoch % log_freq == 0 or epoch == num_epochs - 1:
            if sae is not None:
                torch.save(sae.state_dict(), sae_save_path)
                print('[o] saved SAE to %s' % sae_save_path)
            torch.save(classifier.state_dict(), classifier_save_path)
            print('[o] saved classifier to %s' % classifier_save_path)


def mnist_eval(batch_size=128,
               sae_model_class='MNISTSAE2',
               sae_restore_path='./stage2_sae.pth',
               classifier_model_class='MNISTDenseClassifier2',
               classifier_restore_path='./stage2_classifier.pth',
               loss_type='nll',
               no_sae=False,
               mnist_variant='mnist'):

    if no_sae:
        sae = None
        classifier = init_model(
            classifier_model_class, classifier_restore_path, False, enc_out_features=28*28)
    else:
        sae, classifier = init_sae_classifier(
            sae_model_class, sae_restore_path, classifier_model_class, classifier_restore_path)
        sae.eval()
    classifier.eval()

    # loss
    criterion = init_loss(loss_type, reduction='sum')

    # load data
    data_loader, _, _, _, _ = init_data_loader(mnist_variant, False, batch_size)

    total_loss, num_correct = 0, 0
    with torch.no_grad():
        for img, label in data_loader:
            img = img.view(img.size(0), -1)
            img, label = img.cuda(), label.cuda()

            # =============== forward ===============
            if sae is not None:
                z = sae.encode(img)
                output = classifier(z)
            else:
                output = classifier(img)
            total_loss += criterion(output, label).item()
            prediction = output.argmax(dim=1, keepdim=True)
            num_correct += prediction.eq(label.view_as(prediction)).sum().item()

    dataset_size = len(data_loader.dataset)
    mean_loss = total_loss / dataset_size
    print('\n[test] mean loss: {:.6f}, acc {:.6f}'.format(mean_loss, num_correct / dataset_size))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--sae_model_class', type=str, default='MNISTSAE2')
    parser.add_argument('--sae_restore_path', type=str, default='./stage1_sae.pth')
    parser.add_argument('--sae_save_path', type=str, default='./stage2_sae.pth')
    parser.add_argument('--classifier_model_class', type=str, default='MNISTDenseClassifier2')
    parser.add_argument('--classifier_restore_path', type=str, default=None)
    parser.add_argument('--classifier_save_path', type=str, default='./stage2_classifier.pth')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--loss_type', type=str, default='nll')
    parser.add_argument('--no_train', action='store_true')
    parser.add_argument('--no_sae', action='store_true')
    parser.add_argument('--mnist_variant', type=str, default='mnist')

    args = parser.parse_args()
    print(args)
    print('----------')

    if not args.no_train:
        print('\n'
              '=========\n'
              '= TRAIN =\n'
              '=========\n')
        mnist_train(args.batch_size,
                    args.learning_rate,
                    args.num_epochs,
                    args.sae_model_class,
                    args.sae_restore_path,
                    args.sae_save_path,
                    args.classifier_model_class,
                    args.classifier_restore_path,
                    args.classifier_save_path,
                    args.log_freq,
                    args.weight_decay,
                    args.loss_type,
                    args.no_sae,
                    args.mnist_variant)
        args.sae_restore_path = args.sae_save_path
        args.classifier_restore_path = args.classifier_save_path

    print('\n'
          '========\n'
          '= EVAL =\n'
          '========\n')
    mnist_eval(args.batch_size,
               args.sae_model_class,
               args.sae_restore_path,
               args.classifier_model_class,
               args.classifier_restore_path,
               args.loss_type,
               args.no_sae,
               args.mnist_variant)
