import os
import torch
import argparse
from models import *
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def normalize(x):
    return (x - x.min()) / (x.max() - x.min())


def init_sae_classifier(sae_model_key,
                        sae_restore_path,
                        classifier_model_key,
                        classifier_restore_path):

    # stacked autoencoder
    # SAE_ is the SAE subclass whose encoders should be utilized
    SAE_ = {
        'mnist_ae': MNISTAE,
        'mnist_sae2': MNISTSAE2,
    }[sae_model_key.lower()]
    print('using %r as the SAE model' % (SAE_,))
    sae = SAE_().cuda()
    sae.load_state_dict(torch.load(sae_restore_path))
    print('restored SAE from %s' % sae_restore_path)
    sae.num_trained_blocks = sae.num_blocks

    # obtain output dimensionality of final encoder
    enc_out_features = None
    for module in sae.encoders[-1]:
        if hasattr(module, 'out_features'):
            enc_out_features = module.out_features

    # classifier
    Classifier = {
        'mnist_dense_classifier2': MNISTDenseClassifier2,
    }[classifier_model_key.lower()]
    print('using %r as the classifier model' % (Classifier,))
    classifier = Classifier(enc_out_features).cuda()
    if classifier_restore_path:
        classifier.load_state_dict(torch.load(classifier_restore_path))
        print('restored classifier from %s' % classifier_restore_path)

    return sae, classifier


def init_loss(loss_type, **loss_kwargs):
    Loss = {
        'mse': nn.MSELoss,
        'binary_cross_entropy': nn.BCELoss,
        'nll': nn.NLLLoss,
    }[loss_type.lower()]
    print('using %r as the loss' % (Loss,))
    return Loss(**loss_kwargs)


def init_data_loader(train, batch_size):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(normalize),
    ])
    dataset = MNIST(root='./data', train=train, transform=img_transform, download=True)
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)


def mnist_train(batch_size=128,
                learning_rate=1e-2,
                num_epochs=100,
                sae_model_key='mnist_sae2',
                sae_restore_path='./stage1_sae.pth',
                sae_save_path='./stage2_sae.pth',
                classifier_model_key='mnist_dense_classifier2',
                classifier_restore_path=None,
                classifier_save_path='./stage2_classifier.pth',
                log_freq=10,
                weight_decay=0,
                loss_type='nll'):

    sae, classifier = init_sae_classifier(
        sae_model_key, sae_restore_path, classifier_model_key, classifier_restore_path)

    # loss and optimization
    criterion = init_loss(loss_type)
    parameters = list(sae.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)

    # load data
    data_loader = init_data_loader(True, batch_size)

    # training loop
    sae.train()
    classifier.train()
    for epoch in range(num_epochs):
        mean_loss = 0
        for batch_idx, (img, label) in enumerate(data_loader):
            img = img.view(img.size(0), -1)
            img, label = img.cuda(), label.cuda()

            # =============== forward ===============
            z = sae.encode(img)
            output = classifier(z)
            loss = criterion(output, label)
            mean_loss += (loss - mean_loss) / (batch_idx + 1)

            # =============== backward ==============
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # =================== log ===================
        print('epoch {}/{}, loss={:.6f}'.format(epoch + 1, num_epochs, mean_loss.item()))
        if epoch % log_freq == 0 or epoch == num_epochs - 1:
            torch.save(sae.state_dict(), sae_save_path)
            print('[o] saved SAE to %s' % sae_save_path)
            torch.save(classifier.state_dict(), classifier_save_path)
            print('[o] saved classifier to %s' % classifier_save_path)


def mnist_eval(batch_size=128,
               sae_model_key='mnist_sae2',
               sae_restore_path='./stage2_sae.pth',
               classifier_model_key='mnist_dense_classifier2',
               classifier_restore_path='./stage2_classifier.pth',
               loss_type='nll'):

    sae, classifier = init_sae_classifier(
        sae_model_key, sae_restore_path, classifier_model_key, classifier_restore_path)

    criterion = init_loss(loss_type, reduction='sum')

    # load data
    data_loader = init_data_loader(False, batch_size)

    sae.eval()
    classifier.eval()
    total_loss, num_correct = 0, 0
    with torch.no_grad():
        for img, label in data_loader:
            img = img.view(img.size(0), -1)
            img, label = img.cuda(), label.cuda()

            # =============== forward ===============
            z = sae.encode(img)
            output = classifier(z)  # log-probabilities
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
    parser.add_argument('--sae_model_key', type=str, default='mnist_sae2')
    parser.add_argument('--sae_restore_path', type=str, default='./stage1_sae.pth')
    parser.add_argument('--sae_save_path', type=str, default='./stage2_sae.pth')
    parser.add_argument('--classifier_model_key', type=str, default='mnist_dense_classifier2')
    parser.add_argument('--classifier_restore_path', type=str, default=None)
    parser.add_argument('--classifier_save_path', type=str, default='./stage2_classifier.pth')
    parser.add_argument('--log_freq', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--loss_type', type=str, default='nll')
    parser.add_argument('--no_train', action='store_true')

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
                    args.sae_model_key,
                    args.sae_restore_path,
                    args.sae_save_path,
                    args.classifier_model_key,
                    args.classifier_restore_path,
                    args.classifier_save_path,
                    args.log_freq,
                    args.weight_decay,
                    args.loss_type)
        args.sae_restore_path = args.sae_save_path
        args.classifier_restore_path = args.classifier_save_path

    print('\n'
          '========\n'
          '= EVAL =\n'
          '========\n')
    mnist_eval(args.batch_size,
               args.sae_model_key,
               args.sae_restore_path,
               args.classifier_model_key,
               args.classifier_restore_path,
               args.loss_type)
