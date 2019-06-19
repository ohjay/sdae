import torch
import modules
import argparse
from utils import init_model, init_loss, init_data_loader


def forward(img):
    if sae is not None:
        z = sae.encode(img)
        if isinstance(sae, modules.SVAE):
            z = z[1]  # z consists of a sampled latent vector, a mean, and a log_var
        if sae.is_convolutional and not classifier.is_convolutional:
            z = z.view(z.size(0), -1)
        return classifier(z)
    else:
        return classifier(img)


def do_train(data_loader, criterion):
    """Trains the model using one pass through the training set."""
    if sae is not None:
        sae.train()
    classifier.train()

    mean_loss = 0
    for batch_idx, (img, label) in enumerate(data_loader):
        if sae is not None and not sae.is_convolutional:
            img = img.view(img.size(0), -1)
        img = img.float()
        if torch.cuda.is_available():
            img, label = img.cuda(), label.cuda()

        # =============== forward ===============
        output = forward(img)
        loss = criterion(output, label)
        mean_loss += (loss - mean_loss) / (batch_idx + 1)

        # =============== backward ==============
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # =================== log ===================
    print('[train] epoch {}/{}, loss={:.6f}'.format(epoch + 1, args.num_epochs, mean_loss.item()))


def do_eval(data_loader, criterion):
    """Evaluates the model on the entire validation/test set."""
    if sae is not None:
        sae.eval()
    classifier.eval()

    total_loss, num_correct = 0, 0
    with torch.no_grad():
        for img, label in data_loader:
            if sae is not None and not sae.is_convolutional:
                img = img.view(img.size(0), -1)
            img = img.float()
            if torch.cuda.is_available():
                img, label = img.cuda(), label.cuda()

            # =============== forward ===============
            output = forward(img)
            total_loss += criterion(output, label).item()
            prediction = output.argmax(dim=1, keepdim=True)
            num_correct += prediction.eq(label.view_as(prediction)).sum().item()

    dataset_size = len(data_loader.dataset)
    mean_loss = total_loss / dataset_size
    print('[eval] mean loss: {:.6f}, acc {:.6f}'.format(mean_loss, num_correct / dataset_size))


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
    parser.add_argument('--dataset_key', type=str, default='mnist')
    parser.add_argument('--cub_folder', type=str, default=None)

    args = parser.parse_args()
    print(args)
    print('----------')

    # SAE
    if args.no_sae:
        sae = None
        sae_parameters = []
        enc_out_features = 28 * 28
    else:
        # stacked autoencoder
        sae = init_model(args.sae_model_class, args.sae_restore_path, False)
        sae.num_trained_blocks = sae.num_blocks
        sae_parameters = list(sae.parameters())
        # obtain output dimensionality of final encoder
        enc_out_features = sae.get_enc_out_features(-1)

    # classifier
    classifier = init_model(args.classifier_model_class,
                            args.classifier_restore_path,
                            restore_required=False,
                            enc_out_features=enc_out_features)
    parameters = sae_parameters + list(classifier.parameters())

    # loss and optimization
    criterion_train = init_loss(args.loss_type)
    criterion_eval = init_loss(args.loss_type, reduction='sum')
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    # load data
    data_loader_train, _, _, _, _, _ = init_data_loader(args.dataset_key,
                                                        train_ver=True,
                                                        batch_size=args.batch_size,
                                                        cub_folder=args.cub_folder)
    data_loader_eval, _, _, _, _, _ = init_data_loader(args.dataset_key,
                                                       train_ver=False,
                                                       batch_size=args.batch_size,
                                                       cub_folder=args.cub_folder)

    if args.no_train:
        do_eval(data_loader_eval, criterion_eval)
    else:
        # training loop
        for epoch in range(args.num_epochs):
            do_train(data_loader_train, criterion_train)
            if epoch % args.log_freq == 0 or epoch == args.num_epochs - 1:
                if sae is not None:
                    torch.save(sae.state_dict(), args.sae_save_path)
                    print('[o] saved SAE to %s' % args.sae_save_path)
                torch.save(classifier.state_dict(), args.classifier_save_path)
                print('[o] saved classifier to %s' % args.classifier_save_path)
                do_eval(data_loader_eval, criterion_eval)
