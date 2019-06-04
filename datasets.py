import os
import torch
import numpy as np
import scipy.io as sio
from torch.utils import data
from scipy.ndimage import rotate
from torchvision.datasets import MNIST
from skimage.util import view_as_windows


class OlshausenDataset(data.Dataset):
    """(Whitened) natural scene images.
    Available here: http://www.rctn.org/bruno/sparsenet.
    """

    def __init__(self, mat_path, patch_size=12, step_size=1, normalize=False):
        dataset = sio.loadmat(mat_path)
        images = np.ascontiguousarray(dataset['IMAGES'])  # shape: (512, 512, 10)
        self.patches = np.squeeze(view_as_windows(
            images, (patch_size, patch_size, 10), step=step_size))  # shape: (., ., PS, PS, 10)
        self.patches = self.patches.transpose((0, 1, 4, 2, 3))
        self.patches = self.patches.reshape((-1, patch_size, patch_size))
        if normalize:
            # normalize to range [0, 1]
            _min = self.patches.min()
            _max = self.patches.max()
            self.patches = (self.patches - _min) / (_max - _min)
        if self.patches.dtype != np.float:
            print('converting data type from %r to np.float' % self.patches.dtype)
            self.patches = self.patches.astype(np.float)
        print('image statistics:')
        print('min: %r, mean: %r, max: %r'
              % (self.patches.min(), self.patches.mean(), self.patches.max()))

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        return self.patches[index, :, :], index  # (image, "label")

    def get_minval(self):
        return self.patches.min()

    def get_maxval(self):
        return self.patches.max()


class MNISTVariant(MNIST):
    """Modified MNIST, or original MNIST if variant=None.
    Based on PyTorch's MNIST code at torchvision/datasets/mnist.py."""

    variant_options = (
        'rot',
        'bg_rand',
        'bg_rand_rot',
    )

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 download=False,
                 variant=None,
                 generate=False):

        super(MNIST, self).__init__(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        # check for existence of original MNIST dataset
        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        if variant:
            self.variant = variant.lower()
            assert self.variant in self.variant_options
            self.variant_folder = os.path.join(self.root, 'MNIST_%s' % self.variant, 'processed')

            data_path = os.path.join(self.variant_folder, data_file)
            if not os.path.exists(data_path) or generate:
                self.generate_variant_dataset(self.variant)
        else:
            self.variant, self.variant_folder = None, None
            data_path = os.path.join(self.processed_folder, data_file)

        self.data, self.targets = torch.load(data_path)

    def generate_variant_dataset(self, variant):
        """Generate a dataset corresponding to the given MNIST variant.

        The modified MNIST data will be saved in a similar fashion to
        that of the original MNIST dataset. Also, presumably some randomness will be
        involved, meaning the dataset will change every time this function is called.
        """
        # process and save as torch files
        print('Generating...')

        if not os.path.exists(self.variant_folder):
            os.makedirs(self.variant_folder)

        def _rot(image_data):
            """Destructive rotation."""
            for i in range(image_data.shape[0]):
                rand_deg = np.random.random() * 360.0
                image_data[i] = rotate(image_data[i], rand_deg, reshape=False)

        def _bg_rand(image_data):
            """Destructive random background."""
            noise = np.random.randint(
                0, 256, image_data.shape, dtype=image_data.dtype)
            image_data[image_data == 0] = noise[image_data == 0]

        for data_file in (self.training_file, self.test_file):
            # load original MNIST data
            data, targets = torch.load(os.path.join(self.processed_folder, data_file))

            modified_data = data.numpy()  # shape: (n, 28, 28)
            if variant == 'rot':
                _rot(modified_data)
            elif variant == 'bg_rand':
                _bg_rand(modified_data)
            elif variant == 'bg_rand_rot':
                _rot(modified_data)
                _bg_rand(modified_data)

            with open(os.path.join(self.variant_folder, data_file), 'wb') as f:
                torch.save((torch.from_numpy(modified_data), targets), f)

        print('Done!')
        print('Saved dataset to %s.' % self.variant_folder)

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')
