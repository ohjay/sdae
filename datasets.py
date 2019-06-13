import os
import h5py
import torch
import pickle
import imageio
import numpy as np
import scipy.io as sio
from torch.utils import data
from scipy.ndimage import rotate
from skimage.transform import resize
from skimage.util import view_as_windows
from torchvision.datasets import MNIST, CIFAR10


def crop(img, bounding_box, data_format='hwc'):
    x, y, width, height = bounding_box
    if data_format.lower() in {'hw', 'hwc'}:
        return img[y:y+height, x:x+width]
    elif data_format.lower() in {'chw', 'nhw', 'nhwc'}:
        return img[:, y:y+height, x:x+width]
    elif data_format.lower() == 'nchw':
        return img[:, :, y:y+height, x:x+width]


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
            if self.variant not in self.variant_options:
                if self.variant.startswith('mnist'):
                    self.variant = self.variant[5:]
                if self.variant.startswith('_'):
                    self.variant = self.variant[1:]
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

    @staticmethod
    def get_minval():
        return 0.0  # assumes normalization

    @staticmethod
    def get_maxval():
        return 1.0  # assumes normalization


class CUB2011Dataset(data.Dataset):
    """Caltech-UCSD Birds-200-2011.
    Available here: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html.
    """
    training_file = 'train.h5'
    eval_file = 'eval.h5'

    RESIZE_H = 128
    RESIZE_W = 128

    def __init__(self, cub_folder, train=True, normalize=False, generate=False):
        """CUB_FOLDER should contain the following items (among other things):
        images/, images.txt, train_test_split.txt, classes.txt, image_class_labels.txt."""
        super(CUB2011Dataset, self).__init__()
        self.cub_folder = cub_folder
        self.train = train  # training set or test set

        self.classes_path = os.path.join(self.cub_folder, 'classes.pkl')
        if not os.path.exists(self.classes_path):
            self.process_classes()
        with open(self.classes_path, 'rb') as f:
            self.classes = pickle.load(f)

        self.bounding_boxes_path = os.path.join(self.cub_folder, 'bounding_boxes.pkl')
        if not os.path.exists(self.bounding_boxes_path):
            self.process_bounding_boxes()
        with open(self.bounding_boxes_path, 'rb') as f:
            self.bounding_boxes = pickle.load(f)

        if self.train:
            data_path = os.path.join(self.cub_folder, self.training_file)
        else:
            data_path = os.path.join(self.cub_folder, self.eval_file)

        if not os.path.exists(data_path) or generate:
            self.process_images_and_labels()

        h5f = h5py.File(data_path, 'r')
        self.images = h5f['images'][:]  # shape: (n, 3, h, w)
        self.labels = h5f['labels'][:]  # shape: (n,)
        h5f.close()

        if normalize:
            # normalize to range [0, 1]
            _min = self.images.min()
            self.images = (self.images - _min) / (self.images.max() - _min)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], self.labels[index]  # (image, label)

    def process_classes(self):
        classes = {}
        with open(os.path.join(self.cub_folder, 'classes.txt')) as f:
            for line in f:
                class_id, class_name = line.strip().split()
                class_id = int(class_id) - 1
                classes[class_id] = class_name
        with open(self.classes_path, 'wb') as f:
            pickle.dump(classes, f, pickle.HIGHEST_PROTOCOL)

    def process_bounding_boxes(self):
        bounding_boxes = {}
        with open(os.path.join(self.cub_folder, 'bounding_boxes.txt')) as f:
            for line in f:
                image_id, x, y, width, height = line.strip().split()
                image_id = int(image_id) - 1
                x, y, width, height = [int(float(t)) for t in (x, y, width, height)]
                bounding_boxes[image_id] = (x, y, width, height)
        with open(self.bounding_boxes_path, 'wb') as f:
            pickle.dump(bounding_boxes, f, pickle.HIGHEST_PROTOCOL)

    def process_images_and_labels(self):
        print('Generating...')

        train_test_split = {}
        with open(os.path.join(self.cub_folder, 'train_test_split.txt')) as f:
            for line in f:
                image_id, is_training_image = line.strip().split()
                image_id = int(image_id) - 1
                train_test_split[image_id] = bool(int(is_training_image))

        image_class_labels = {}
        with open(os.path.join(self.cub_folder, 'image_class_labels.txt')) as f:
            for line in f:
                image_id, class_id = [int(t) - 1 for t in line.strip().split()]
                image_class_labels[image_id] = class_id

        train_images, train_labels = [], []
        eval_images, eval_labels = [], []
        with open(os.path.join(self.cub_folder, 'images.txt')) as f:
            for line in f:
                image_id, image_name = line.strip().split()
                image_id = int(image_id) - 1
                image_path = os.path.join(self.cub_folder, 'images', image_name)
                image = imageio.imread(image_path)
                image = crop(image, self.bounding_boxes[image_id])
                image = resize(image, (self.RESIZE_H, self.RESIZE_W))
                if len(image.shape) == 2:
                    # convert to three-channel image
                    image = np.stack([image] * 3, axis=-1)
                image = np.transpose(image, (2, 0, 1))  # reshape to (c, h, w)
                assert image.shape == (3, self.RESIZE_H, self.RESIZE_W), image_name
                if train_test_split[image_id]:
                    train_images.append(image)
                    train_labels.append(image_class_labels[image_id])
                else:
                    eval_images.append(image)
                    eval_labels.append(image_class_labels[image_id])
        train_images, train_labels = [np.array(ta) for ta in (train_images, train_labels)]
        eval_images, eval_labels = [np.array(ea) for ea in (eval_images, eval_labels)]

        train_eval_triplets = [
            (os.path.join(self.cub_folder, self.training_file), train_images, train_labels),
            (os.path.join(self.cub_folder, self.eval_file), eval_images, eval_labels)
        ]
        for data_path, images, labels in train_eval_triplets:
            h5f = h5py.File(data_path, 'w')
            h5f.create_dataset('images', data=images)
            h5f.create_dataset('labels', data=labels)
            h5f.close()

        print('Done!')
        print('Saved processed dataset to %s.' % self.cub_folder)

    def get_minval(self):
        return self.images.min()

    def get_maxval(self):
        return self.images.max()


class CIFAR10Dataset(CIFAR10):
    """CIFAR-10."""

    @staticmethod
    def get_minval():
        return 0.0  # assumes normalization

    @staticmethod
    def get_maxval():
        return 1.0  # assumes normalization


class InterpolationDataset(data.Dataset):
    """Interpolation between two grayscale images."""

    # change these
    image0_path = 'dandelion0.jpg'
    image1_path = 'dandelion1.jpg'
    dataset_len = 6400

    def __init__(self, root, normalize=True, generate=False):
        super(InterpolationDataset, self).__init__()
        self.root = root

        self.image0 = imageio.imread(self.image0_path, as_gray=True)
        self.image1 = imageio.imread(self.image1_path, as_gray=True)

        self.data_path = os.path.join(self.root, 'interpolation.h5')
        if not os.path.exists(self.data_path) or generate:
            self.generate()

        h5f = h5py.File(self.data_path, 'r')
        self.images = h5f['images'][:]  # shape: (n, 1, h, w)
        h5f.close()

        if normalize:
            # normalize to range [0, 1]
            _min = self.images.min()
            self.images = (self.images - _min) / (self.images.max() - _min)
        print('image statistics:')
        print('min: %r, mean: %r, max: %r'
              % (self.images.min(), self.images.mean(), self.images.max()))

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, index):
        return self.images[index], index  # (image, "label")

    def generate(self):
        print('Generating...')

        images = []
        for i in range(self.dataset_len):
            a = float(i) / (self.dataset_len - 1)
            images.append(a * self.image0 + (1.0 - a) * self.image1)
        images = np.array(images)
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=1)
        elif len(images.shape) == 4:
            images = np.transpose(images, (0, 3, 1, 2))
        assert images.shape[1] == 1  # single-channel images

        if images.dtype != np.float:
            images = images.astype(np.float)
        if images.max() > 1.0:
            images /= images.max()

        h5f = h5py.File(self.data_path, 'w')
        h5f.create_dataset('images', data=images)
        h5f.close()

        print('Done!')
        print('Saved dataset to %s.' % self.data_path)

    def get_minval(self):
        return self.images.min()

    def get_maxval(self):
        return self.images.max()

    @property
    def sample_h(self):
        return self.images.shape[2]

    @property
    def sample_w(self):
        return self.images.shape[3]
