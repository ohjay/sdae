import numpy as np
import scipy.io as sio
from torch.utils import data
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
