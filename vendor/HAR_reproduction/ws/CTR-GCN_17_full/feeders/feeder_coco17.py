import numpy as np
from torch.utils.data import Dataset

from feeders import tools

# Left/right joint pairs of COCO-17; the nose (0) is its own mirror.
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]
FLIP_INDEX = np.arange(17)
for left, right in FLIP_PAIRS:
    FLIP_INDEX[left], FLIP_INDEX[right] = right, left


class Feeder(Dataset):
    """RTMPose COCO-17 clips extracted from the ETRI RGB videos.

    The npz stores x_<split> as [N, C=3, T, V=17, M=2] with channels
    (x, y, score): x/y are normalised to [-1, 1] over the RGB image, the score
    is RTMPose's keypoint confidence. NTU-style rotation/flip augments assume
    metric 3D joints, so only crop-resize and additive noise are offered here.
    """

    def __init__(self, data_path, split='train', p_interval=1, window_size=-1,
                 random_flip=False, noise_std=0.0, exclude_classes=None, debug=False):
        self.data_path = data_path
        self.split = split
        # For the ZSL backbone: drop the unseen classes so the feature extractor
        # provably never sees them. Labels keep their original 0..54 ids.
        self.exclude_classes = set(exclude_classes or [])
        self.p_interval = p_interval
        self.window_size = window_size
        self.random_flip = random_flip
        self.noise_std = noise_std
        self.debug = debug
        self.load_data()

    def load_data(self):
        npz_data = np.load(self.data_path, allow_pickle=True)
        if self.split not in ('train', 'val', 'test'):
            raise NotImplementedError('data split only supports train/val/test')

        self.data = npz_data[f'x_{self.split}'].astype(np.float32)
        self.label = npz_data[f'y_{self.split}'].astype(np.int64)
        names = npz_data.get(f'{self.split}_sample_name')
        if names is None:
            self.sample_name = [f'{self.split}_{i}' for i in range(len(self.data))]
        else:
            self.sample_name = [str(name) for name in names]

        # Clips where the estimator never found a person carry no signal.
        valid = np.abs(self.data).sum(axis=(1, 2, 3, 4)) > 0
        dropped = int((~valid).sum())
        if dropped:
            print(f'Filtered {dropped} empty {self.split} skeleton samples from {self.data_path}')
            self.data = self.data[valid]
            self.label = self.label[valid]
            self.sample_name = [n for n, keep in zip(self.sample_name, valid) if keep]

        if self.exclude_classes:
            keep = ~np.isin(self.label, list(self.exclude_classes))
            print(f'Excluded {int((~keep).sum())} {self.split} samples of classes '
                  f'{sorted(self.exclude_classes)}')
            self.data = self.data[keep]
            self.label = self.label[keep]
            self.sample_name = [n for n, k in zip(self.sample_name, keep) if k]

        if self.debug:
            self.data = self.data[:100]
            self.label = self.label[:100]
            self.sample_name = self.sample_name[:100]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data_numpy = np.array(self.data[index])
        valid_frame_num = int(np.sum(data_numpy.sum(0).sum(-1).sum(-1) != 0))
        data_numpy = tools.valid_crop_resize(
            data_numpy, valid_frame_num, self.p_interval, self.window_size
        )
        if self.random_flip and np.random.rand() < 0.5:
            # Mirror the image: negate x and swap left/right joints. Only valid
            # for 2D image coordinates - the Kinect 3D pipeline cannot do this.
            data_numpy = data_numpy[:, :, FLIP_INDEX, :].copy()
            data_numpy[0] *= -1.0
        if self.noise_std > 0:
            # Perturb coordinates only; the score channel stays as measured.
            noise = np.random.randn(*data_numpy[:2].shape).astype(np.float32) * self.noise_std
            data_numpy = data_numpy.copy()
            data_numpy[:2] += noise
        return data_numpy, self.label[index], index

    def top_k(self, score, top_k):
        rank = score.argsort()
        hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod
