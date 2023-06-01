from data import ISBIDataset, PKUDataset
from preprocessing import Augmentation
from paths import Paths
import tensorflow as tf
from config import cfg
import numpy as np
import cv2


def resize(image: np.ndarray, landmarks: np.ndarray):
    image_height, image_width = image.shape[0:2]
    ratio_height, ratio_width = (image_height / cfg.HEIGHT), (image_width / cfg.WIDTH)

    image = cv2.resize(np.array(image), dsize=(cfg.WIDTH, cfg.HEIGHT), interpolation=cv2.INTER_CUBIC)
    landmarks = np.vstack([
        landmarks[:, 0] / ratio_width,
        landmarks[:, 1] / ratio_height
    ]).T

    return image, landmarks


class Dataset(tf.keras.utils.Sequence):

    def __init__(
        self,
        name: str,
        mode: str,
        batch_size: int = 1,
        augmentation: Augmentation = None,
        shuffle: bool = False,
    ):

        if name == "isbi":
            self.dataset = ISBIDataset(Paths.dataset_root_path(name), mode)
        elif name == "pku":
            self.dataset = PKUDataset(Paths.dataset_root_path(name), mode)
        else:
            raise ValueError("\'{}\' no such dataset exists in your datasets repository.".format(name))

        self.batch_size = batch_size
        self.shuffle = shuffle

        if self.shuffle:
            self.dataset.shuffle()

        self.augmentation = augmentation

    def on_epoch_end(self):
        if self.shuffle:
            self.dataset.shuffle()

    def __getitem__(self, index: int):

        start_index = index * self.batch_size
        end_index = min((index + 1) * self.batch_size, len(self.dataset))

        images = []
        labels = []

        for index in range(start_index, end_index):
            image, landmarks = self.dataset[index]

            if self.augmentation is not None:
                image, landmarks = self.augmentation.apply(image, landmarks)

            image, landmarks = resize(image, landmarks)
            images.append(image)
            labels.append(landmarks)

        return tf.stack(images), tf.stack(labels)

    def __len__(self):
        return len(self.dataset) // self.batch_size
