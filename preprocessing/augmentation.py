from preprocessing.utils import identity, high_pass_filtering, solarization, unsharp_masking, inversion, \
    contrast_limited_histogram_equalization, horizontal_flip, random_cropping, vertical_shift
from config import cfg
import numpy as np


class Augmentation(object):

    def __init__(
        self,
        highpass_filter: bool = False,
        unsharp_mask: bool = False,
        solarize: bool = False,
        invert: bool = False,
        clahe: bool = False,
        random_flip: bool = False,
        random_shift: bool = False,
        random_crop: bool = False,
        landmark_shift: bool = False
    ):
        self.photometric_transformations = [identity]
        if highpass_filter:
            self.photometric_transformations.append(high_pass_filtering)
        if solarize:
            self.photometric_transformations.append(solarization)
        if unsharp_mask:
            self.photometric_transformations.append(unsharp_masking)
        if invert:
            self.photometric_transformations.append(inversion)
        if clahe:
            self.photometric_transformations.append(contrast_limited_histogram_equalization)

        self.geometric_transformations = []
        if random_flip:
            self.geometric_transformations.append(horizontal_flip)
        if random_shift:
            self.geometric_transformations.append(vertical_shift)
        if random_crop:
            self.geometric_transformations.append(random_cropping)

        self.landmark_shift = landmark_shift

    def apply(self, image: np.ndarray, landmarks: np.ndarray):
        # Photometric Transformation
        photometric_transforma_fn = np.random.choice(self.photometric_transformations)
        image = photometric_transforma_fn(image)

        # Geometric Transformation
        np.random.shuffle(self.geometric_transformations)
        for geometric_transform_fn in self.geometric_transformations:
            prob = np.random.uniform()
            if prob < 0.5:
                image, landmarks = geometric_transform_fn(image, landmarks)

        # Cephalometric Transformations
        choice = np.random.uniform()
        if 0.0 < choice < 0.25 and self.landmark_shift:
            transformed_landmarks = np.zeros_like(landmarks)
            transformed_landmarks[:, 0] = landmarks[:, 0] - np.random.randint(-10, 10, cfg.NUM_LANDMARKS)
            transformed_landmarks[:, 1] = landmarks[:, 1] - np.random.randint(-10, 10, cfg.NUM_LANDMARKS)
            landmarks = transformed_landmarks

        return image, landmarks
