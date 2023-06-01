from utils import convert_image_dtype, craniofacial_region_proposals, decode_bounding_boxes, transform_bounding_boxes
from scipy.ndimage.filters import gaussian_filter, maximum_filter, minimum_filter
from PIL import Image, ImageOps
import numpy as np
import cv2


def histogram_equalization(image: np.ndarray):
    image = cv2.equalizeHist(image[:, :, 0])
    image = np.repeat(
        np.expand_dims(image, axis=-1),
        repeats=3,
        axis=-1
    )

    return image


def unsharp_masking(image: np.ndarray, alpha: float = 1.0):
    filter = np.random.choice([gaussian_filter, maximum_filter, minimum_filter])

    image = convert_image_dtype(image, dtype="float32")
    blurred_image = filter(image, 3)

    image = image + alpha * (image - blurred_image)
    image = np.clip(image, 0.0, 1.0)
    image = convert_image_dtype(image, dtype="uint8")

    return image


def inversion(image: np.ndarray):
    image = Image.fromarray(image)
    image = ImageOps.invert(image)
    image = np.array(image)

    return image


def solarization(image: np.ndarray):
    image = Image.fromarray(image)
    image = ImageOps.solarize(image)
    image = np.array(image)

    return image


def contrast_limited_histogram_equalization(image: np.ndarray):
    clahe = cv2.createCLAHE(clipLimit=1, tileGridSize=(128, 128))

    image = clahe.apply(image[:, :, 0])
    image = np.repeat(
        np.expand_dims(image, axis=-1),
        repeats=3,
        axis=-1
    )

    return image


def high_pass_filtering(image: np.ndarray):
    def high_pass_filter(image: np.ndarray):
        height, width = image.shape[:-1]

        center_x = image.shape[0] // 2
        center_y = image.shape[1] // 2

        filter = np.ones(shape=(height, width, 2), dtype=np.uint8)
        x, y = np.ogrid[:height, :width]
        region = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius * radius
        filter[region] = 0

        return filter

    radius = np.random.uniform()

    image = convert_image_dtype(image, dtype="float32")
    dft = cv2.dft(image[:, :, 0], flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    mask = high_pass_filter(image)
    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    image = cv2.idft(f_ishift)
    image = cv2.magnitude(image[:, :, 0], image[:, :, 1])
    image = np.uint8(cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, -1))

    image = np.repeat(
        np.expand_dims(image, axis=-1),
        repeats=3,
        axis=-1
    )

    return image


def horizontal_shift(image: np.ndarray, landmarks: np.ndarray, margin: int = 100):
    image_height, image_width = image.shape[0:2]
    x_min, x_max = landmarks[:, 0].min(), landmarks[:, 0].max()

    shift = np.random.choice(["left", "right"])
    if shift == "right":
        value = -np.random.randint(0, image_width - (x_max + margin))
    else:
        value = np.random.randint(0, (x_min - margin))

    image = Image.fromarray(image)
    image = image.transform(image.size, Image.AFFINE, (1, 0, value, 0, 1, 0))
    image = np.array(image)

    landmarks[:, 0] -= value

    return image, landmarks


def vertical_shift(image: np.ndarray, landmarks: np.ndarray, margin: int = 100):
    image_height, image_width = image.shape[0:2]
    y_min, y_max = landmarks[:, 1].min(), landmarks[:, 1].max()

    shift = np.random.choice(["upward", "downward"])
    if shift == "downward":
        value = -np.random.randint(0, image_height - (y_max + margin))
    else:
        value = np.random.randint(0, (y_min - margin))

    image = Image.fromarray(image)
    image = image.transform(image.size, Image.AFFINE, (1, 0, 0, 0, 1, value))
    image = np.array(image)

    landmarks[:, 1] -= value

    return image, landmarks


def random_cropping(image: np.ndarray, landmarks: np.ndarray):
    image_height, image_width = image.shape[0:2]
    margin = np.random.randint(128, 192)

    bbox = craniofacial_region_proposals(landmarks, image_height, image_width, margin)
    bbox = decode_bounding_boxes(bbox, image_height, image_width)
    bbox = transform_bounding_boxes(bbox, mode="xyxy")
    bbox = np.squeeze(bbox)

    x1, y1, x2, y2 = bbox

    image = image[int(y1):int(y2), int(x1):int(x2)]
    landmarks[:, 0] -= x1
    landmarks[:, 1] -= y1

    return image, landmarks


def horizontal_flip(image: np.ndarray, landmarks: np.ndarray):
    image_height, image_width = image.shape[0:2]

    image = Image.fromarray(image)
    image = ImageOps.mirror(image)
    image = np.array(image)

    landmarks[:, 0] = image_width - landmarks[:, 0]

    return image, landmarks


def vertical_flip(image: np.ndarray, landmarks: np.ndarray):
    image_height, image_width = image.shape[0:2]

    image = Image.fromarray(image)
    image = ImageOps.flip(image)
    image = np.array(image)

    landmarks[:, 1] = image_height - landmarks[:, 1]

    return image, landmarks


def identity(image: np.ndarray, landmarks: np.ndarray = None):
    if landmarks is None:
        return image
    else:
        return image, landmarks
