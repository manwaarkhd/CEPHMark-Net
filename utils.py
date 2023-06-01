from typing import Union
import tensorflow as tf
from config import cfg
import numpy as np
import math
import cv2
import os

def craniofacial_region_proposals(
    landmarks: Union[tf.Tensor, np.ndarray],
    image_height: int,
    image_width: int,
    margin: int = cfg.BOX_MARGIN
):
    landmarks = tf.reshape(landmarks, (-1, cfg.NUM_LANDMARKS, 2))

    x_min = tf.reduce_min(landmarks, axis=1)[:, 0] - margin
    y_min = tf.reduce_min(landmarks, axis=1)[:, 1] - margin
    x_max = tf.reduce_max(landmarks, axis=1)[:, 0] + margin
    y_max = tf.reduce_max(landmarks, axis=1)[:, 1] + margin

    bounding_boxes = tf.stack([
        x_min / image_width,            # x
        y_min / image_height,           # y
        (x_max - x_min) / image_width,  # width
        (y_max - y_min) / image_height  # height
    ], axis=-1)

    return clip_bounding_boxes(bounding_boxes)

def clip_bounding_boxes(boxes):
    boxes = tf.reshape(boxes, shape=(-1, 1, 4))

    boxes = transform_bounding_boxes(boxes, mode="xyxy")
    boxes = tf.clip_by_value(
        boxes,
        clip_value_min=0.0,
        clip_value_max=1.0
    )
    boxes = transform_bounding_boxes(boxes, mode="xywh")

    return boxes

def decode_bounding_boxes(
    bounding_boxes,
    image_height,
    image_width
):
    bboxes = tf.reshape(bounding_boxes, (-1, 1, 4))

    bboxes = tf.stack([
        bboxes[:, :, 0] * image_width,
        bboxes[:, :, 1] * image_height,
        bboxes[:, :, 2] * image_width,
        bboxes[:, :, 3] * image_height],
        axis=2
    )

    return bboxes

def craniofacial_landmark_regions(
    landmarks: tf.Tensor,
    height: int,
    width: int,
    size: int = 3
):
    offset = size / 2

    proposals = tf.stack([
        landmarks[:, :, 0] - offset / width,
        landmarks[:, :, 1] - offset / height,
        landmarks[:, :, 0] + offset / width,
        landmarks[:, :, 1] + offset / height
    ], axis=-1)
    proposals = tf.clip_by_value(proposals, clip_value_min=0.0, clip_value_max=1.0)

    return proposals

def encode_cephalometric_landmarks(
    landmarks: tf.Tensor,
    height: int,
    width: int
):
    landmarks = tf.reshape(landmarks, shape=(-1, cfg.NUM_LANDMARKS, 2))

    landmarks = tf.stack([
        landmarks[:, :, 0] / width,
        landmarks[:, :, 1] / height
    ], axis=-1)

    return landmarks

def decode_cephalometric_landmarks(
    landmarks: tf.Tensor,
    height: int,
    width: int
):
    landmarks = tf.reshape(landmarks, shape=(-1, cfg.NUM_LANDMARKS, 2))

    landmarks = tf.stack([
        landmarks[:, :, 0] * width,
        landmarks[:, :, 1] * height
    ], axis=-1)

    return landmarks

def transform_bounding_boxes(
    boxes: tf.Tensor,
    mode: str = "xyxy"
):
    boxes = tf.reshape(boxes, shape=(-1, 1, 4))

    if mode == "xyxy":
        x1 = boxes[:, :, 0]
        y1 = boxes[:, :, 1]
        x2 = boxes[:, :, 0] + boxes[:, :, 2]
        y2 = boxes[:, :, 1] + boxes[:, :, 3]

        return tf.stack([x1, y1, x2, y2], axis=-1)

    elif mode == "xywh":
        x = boxes[:, :, 0]
        y = boxes[:, :, 1]
        w = boxes[:, :, 2] - boxes[:, :, 0]
        h = boxes[:, :, 3] - boxes[:, :, 1]

        return tf.stack([x, y, w, h], axis=-1)

    else:
        raise ValueError("inappropriate mode value")

def convert_image_dtype(image: np.ndarray, dtype: str):
    if dtype == "float32" or dtype == "float64":
        image = image.astype(dtype) / 255
    elif dtype == "uint8":
        image = (image * 255).astype(dtype)
    else:
        raise ValueError("{} no such dtype exists.".format(dtype))

    return image

def rescale_input(inputs: tf.Tensor, scale: float, offset: int = 0.0, dtype=tf.float32):
    scale = tf.cast(scale, dtype)
    offset = tf.cast(offset, dtype)
    return tf.cast(inputs, dtype) * scale + offset


def clear_statistics(path, mode: str):
    stats = np.array([])
    if mode == "all":
        for mode in ["train", "valid", "test"]:
            file_name = mode + "_stats"
            file = os.path.join(path, file_name)
            np.save(file, stats)
    elif mode in ["train", "valid", "test"]:
        file_name = mode + "_stats"
        file = os.path.join(path, file_name)
        np.save(file, stats)
    else:
        raise ValueError("{} is not a valid mode.".format(mode))

def save_statistics(results: np.ndarray, path: str, mode: str):
    if mode in ["train", "valid", "test"]:
        file_name = mode + "_stats.npy"
        path = os.path.join(path, file_name)
        if os.path.isfile(path):
            stats = list(np.load(path))
        else:
            stats = []
        stats.append(results)
        stats = np.vstack(stats)
        np.save(path, stats)
    else:
        raise ValueError("{} is not a valid mode.".format(mode))
