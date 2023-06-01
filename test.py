from utils import encode_cephalometric_landmarks, decode_cephalometric_landmarks, craniofacial_landmark_regions, rescale_input, save_statistics
from network.model import Network
from data import Dataset
from paths import Paths
import tensorflow as tf
from config import cfg


def test_step(
    data: list,
    network: Network
):
    images, landmarks = data
    image_height, image_width = images.shape[1:-1]

    true_locations = encode_cephalometric_landmarks(landmarks, image_height, image_width)
    pred_landmarks = network.landmark_detection_module(inputs=images, training=False)
    pred_landmarks = tf.stack([pred_landmarks[:, 0::2], pred_landmarks[:, 1::2]], axis=-1)

    block3_proposals = craniofacial_landmark_regions(pred_landmarks, height=(image_height / 8),  width=(image_width / 8),  size=7)
    block4_proposals = craniofacial_landmark_regions(pred_landmarks, height=(image_height / 16), width=(image_width / 16), size=5)
    block5_proposals = craniofacial_landmark_regions(pred_landmarks, height=(image_height / 32), width=(image_width / 32), size=3)
    proposals = tf.stack([block3_proposals, block4_proposals, block5_proposals])

    pred_locations = []
    for index in range(cfg.NUM_LANDMARKS):
        candidate_regions = proposals[:, :, index, :]
        refine_locations = network.landmark_refinement_module.heads[index](inputs=[images, candidate_regions], training=False) + pred_landmarks[:, index, :]
        pred_locations.append(refine_locations)
    pred_locations = tf.stack(pred_locations, axis=1)

    true_landmarks = decode_cephalometric_landmarks(true_locations, height=cfg.ORIGINAL_HEIGHT, width=cfg.ORIGINAL_WIDTH)
    pred_landmarks = decode_cephalometric_landmarks(pred_locations, height=cfg.ORIGINAL_HEIGHT, width=cfg.ORIGINAL_WIDTH)

    delta_x = pred_landmarks[:, :, 0] - true_landmarks[:, :, 0]
    delta_y = pred_landmarks[:, :, 1] - true_landmarks[:, :, 1]

    landmark_errors = tf.sqrt(
        tf.square(delta_x) + tf.square(delta_y)
    )

    return landmark_errors


def test_on_batch(
    data: Dataset,
    network: Network
):
    landmark_errors = []
    for index in range(len(data)):
        images, landmarks = data[index]
        images = rescale_input(images, scale=(1 / 255), offset=0)

        errors = test_step(
            data=[
                images,
                landmarks
            ],
            network=network
        )

        landmark_errors.append(errors)
    return tf.reshape(tf.stack(landmark_errors, axis=0), shape=(-1, cfg.NUM_LANDMARKS))


def test(
    data: Dataset,
    network: Network
):
    landmark_errors = test_on_batch(data, network) * cfg.IMAGE_RESOLUTION
    errors = tf.math.reduce_mean(landmark_errors, axis=0)
    stds = tf.sqrt(
        tf.reduce_mean(
            tf.square(landmark_errors - errors),
            axis=0
        )
    )

    sdr_2dot0mm = tf.reduce_mean(
        tf.cast(
            tf.less_equal(
                landmark_errors,
                tf.constant(2, dtype=tf.float32)
            ), dtype=tf.float32
        ), axis=0
    )

    sdr_2dot5mm = tf.reduce_mean(
        tf.cast(
            tf.less_equal(
                landmark_errors,
                tf.constant(2.5, dtype=tf.float32)
            ), dtype=tf.float32
        ), axis=0
    )

    sdr_3dot0mm = tf.reduce_mean(
        tf.cast(
            tf.less_equal(
                landmark_errors,
                tf.constant(3.0, dtype=tf.float32)
            ), dtype=tf.float32
        ), axis=0
    )

    sdr_4dot0mm = tf.reduce_mean(
        tf.cast(
            tf.less_equal(
                landmark_errors,
                tf.constant(4.0, dtype=tf.float32)
            ), dtype=tf.float32
        ), axis=0
    )

    print("-" * 75)
    print("      Landmarks           MRE  \u00B1  STD     2.0mm    2.5mm    3.0mm    4.0mm")
    print("-" * 75)
    for index in range(cfg.NUM_LANDMARKS):
        print("%22s%8.2f \u00B1 %5.2f %8.2f %8.2f %8.2f %8.2f" % (
            cfg.ANATOMICAL_LANDMARKS[str(index)],
            errors[index],
            stds[index],
            sdr_2dot0mm[index] * 100,
            sdr_2dot5mm[index] * 100,
            sdr_3dot0mm[index] * 100,
            sdr_4dot0mm[index] * 100,
        ))
    print("-" * 75)
    print("%22s%8.2f \u00B1 %5.2f %8.2f %8.2f %8.2f %8.2f" % (
        "Average",
        tf.reduce_mean(errors),
        tf.reduce_mean(stds),
        tf.reduce_mean(sdr_2dot0mm) * 100,
        tf.reduce_mean(sdr_2dot5mm) * 100,
        tf.reduce_mean(sdr_3dot0mm) * 100,
        tf.reduce_mean(sdr_4dot0mm) * 100,
    ))

if __name__ == "__main__":
    from network.model import Network
    from data import Dataset
    from config import cfg

    train_data = Dataset(name="isbi", mode="test", batch_size=1, shuffle=False)

    network = Network(
        backbone_name="resnet50",
        freeze_backbone=False,
        backbone_weights=None
    )

    test(train_data, network)

