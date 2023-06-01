from tensorflow.keras.layers import Layer
import tensorflow as tf


class ROIAlign2D(Layer):

    def __init__(self, crop_size: tuple, name: str):
        super(ROIAlign2D, self).__init__()
        self.crop_size = crop_size
        self._name = name

    def call(self, inputs, *args, **kwargs):
        feature_maps, roi_proposals = inputs

        if roi_proposals.shape[0] is not None:
            x1, y1, x2, y2 = tf.split(roi_proposals, 4, axis=-1)
            roi_proposals = tf.concat([y1, x1, y2, x2], axis=-1)

        roi_proposals = tf.stop_gradient(roi_proposals)

        cropped_maps = []
        for index in range(len(feature_maps)):
            cropped_featuremap = tf.image.crop_and_resize(
                feature_maps[index],
                roi_proposals[index],
                box_indices=tf.range(tf.shape(roi_proposals)[1]),
                crop_size=self.crop_size,
                method="bilinear"
            )
            cropped_maps.append(cropped_featuremap)
        cropped_maps = tf.concat(cropped_maps, axis=-1)

        return cropped_maps
