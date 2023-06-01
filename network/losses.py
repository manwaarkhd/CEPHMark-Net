from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from tensorflow.keras.losses import Loss
import tensorflow as tf


class MeanSquaredError(Loss):

    def __init__(self):
        super(MeanSquaredError, self).__init__()

    def call(self, y_true, y_pred, mask=None):

        loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
        if mask is not None:
            loss = loss * mask

        return loss


class MeanRadialError(Loss):

    def __init__(self):
        super(MeanRadialError, self).__init__()

    def call(self, y_true, y_pred):

        delta_x = y_pred[:, :, 0] - y_true[:, :, 0]
        delta_y = y_pred[:, :, 1] - y_true[:, :, 1]

        loss = tf.reduce_mean(
            tf.sqrt(
                tf.square(delta_x) + tf.square(delta_y)
            )
        )

        return loss
