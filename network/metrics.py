from tensorflow.keras.metrics import Metric
import tensorflow as tf

class SuccessfulDetectionRate(Metric):

    def __init__(self, precision_range: int = 40):
        super(SuccessfulDetectionRate, self).__init__()
        self.precision_range = precision_range
        self.successful_detection_rate = None

    def update_state(self, y_true, y_pred):
        delta_x = y_pred[:, :, 0] - y_true[:, :, 0]
        delta_y = y_pred[:, :, 1] - y_true[:, :, 1]

        errors = tf.sqrt(tf.square(delta_x) + tf.square(delta_y))

        self.successful_detection_rate = tf.reduce_mean(
            tf.cast(
                errors <= self.precision_range,
                tf.float32
            )
        )

    def result(self):
        return self.successful_detection_rate
