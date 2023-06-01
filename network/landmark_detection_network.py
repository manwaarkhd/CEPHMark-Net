from tensorflow.keras.layers import Flatten, Dense
import tensorflow as tf
from config import cfg


class LandmarkDetectionNetwork(object):

    def __init__(
        self,
        inputs: tf.keras.layers
    ):
        self.inputs = inputs
        x = Flatten(name="detection_block_flatten")(self.inputs)
        self.outputs = Dense(
            units=cfg.NUM_LANDMARKS*2,
            activation="linear",
            use_bias=False,
            name="detection_block_output"
        )(x)
