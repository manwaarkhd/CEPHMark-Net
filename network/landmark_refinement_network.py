from tensorflow.keras.layers import Conv2D, Flatten, Dense
import tensorflow as tf
from config import cfg


class LandmarkRefinementNetwork(object):

    def __init__(
        self,
        inputs: tf.keras.layers
    ):
        self.inputs = inputs

        self.outputs = []
        for index in range(cfg.NUM_LANDMARKS):
            x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="refinement_head" + str(index + 1) + "_conv2d")(self.inputs)
            x = Flatten(name="refinement_head" + str(index + 1) + "_flatten")(x)
            x = Dense(units=512, activation="relu", name="refinement_head" + str(index + 1) + "_fc")(x)
            head = Dense(
                units=2,
                activation="linear",
                use_bias=False,
                name="refinement_head" + str(index + 1) + "_output"
            )(x)

            self.outputs.append(head)
