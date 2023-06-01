from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, BatchNormalization, Activation, Add
from tensorflow.keras.models import Model
import tensorflow as tf


class Darknet19(Model):

    def __init__(
        self,
        inputs: tf.keras.layers.Layer = None,
        input_shape: tf.TensorShape = None,
        weights: str = None
    ):
        if inputs is None:
            if input_shape is None:
                self.inputs = Input(shape=(None, None, 3), name="input")
            else:
                self.inputs = Input(shape=input_shape, name="input")
        else:
            self.inputs = inputs

        # block1
        x = Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same", name="block1_conv1")(self.inputs)
        x = MaxPool2D(pool_size=(2, 2), strides=2, name="block1_pool")(x)

        # block2
        x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same", name="block2_conv1")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, name="block2_pool")(x)

        # block3
        x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv1")(x)
        x = Conv2D(filters=64,  kernel_size=(1, 1), activation="relu", padding="same", name="block3_conv2")(x)
        x = Conv2D(filters=128, kernel_size=(3, 3), activation="relu", padding="same", name="block3_conv3")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, name="block3_pool")(x)

        # block4
        x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv1")(x)
        x = Conv2D(filters=128, kernel_size=(1, 1), activation="relu", padding="same", name="block4_conv2")(x)
        x = Conv2D(filters=256, kernel_size=(3, 3), activation="relu", padding="same", name="block4_conv3")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, name="block4_pool")(x)

        # block5
        x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv1")(x)
        x = Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", name="block5_conv2")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv3")(x)
        x = Conv2D(filters=256, kernel_size=(1, 1), activation="relu", padding="same", name="block5_conv4")(x)
        x = Conv2D(filters=512, kernel_size=(3, 3), activation="relu", padding="same", name="block5_conv5")(x)
        x = MaxPool2D(pool_size=(2, 2), strides=2, name="block5_pool")(x)

        # block6
        x = Conv2D(filters=1024, kernel_size=(3, 3), activation="relu", padding="same", name="block6_conv1")(x)
        x = Conv2D(filters=512,  kernel_size=(1, 1), activation="relu", padding="same", name="block6_conv2")(x)
        x = Conv2D(filters=1024, kernel_size=(3, 3), activation="relu", padding="same", name="block6_conv3")(x)
        x = Conv2D(filters=512,  kernel_size=(1, 1), activation="relu", padding="same", name="block6_conv4")(x)
        x = Conv2D(filters=1024, kernel_size=(3, 3), activation="relu", padding="same", name="block6_conv5")(x)

        self.outputs = Conv2D(
            filters=1024,
            kernel_size=(1, 1),
            activation="relu",
            padding="same",
            name="output_block_conv2d"
        )(x)
        super(Darknet19, self).__init__(inputs=self.inputs, outputs=self.outputs, name="Darknet19")


class Darknet53(Model):

    def __init__(
        self,
        inputs: tf.keras.layers.Layer = None,
        input_shape: tf.TensorShape = None,
        weights: str = None
    ):

        if inputs is None:
            if input_shape is None:
                self.inputs = Input(shape=(None, None, 3), name="input")
            else:
                self.inputs = Input(shape=input_shape, name="input")
        else:
            self.inputs = inputs

        x = self.conv3x3(self.inputs, filters=32, strides=1, name="stem_block_conv2d")
        x = BatchNormalization(momentum=0.99, epsilon=0.001, name="stem_block_bn")(x)
        block0_output = Activation(activation="relu", name="stem_block_activation")(x)

        block1_output = self.make_layers(block0_output, filters=64,   num_blocks=1, name="block1")
        block2_output = self.make_layers(block1_output, filters=128,  num_blocks=2, name="block2")
        block3_output = self.make_layers(block2_output, filters=256,  num_blocks=8, name="block3")
        block4_output = self.make_layers(block3_output, filters=512,  num_blocks=8, name="block4")
        block5_output = self.make_layers(block4_output, filters=1024, num_blocks=4, name="block5")

        self.outputs = block5_output
        super(Darknet53, self).__init__(inputs=self.inputs, outputs=self.outputs, name="Darknet53")

    def conv3x3(self, x, filters: int, strides: int, name: str):
        return Conv2D(
            filters=filters,
            kernel_size=(3, 3),
            strides=strides,
            padding="same",
            name=name
        )(x)

    def residual_block(self, x, filters: int, name: str, activation: str = "relu", strides: int = 1):
        identity = x

        x = self.conv3x3(x, filters=filters // 2, strides=strides, name=f"{name}_conv1")
        x = BatchNormalization(momentum=0.99, epsilon=0.001, name=f"{name}_batchnorm1")(x)
        x = Activation(activation, name=f"{name}_activation1")(x)

        x = self.conv3x3(x, filters=filters, strides=strides, name=f"{name}_conv2")
        x = BatchNormalization(momentum=0.99, epsilon=0.001, name=f"{name}_batchnorm2")(x)
        x = Activation(activation, name=f"{name}_activation2")(x)

        x = Add(name=f"{name}_out")([identity, x])

        return x

    def make_layers(
        self,
        x,
        filters: int,
        num_blocks: int,
        name: str
    ):
        x = self.conv3x3(x, filters=filters, strides=2, name=f"{name}.0_conv2d")
        x = BatchNormalization(momentum=0.99, epsilon=0.001, name=f"{name}.0_batchnorm")(x)
        x = Activation(activation="relu", name=f"{name}.0_activation")(x)

        for index in range(num_blocks):
            x = self.residual_block(
                x,
                filters=filters,
                name=f"{name}.{index + 1}"
            )

        return x
