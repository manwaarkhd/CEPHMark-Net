from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, ReLU, MaxPool2D, Add, Rescaling
from tensorflow.keras.models import Model
import tensorflow as tf


def conv3x3(x, filters: int, name: str, strides: int = 1):
    return Conv2D(
        filters=filters,
        kernel_size=(3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
        name=name
    )(x)


def residual_block(
    x,
    filters: int,
    name: str,
    downsample: list = None,
    strides: int = 1,
):
    identity = x

    x = conv3x3(x, filters=filters, strides=strides, name=f"{name}_conv1")
    x = BatchNormalization(momentum=0.99, epsilon=1e-5, name=f"{name}_bn1")(x)
    x = ReLU(name=f"{name}_relu1")(x)

    x = conv3x3(x, filters=filters, name=f"{name}_conv2")
    x = BatchNormalization(momentum=0.99, epsilon=1e-5, name=f"{name}_bn2")(x)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    x = Add(name=f"{name}_add")([identity, x])
    x = ReLU(name=f"{name}_out")(x)

    return x


def make_layers(
    x,
    filters: int,
    num_blocks: int,
    name: str,
    strides: int = 1
):
    downsample = None

    if strides != 1 or filters != x.shape[3]:
        downsample = [
            Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, use_bias=False, name=f"{name}.1_conv2d"),
            BatchNormalization(momentum=0.99, epsilon=1e-5, name=f"{name}.1_bn")
        ]

    x = residual_block(x, filters=filters, strides=strides, downsample=downsample, name=f"{name}.1")
    for index in range(1, num_blocks):
        x = residual_block(x, filters=filters, name=f"{name}.{index + 1}")

    return x


class ResNet18(Model):

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

        x = ZeroPadding2D(padding=3, name="stem_block_padding1")(self.inputs)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, use_bias=False, name="stem_block_conv2d")(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-5, name="stem_block_bn")(x)
        x = ReLU(name="stem_block_relu")(x)

        x = ZeroPadding2D(padding=1, name="stem_block_padding2")(x)
        output_block_1 = MaxPool2D(pool_size=(3, 3), strides=2, name="stem_block_maxpool")(x)

        output_block_2 = make_layers(output_block_1,  filters=64, num_blocks=2, strides=1, name="block2")
        output_block_3 = make_layers(output_block_2, filters=128, num_blocks=2, strides=2, name="block3")
        output_block_4 = make_layers(output_block_3, filters=256, num_blocks=2, strides=2, name="block4")
        output_block_5 = make_layers(output_block_4, filters=512, num_blocks=2, strides=2, name="block5")

        self.outputs = output_block_5
        super(ResNet18, self).__init__(inputs=self.inputs, outputs=self.outputs, name="ResNet18")


class ResNet34(Model):

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

        x = ZeroPadding2D(padding=3, name="stem_block_padding1")(self.inputs)
        x = Conv2D(filters=64, kernel_size=(7, 7), strides=2, use_bias=False, name="stem_block_conv2d")(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-5, name="stem_block_bn")(x)
        x = ReLU(name="stem_block_relu")(x)

        x = ZeroPadding2D(padding=1, name="stem_block_padding2")(x)
        output_block_1 = MaxPool2D(pool_size=(3, 3), strides=2, name="stem_block_maxpool")(x)

        output_block_2 = make_layers(output_block_1,  filters=64, num_blocks=3, strides=1, name="block2")
        output_block_3 = make_layers(output_block_2, filters=128, num_blocks=4, strides=2, name="block3")
        output_block_4 = make_layers(output_block_3, filters=256, num_blocks=6, strides=2, name="block4")
        output_block_5 = make_layers(output_block_4, filters=512, num_blocks=3, strides=2, name="block5")

        self.outputs = output_block_5

        super(ResNet34, self).__init__(inputs=self.inputs, outputs=self.outputs, name="ResNet34")


class ResNet50(Model):

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

        model = tf.keras.applications.resnet.ResNet50(
            include_top=False,
            weights=weights,
            input_tensor=self.inputs
        )
        self.outputs = model.outputs

        super(ResNet50, self).__init__(inputs=self.inputs, outputs=self.outputs, name="ResNet50")
