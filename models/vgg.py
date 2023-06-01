from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import tensorflow as tf


class VGG16(Model):

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

        model = tf.keras.applications.vgg16.VGG16(
            include_top=False,
            input_tensor=inputs,
            weights=weights
        )
        self.outputs = model.outputs

        super(VGG16, self).__init__(inputs=self.inputs, outputs=self.outputs, name="VGG16")


class VGG19(Model):

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

        model = tf.keras.applications.vgg19.VGG19(
            include_top=False,
            input_tensor=inputs,
            weights=weights
        )
        self.outputs = model.outputs

        super(VGG19, self).__init__(inputs=self.inputs, outputs=self.outputs, name="VGG19")
