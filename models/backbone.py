from models.darknet import *
from models.resnet import *
from models.vgg import *

from tensorflow.keras.models import Model
import tensorflow as tf
import os


class Backbone(object):

    def __init__(
        self,
        name: str,
        inputs: tf.keras.layers,
        weights_root_path: str = None
    ):
        if name == "vgg16":
            self.base_layers = VGG16(inputs=inputs, weights=None).outputs
            self.network = Model(
                inputs=inputs,
                outputs=self.base_layers,
                name="vgg16"
            )
        elif name == "vgg19":
            self.base_layers = VGG19(inputs=inputs, weights=None).outputs
            self.network = Model(
                inputs=inputs,
                outputs=self.base_layers,
                name="vgg19"
            )
        elif name == "resnet18":
            self.base_layers = ResNet18(inputs=inputs, weights=None).outputs
            self.network = Model(
                inputs=inputs,
                outputs=self.base_layers,
                name="resnet18"
            )
        elif name == "resnet34":
            self.base_layers = ResNet34(inputs=inputs, weights=None).outputs
            self.network = Model(
                inputs=inputs,
                outputs=self.base_layers,
                name="resnet18"
            )
        elif name == "resnet50":
            self.base_layers = ResNet50(inputs=inputs, weights=None).outputs
            self.network = Model(
                inputs=inputs,
                outputs=self.base_layers,
                name="resnet50"
            )
        elif name == "darknet19":
            self.base_layers = Darknet19(inputs=inputs, weights=None).outputs
            self.network = Model(
                inputs=inputs,
                outputs=self.base_layers,
                name="resnet50"
            )
        elif name == "darknet53":
            self.base_layers = Darknet53(inputs=inputs, weights=None).outputs
            self.network = Model(
                inputs=inputs,
                outputs=self.base_layers,
                name="resnet50"
            )
        else:
            raise ValueError("\'{}\' no such backbone exists.".format(name))

        if weights_root_path is not None:
            if os.path.isdir(weights_root_path):
                self.load_weights(weights_root_path)
            else:
                ValueError("\'{}\' no such file or directory.".format(weights_root_path))

    def load_weights(self, path: str):
        path = os.path.join(path, "")
        self.network.load_weights(path)

    def save_weights(self, path):
        path = os.path.join(path, "")
        self.network.save_weights(path)

    def freeze(self):
        for layer in self.network.layers:
            layer.trainable = False

    def unfreeze(self):
        for layer in self.network.layers:
            layer.trainable = True

    def summary(self):
        self.network.summary()
