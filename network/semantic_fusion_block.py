from tensorflow.keras.models import Model
import tensorflow as tf


class SemanticFusionBlock(Model):

    def __init__(
        self,
        num_filters: int,
        name: str,
        **kwargs
    ):
        super(SemanticFusionBlock, self).__init__(name=name, **kwargs)

        self.block3_lateral_conv2d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), activation="relu", padding="same", name="block3_lateral_conv2d")
        self.block4_lateral_conv2d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), activation="relu", padding="same", name="block4_lateral_conv2d")
        self.block5_lateral_conv2d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(1, 1), activation="relu", padding="same", name="block5_lateral_conv2d")

        self.block5_lateral_upsampling = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="block5_lateral_upsampling")
        self.block4_lateral_upsampling = tf.keras.layers.UpSampling2D(size=2, interpolation="nearest", name="block4_lateral_upsampling")

        self.block5_block4_fusion = tf.keras.layers.Add(name="block5_block4_add")
        self.block4_block3_fusion = tf.keras.layers.Add(name="block4_block3_add")

        self.block3_posthoc_conv2d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), activation="relu", padding="same", name="block3_posthoc_conv2d")
        self.block4_posthoc_conv2d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), activation="relu", padding="same", name="block4_posthoc_conv2d")
        self.block5_posthoc_conv2d = tf.keras.layers.Conv2D(filters=num_filters, kernel_size=(3, 3), activation="relu", padding="same", name="block5_posthoc_conv2d")

        self.block3_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name="block3_batchnorm")
        self.block4_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name="block4_batchnorm")
        self.block5_batchnorm = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name="block5_batchnorm")

    def call(self, inputs, training=True, **kwargs):
        C3, C4, C5 = inputs

        block5_lateral_output = self.block5_lateral_conv2d(C5)
        block5_feature_output = self.block5_posthoc_conv2d(block5_lateral_output)  # P5
        P5 = self.block5_batchnorm(block5_feature_output)

        block4_lateral_output = self.block4_lateral_conv2d(C4)
        feat_a = self.block5_lateral_upsampling(block5_lateral_output)
        feat_b = block4_lateral_output
        block4_fusion_output = self.block5_block4_fusion([feat_a, feat_b])
        block4_feature_output = self.block4_posthoc_conv2d(block4_fusion_output)   # P4
        P4 = self.block4_batchnorm(block4_feature_output)

        block3_lateral_output = self.block3_lateral_conv2d(C3)
        feat_a = self.block4_lateral_upsampling(block4_fusion_output)
        feat_b = block3_lateral_output
        block3_fusion_output = self.block4_block3_fusion([feat_a, feat_b])
        block3_feature_output = self.block3_posthoc_conv2d(block3_fusion_output)    # P3
        P3 = self.block3_batchnorm(block3_feature_output)

        return P3, P4, P5