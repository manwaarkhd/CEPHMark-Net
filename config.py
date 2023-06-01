from tensorflow.keras.layers import Input
from easydict import EasyDict as edict
import tensorflow as tf

config = edict()

# original height and width of image
config.ORIGINAL_HEIGHT = 2400
config.ORIGINAL_WIDTH = 1935

# height and width to resize image
config.HEIGHT = 320
config.WIDTH  = 256

# input cephalogram image to the base network
config.IMAGE_INPUT = Input(shape=(config.HEIGHT, config.WIDTH, 3), name="cephalogram")

# landmark region proposals (LRPs) input to the landmark detection network
config.PROPOSALS_INPUT = Input(shape=(None, 4), name="landmark_region_proposals")

# Image resolution (mm/pixel)
config.IMAGE_RESOLUTION = 0.1

# cephalometric landmarks
config.ANATOMICAL_LANDMARKS = {
    "0": "Sella",
    "1": "Nasion",
    "2": "Orbitale",
    "3": "Porion",
    "4": "A-point",
    "5": "B-point",
    "6": "Pogonion",
    "7": "Menton",
    "8": "Gnathion",
    "9": "Gonion",
    "10": "Lower Incisal Incision",
    "11": "Upper Incisal Incision",
    "12": "Upper Lip",
    "13": "Lower Lip",
    "14": "Subnasale",
    "15": "Soft Tissue Pogonion",
    "16": "Posterior Nasal Spine",
    "17": "Anterior Nasal Spine",
    "18": "Articulare",
}

# number of cephalometric landmarks
config.NUM_LANDMARKS = 19


config.BACKBONE_BLOCKS_INFO = {
    "vgg16": {
        "C1": "block1_conv2",
        "C2": "block2_conv2",
        "C3": "block3_conv3",
        "C4": "block4_conv3",
        "C5": "block5_conv3"
    },
    "vgg19": {
        "C1": "block1_conv2",
        "C2": "block2_conv2",
        "C3": "block3_conv4",
        "C4": "block4_conv4",
        "C5": "block5_conv4"
    },
    "darknet19": {
        "C1": "block1_conv1",
        "C2": "block2_conv1",
        "C3": "block3_conv3",
        "C4": "block4_conv3",
        "C5": "block5_conv5",
        "C6": "block6_conv5",
    },
    "darknet53": {
        "C1": "block1.1_out",
        "C2": "block2.2_out",
        "C3": "block3.8_out",
        "C4": "block4.8_out",
        "C5": "block5.4_out"
    },
    "resnet18": {
        "C2": "block2.2_out",
        "C3": "block3.2_out",
        "C4": "block4.2_out",
        "C5": "block5.2_out"
    },
    "resnet34": {
        "C2": "block2.3_out",
        "C3": "block3.4_out",
        "C4": "block4.6_out",
        "C5": "block5.3_out"
    },
    "resnet50": {
        "C2": "conv2_block3_out",
        "C3": "conv3_block4_out",
        "C4": "conv4_block6_out",
        "C5": "conv5_block3_out"
    }
}

# Region of interest pool size
config.ROI_POOL_SIZE = (5, 5)

# margin (in pixels) at each side of lateral skull face
config.BOX_MARGIN = 32

config.TRAIN = edict()
# number of epochs
config.TRAIN.EPOCHS = 10
# optimizer
config.TRAIN.OPTIMIZER = tf.keras.optimizers.Adam(learning_rate=0.0001)

cfg = config
