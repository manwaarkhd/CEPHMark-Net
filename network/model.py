from network.landmark_refinement_network import LandmarkRefinementNetwork
from network.landmark_detection_network import LandmarkDetectionNetwork
from network.semantic_fusion_block import SemanticFusionBlock
from easydict import EasyDict as edict
from network.pooling import ROIAlign2D
from models import Backbone
import tensorflow as tf
from config import cfg


class Network(object):

    def __init__(
        self,
        backbone_name: str,
        freeze_backbone: bool = False,
        backbone_weights: str = None,
    ):
        # Backbone feature extractor
        self.backbone = Backbone(name=backbone_name, inputs=cfg.IMAGE_INPUT, weights_root_path=backbone_weights)
        if freeze_backbone:
            self.backbone.freeze()
        C3 = self.backbone.network.get_layer(cfg.BACKBONE_BLOCKS_INFO[backbone_name]["C3"]).output
        C4 = self.backbone.network.get_layer(cfg.BACKBONE_BLOCKS_INFO[backbone_name]["C4"]).output
        C5 = self.backbone.network.get_layer(cfg.BACKBONE_BLOCKS_INFO[backbone_name]["C5"]).output

        # Landmark detection module
        detection_network = LandmarkDetectionNetwork(inputs=self.backbone.network.output)
        self.landmark_detection_module = tf.keras.models.Model(
            inputs=cfg.IMAGE_INPUT,
            outputs=detection_network.outputs,
            name="landmark_detection_module"
        )

        # Semantic fusion block
        self.fusion_block = SemanticFusionBlock(num_filters=256, name="semantic_fusion_block")
        P3, P4, P5 = self.fusion_block(inputs=[C3, C4, C5])

        # Craniofacial feature extraction & rescaling
        region_proposal_map = ROIAlign2D(
            crop_size=cfg.ROI_POOL_SIZE,
            name="craniofacial_feature_extraction"
        )([
            (P3, P4, P5),
            cfg.PROPOSALS_INPUT]
        )

        # Landmark refinement module
        refinement_network = LandmarkRefinementNetwork(inputs=region_proposal_map)
        self.landmark_refinement_module = edict()
        self.landmark_refinement_module.heads = []
        for index in range(len(refinement_network.outputs)):
            head = tf.keras.models.Model(
                inputs=[cfg.IMAGE_INPUT, cfg.PROPOSALS_INPUT],
                outputs=refinement_network.outputs[index],
                name="landmark_refinement_head" + str(index + 1)
            )
            self.landmark_refinement_module.heads.append(head)

        # Overall framework
        self.model = tf.keras.models.Model(
            inputs=[cfg.IMAGE_INPUT, cfg.PROPOSALS_INPUT],
            outputs=[
                detection_network.outputs,
                refinement_network.outputs
            ]
        )