# A Two-stage Regression Framework for Automated Cephalometric Landmark Detection Incorporating Semantically Fused Anatomical Features and Multi-head Refinement Loss
<p align="justify">
Accurate identification and precise localization of cephalometric landmarks provides clinicians with essential insights into craniofacial deformities, aiding in the assessment of treatment strategies for improved patient outcomes. The current methodologies heavily depend on the utilization of multiple CNNs for predicting landmark coordinates, which makes them computationally burdensome and unsuitable for translation to clinical applications. To overcome this limitation, we propose a novel two-stage end-to-end trainable regression framework for cephalometric landmark detection. In the initial stage, a single neural network is employed to estimate the locations of all landmarks simultaneously, enabling the identification of potential landmark regions. In the second stage, a semantic fusion block leverages the in-network multi-resolution feature hierarchy to produce high-level semantically rich features. These feature maps are then cropped based on coarsely detected landmark locations and concurrent refinement loss is used to fine-tuned and refine the landmark locations. The proposed framework demonstrates potential for enhancing clinical workflow and treatment outcomes in orthodontics. This is achieved through the utilization of a single CNN backbone augmented with multi-resolution semantically fused anatomical features, which effectively enhances representation learning in a computationally efficient manner. The performance of proposed framework is evaluated on two publicly available anatomical landmark data-sets. The experimental results demonstrate that our framework achieves a state-of-the-art detection accuracy of <strong>87.17%</strong> within the clinically accepted range of 2mm.

  The salient features of our proposed framework are summarized as follows:
  <ul>
    <li> A single, end-to-end trainable, multi-head CNN architecture with a backbone feature extractor. This design allows for the reuse of multi-resolution feature maps from different layers of the backbone network, computed during the forward pass, without additional computational cost.
    <li> Joint learning of both modules that enables the framework to leverage both global hard/soft tissue characteristics and geometric landmark relations in a unified manner.
    <li> A semantic fusion block that effectively leverages feature maps from multiple blocks of the backbone network to generate semantically rich features. Subsequently, the refinement module extracts anatomically relevant features to further refine the initially coarsely predicted landmark locations, with the help of a multi-head refinement loss.
  </ul>
</p>

<div align="center">
  <img src="docs/framework-schematic-diagram.png">
</div>
<div align="center"> Schematic representation of the proposed framework for landmark detection in cephalometric images. </div>
