<h2 align="center">A Two-stage Regression Framework for Automated Cephalometric Landmark Detection Incorporating Semantically Fused Anatomical Features and Multi-head Refinement Loss</h2>
<p align="center">
  <a href="https://scholar.google.com/citations?hl=en&user=eTZ3L4QAAAAJ"><strong> Muhammad Anwaar Khalid </strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=n1GKwfkAAAAJ&hl=en"><strong> Atif Khurshid </strong></a>
  ·
  <a href="https://scholar.google.com/citations?hl=en&user=1yHShlwAAAAJ"><strong> Kanwal Zulfiqar </strong></a>
  ·
  <a href="https://scholar.google.com/citations?hl=en&user=e2w698UAAAAJ"><strong> Ulfat Bashir </strong></a>
  ·
  <a href="https://scholar.google.com/citations?user=tpfgan0AAAAJ&hl=en"><strong> Muhammad Moazam Fraz </strong></a>
  <br>
</p>
<p align="justify">
This repository contains the implementation of <b>CEPHMark-Net</b>, a novel, end-to-end trainable two-stage regression framework for accurate cephalometric landmark detection. This work has been published in <a href="https://www.sciencedirect.com/science/article/abs/pii/S095741742401707X"><i>Expert Systems with Applications</i></a>. The framework aims to streamline the detection process and improve clinical workflows by providing precise localization of landmarks with reduced computational overhead.
</p>

<div align="center">
  <img src="docs/framework-schematic-diagram.png">
</div>
<br>
<div align="center"> Schematic of the proposed cephalometric landmark detection framework, featuring a <b>landmark detection module (LDM)</b>, <b>semantic fusion block (SFB)</b>, and <b>landmark refinement module (LRM)</b>. </div>
<br>

<h2 align="left">Overview</h2>
<p align="justify">
Accurate identification and precise localization of cephalometric landmarks provides clinicians with essential insights into craniofacial deformities, aiding in the assessment of treatment strategies for improved patient outcomes. The current methodologies heavily depend on the utilization of multiple CNNs for predicting landmark coordinates, which makes them computationally burdensome and unsuitable for translation to clinical applications. To overcome this limitation, we propose a novel two-stage end-to-end trainable regression framework for cephalometric landmark detection. In the initial stage, a single neural network is employed to estimate the locations of all landmarks simultaneously, enabling the identification of potential landmark regions. In the second stage, a semantic fusion block leverages the in-network multi-resolution feature hierarchy to produce high-level semantically rich features. These feature maps are then cropped based on coarsely detected landmark locations and concurrent refinement loss is used to fine-tuned and refine the landmark locations. The proposed framework demonstrates potential for enhancing clinical workflow and treatment outcomes in orthodontics. This is achieved through the utilization of a single CNN backbone augmented with multi-resolution semantically fused anatomical features, which effectively enhances representation learning in a computationally efficient manner. The performance of proposed framework is evaluated on two publicly available anatomical landmark data-sets. The experimental results demonstrate that our framework achieves a state-of-the-art detection accuracy of <strong>87.17%</strong> within the clinically accepted range of 2mm.

  The salient features of our proposed framework are summarized as follows:
  <ul>
    <li> A single, end-to-end trainable, multi-head CNN architecture with a backbone feature extractor. This design allows for the reuse of multi-resolution feature maps from different layers of the backbone network, computed during the forward pass, without additional computational cost.
    <li> Joint learning of both modules that enables the framework to leverage both global hard/soft tissue characteristics and geometric landmark relations in a unified manner.
    <li> A semantic fusion block that effectively leverages feature maps from multiple blocks of the backbone network to generate semantically rich features. Subsequently, the refinement module extracts anatomically relevant features to further refine the initially coarsely predicted landmark locations, with the help of a multi-head refinement loss.
  </ul>
</p>

## Installatoin
### Prerequisites
- Python 3.8
- TensorFlow 2.10
- OpenCV 4.9
### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/manwaarkhd/CEPHMark-Net.git
   cd CEPHMark-Net
   ```
3. Set up a Python virtual environment (optional but recommended):
   ```bash
   python3 -m venv pyenv
   source pyenv/bin/activate  # On Windows use `env\Scripts\activate`
   ```
5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset
We utilized the publicly available [ISBI 2015 Dataset](https://ieeexplore.ieee.org/iel7/42/4359023/07061486.pdf?casa_token=Wv4hnXcVbf4AAAAA:eIQCBU1Y_6s0H9s1WXZk-c30fQQq-B7_nz-ADloTje8WqKzfZPE_7TXpaCxSob5L0CqG9F8rpkvk) by Wang et al. (2015), which consists of 400 high-resolution X-ray images. Each image has spatial dimensions of 1935 × 2400 pixels, with a spatial resolution of 0.1 mm/pixel in both directions. We employed the same 150 images for training as used in the ISBI Grand Challenge 2015. The remaining 250 images are reserved for evaluation and further partitioned into two distinct subsets: Test1 and Test2. Test1 serves as our validation set for assessing the accuracy of our method during the development phase, and Test2 is used as our test set for the final evaluation of our proposed method. <br>
After downloading the files, please create a folder named `datasets` and organize it as follows:
```
datasets/
└── ISBI Dataset/
    ├── Dataset/
    │   ├── Training/
    │   │   ├── 001.bmp
    │   │   ├── 002.bmp
    │   │   └── ...
    │   └── Testing/
    │       ├── Test1/
    │       │   ├── 151.bmp
    │       │   ├── 152.bmp
    │       │   └── ...
    │       └── Test2/
    │           ├── 301.bmp
    │           ├── 302.bmp
    │           └── ...
    └── Annotations/
        ├── Junior Orthodontist/
        │   ├── 001.txt
        │   ├── 002.txt
        │   ├── ...
        │   └── 400.txt
        └── Senior Orthodontist/
            ├── 001.txt
            ├── 002.txt
            ├── ...
            └── 400.txt
```

## Usage
### Preparing the Data
1. Download the dataset from [link](https://figshare.com/s/37ec464af8e81ae6ebbf).
2. Organize the dataset as shown in the Datset section below.
3. Ensure that annotation files are placed correctly within the `Annotations` directory.

### Configuration:
The configuration of the CEPHMark-Net framework is managed through the `config.py` file. This file contains various settings, hyper-parameters, and heuristics that can be adjusted to fine-tune the model's performance and adapt it to different datasets. To customize the configuration for your specific use case, edit the `config.py` file accordingly. Here are some common modifications you might consider:
- **Adjusting Image Dimensions:** If your dataset has different image dimensions, modify **`config.ORIGINAL_HEIGHT`**, **`config.ORIGINAL_WIDTH`**.
- **Region of Interest (ROI) Pooling:** Adjust the size of the pooling region used for extracting features from the detected regions by **`config.ROI_POOL_SIZE`**.
- **Updating Training Parameters:** Modify **`config.TRAIN.EPOCHS`** and **`config.TRAIN.OPTIMIZER`** to set the number of training epochs and choose a different optimizer or learning rate.

### Training
To train the model on `Train` dataset, use:
```bash
python train.py
```
### Validation
To run inference on `Test1` dataset using pre-trained weights:
```bash
python valid.py
```
### Testing
To evaluate the model on the `Test2` dataset:
```bash
python test.py
```

## Results
For a comprehensive analysis of the results, including quantitative metrics such as mean squared error (MSE) and landmark detection accuracy, as well as qualitative comparisons with traditional methods and other state-of-the-art approaches, please refer to our [paper](https://www.sciencedirect.com/science/article/abs/pii/S095741742401707X). The paper provides detailed tables, charts, and visualizations illustrating the performance improvements and validation of our method.

## Citation
If you find our work useful in your research, please consider citing our paper:
```BibTeX
@article{khalid2024two,
  title={A two-stage regression framework for automated cephalometric landmark detection incorporating semantically fused anatomical features and multi-head refinement loss},
  author={Khalid, Muhammad Anwaar and Khurshid, Atif and Zulfiqar, Kanwal and Bashir, Ulfat and Fraz, Muhammad Moazam},
  journal={Expert Systems with Applications},
  pages={124840},
  year={2024},
  publisher={Elsevier}
}
```

