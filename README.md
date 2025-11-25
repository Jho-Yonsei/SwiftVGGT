<p align="center">
<h1 align="center">
  SwiftVGGT: A Scalable Visual Geometry Grounded Transformer for Large-Scale Scenes
  <br />
</h1>
  <p align="center">
    <a href="https://Jho-Yonsei.github.io/">Jungho Lee </a>&nbsp;·&nbsp;
    <a href="https://hydragon.co.kr">Minhyeok Lee</a>&nbsp;·&nbsp;
    Sunghun Yang &nbsp;·&nbsp;
    Minseok Kang &nbsp;·&nbsp;
    <a href="http://mvp.yonsei.ac.kr/">Sangyoun Lee</a>&nbsp;&nbsp;
  </p>
  <p align="center">
    Yonsei University
  </p>
  <p align="center">
    <a href="https://Jho-Yonsei.github.io/SwiftVGGT"><img src="https://img.shields.io/badge/SwiftVGGT-ProjectPage-blue.svg"></a>
    <a href="https://arxiv.org/abs/2511.18290"><img src="https://img.shields.io/badge/SwiftVGGT-arXiv-red.svg"></a>
  </p>
  <div align="center"></div>
</p>
</p>

## 🔭 Introduction

<p align="center">
  <img src="./assets/images/teaser.png" alt="Teaser">
</p>

<p align="justify">
  <strong>Abstract:</strong> 3D reconstruction in large-scale scenes is a fundamental task in 3D perception, but the inherent trade-off between accuracy and computational efficiency remains a sigificant challenge. Existing methods either prioritize speed and produce low-quality results, or achieve high-quality reconstruction at the cost of slow inference times. In this paper, we propose SwiftVGGT, a training-free method that significantly reduce inference time while preserving high-quality dense 3D reconstruction. To maintain global consistency in large-scale scenes, SwiftVGGT performs loop closure without relying on the external Visual Place Recognition (VPR) model. This removes redundant computation and enables accurate reconstruction over kilometer-scale environments. Furthermore, we propose a simple yet effective point sampling method to align neighboring chunks using a single Sim(3)-based Singular Value Decomposition (SVD) step. This eliminates the need for the Iteratively Reweighted Least Squares (IRLS) optimization commonly used in prior work, leading to substantial speed-ups. We evaluate SwiftVGGT on multiple datasets and show that it achieves state-of-the-art reconstruction quality while requiring only 33% of the inference time of recent VGGT-based large-scale reconstruction approaches.
</p>

## 🆕 News
- 2025-11-25: [[arXiv paper]](https://arxiv.org/abs/2511.18290) is available.
- 2025-11-21: Code, [[project page]](https://Jho-Yonsei.github.io/SwiftVGGT/) are available.

## 🔧 Installation
Clone the repository and create an anaconda environment using.

```
git clone https://github.com/Jho-Yonsei/SwiftVGGT.git
cd SwiftVGGT

conda create -y -n swiftvggt python=3.10.18
conda activate swiftvggt

pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

python setup.py install
```

Then, download the VGGT checkpoint.

```
mkdir -p ckpt
wget https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
mv model_tracker_fixed_e20.pt ckpt
```


## 🔦 Inference for Custom Data
Put the ```--image_dir``` to your image path and run following command:
```
CUDA_VISIBLE_DEVICES=<GPU> python run.py --image_dir <image_path> --output_path <output_path> --save_points
```

If ```Out of Memory``` occurs, then set smaller ```--chunk_size``` and ```--overlap_size```.


## 🔦 Inference & Evaluation

For inference and evaluation of KITTI odometry dataset, just add ```--gt_pose_path``` as follows:
```
CUDA_VISIBLE_DEVICES=<GPU> python run.py --image_dir <image_path> --gt_pose_path <gt_pose_path> --output_path <output_path>
```

## 🌟 Acknowledgements
Our repository is built upon [VGGT](https://github.com/facebookresearch/vggt), [VGGT-Long](https://github.com/DengKaiCQ/VGGT-Long), and [FastVGGT](https://github.com/mystorm16/FastVGGT). We thank to all the authors for their awesome works.

## 📚 BibTex