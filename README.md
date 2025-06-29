## Latest updates

* Pre-trained model and demos are now available! 
* Full code - coming soon!

# Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models [CVPR'25]

<div>
  <img src="docs/salf-cbm-gif-v2.gif" style="width:100%"> 
</div>
<br>

<a href="https://itaybenou.github.io/show-and-tell/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=red" height=20.5></a> 
<a href="https://arxiv.org/abs/2502.20134"><img src="https://img.shields.io/badge/arXiv-2502.20134-b31b1b.svg" height=20.5></a>

> **Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models**
>
> Itay Benou, Tammy Riklin-Raviv
>
> Abstract: Modern deep neural networks have now reached human-level performance across a variety of tasks. However, unlike humans they lack the ability to explain their decisions by showing where and telling what concepts guided them. In this work, we present a unified framework for transforming any vision neural network into a spatially and conceptually interpretable model. We introduce a spatially-aware concept bottleneck layer that projects “black-box” features of pre-trained backbone models into interpretable concept maps, without requiring human labels. By training a classification layer over this bottleneck, we obtain a self-explaining model that articulates which concepts most influenced its prediction, along with heatmaps that ground them in the input image. Accordingly, we name this method “Spatially-Aware and Label-Free Concept Bottleneck Model” (SALF-CBM). Our results show that the proposed SALF-CBM: (1) Outperforms non-spatial CBM methods, as well as the original backbone, on a variety of classification tasks; (2) Produces high-quality spatial explanations, outperforming widely used heatmap-based methods on a zero-shot segmentation task; (3) Facilitates model exploration and debugging, enabling users to query specific image regions and refine the model's decisions by locally editing its concept maps.

## Installation

1. Clone the repository.
2. Install Python (3.9) and PyTorch (1.13).
3. Run `pip install -r requirements.txt`.
4. Download pretrained model by running  `bash download_model.sh`.
5. Download the [ViT-B SAM model checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it under `segment_anything/checkpoint` folder.

## Run Demos

Follow the instructions in the following notebooks:
1. Explaining model predictions: `demos/model_explainability.ipynb`.
2. Explain Anything feature: `demos/explain_anything.ipynb`.
3. Model debugging feature: `demos/model_debugging.ipynb`.

## Cite this work
Itay Benou and Tammy Riklin Raviv, [*Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models*](https://openaccess.thecvf.com/content/CVPR2025/papers/Benou_Show_and_Tell_Visually_Explainable_Deep_Neural_Nets_via_Spatially-Aware_CVPR_2025_paper.pdf), CVPR 2025.

```
@inproceedings{benou2025show,
  title={Show and Tell: Visually Explainable Deep Neural Nets via Spatially-Aware Concept Bottleneck Models},
  author={Benou, Itay and Raviv, Tammy Riklin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={30063--30072},
  year={2025}
}
```