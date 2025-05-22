# AdapTok: Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space

<p align="center">
<img src="assets/vis.png" width=95%>
<p>

## Overview

We propose AdapTok, an adaptive temporal causal video tokenizer that can flexibly allocate tokens for different frames based on video content. AdapTok is equipped with a block-wise masking strategy that randomly drops tail tokens of each block during training, and a block causal scorer to predict the reconstruction quality of video frames using different numbers of tokens. During inference, an adaptive token allocation strategy based on integer linear programming is further proposed to adjust token usage given predicted scores. Such design allows for sample-wise, content-aware, and temporally dynamic token allocation under a controllable overall budget. Extensive experiments for video reconstruction and generation on UCF-101 and Kinetics-600 demonstrate the effectiveness of our approach. Without additional image data, AdapTok consistently improves reconstruction quality and generation performance under different token budgets, allowing for more scalable and token-efficient generative video modeling.

<p align="center">
<img src="assets/framework.png" width=95%>
<p>


## Get Started

1. Install pytorch 2.5.1 and torchvision 0.20.1
   ```
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

2. Install other dependencies
   ```
   pip install -r requirements.txt
   ```

## Training

```
bash scripts/train_adaptok.sh
bash scripts/train_adaptok_scorer.sh
bash scripts/train_adaptok_ar.sh
bash scripts/train_adaptok_ar_fp.sh
```

## Evaluation

```
bash scripts/eval_adaptok.sh
bash scripts/eval_adaptok_ar.sh
bash scripts/eval_adaptok_ar_fp.sh
```

## Citation

If you find this code useful in your research, please consider citing:
```
@article{adaptok,
    title={Learning Adaptive and Temporally Causal Video Tokenization in a 1D Latent Space},
}
```