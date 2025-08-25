# Orthogonal Position Representations for Transformer in Neural Machine Translation


## Installation

* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.6
* For training new models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)


## Prepare Training Data

1. Download the preprocessed [WMT'16 En-De dataset](https://drive.google.com/uc?export=download&id=0B_bZck-ksdkpM25jRUN2X2UxMm8) provided by Google to project root dir

2. Generate binary dataset at `data-bin/wmt16_en_de_google`

> `bash prepare-wmt-en2de.sh`

## Train

### Train opr transformer model

> `bash train-wmt-en2e-opr.sh`

### Train opr+rpr transformer model

> `bash train-wmt-en2e-opr-rpr.sh`

NOTE: BLEU will be calculated automatically when finishing training
