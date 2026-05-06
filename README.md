[README.md](https://github.com/user-attachments/files/27420651/README.md)
# SimCLR & Image Classification with Knowledge Distillation

A collection of TensorFlow 2 notebooks exploring contrastive representation learning (SimCLR), transfer learning, and knowledge distillation on standard image classification benchmarks.

---

## Overview

This project contains two complementary tracks:

**Track 1 — SimCLRv2 (TF2 Colabs)**  
Work with Google's pretrained SimCLRv2 models via TensorFlow Hub. Load checkpoints, run inference on ImageNet/tf-flowers, fine-tune on downstream tasks, and distill large teacher models into smaller student networks using self-training (no labels required).

**Track 2 — CIFAR Experiments**  
Train and evaluate image classifiers on CIFAR-10 and CIFAR-100 using transfer learning (ResNet50, MobileNetV2) and custom CNN architectures, with knowledge distillation experiments for model compression.

---

## Notebooks

### SimCLR (TF2)

| Notebook | Description |
|---|---|
| `TF2_load_and_inference.ipynb` | Load SimCLRv2 checkpoints or Hub modules, inspect parameter counts, and run inference on sample images |
| `TF2_finetuning.ipynb` | Fine-tune a pretrained SimCLRv2 model on a downstream classification task (tf-flowers) |
| `TF2_distillation_self_training.ipynb` | Distill/self-train a SimCLRv2 teacher model into a smaller student without needing ground-truth labels |

### CIFAR Experiments

| Notebook | Description |
|---|---|
| `cifar10__1_.ipynb` | Transfer learning on CIFAR-10 using a frozen ResNet50 encoder + dense classifier head |
| `cifar10_distillation_.ipynb` | Train a CNN teacher on CIFAR-10, then distill it into a smaller student CNN using KL-divergence loss |
| `cifar100.ipynb` | Transfer learning on CIFAR-100 using a frozen MobileNetV2 encoder + dense classifier head |
| `cifar100_distillation.ipynb` | Teacher–student distillation on CIFAR-100 with temperature scaling and a mixed CE + KL loss |

---

## Key Concepts

**SimCLRv2** is Google's self-supervised contrastive learning framework. Pretrained models are available at multiple scales (ResNet-50 through ResNet-152 2×+SK) and fine-tuning levels (1%, 10%, 100% of ImageNet labels).

**Knowledge Distillation** compresses a large, accurate *teacher* model into a smaller, faster *student* model. The student learns from soft probability distributions (logits) produced by the teacher rather than hard labels, guided by two losses:
- **Cross-entropy loss** against the true labels
- **KL-divergence loss** against the teacher's softened outputs (controlled by a temperature parameter)

The balance between them is set by α (alpha).

**Transfer Learning** notebooks freeze a pretrained ImageNet backbone and train only a lightweight classification head on top, making training feasible on limited hardware.

---

## Requirements

```
tensorflow >= 2.x
tensorflow-hub
tensorflow-datasets
numpy
matplotlib
```

All SimCLR notebooks are designed to run on **Google Colab** (Google Cloud authentication is required to access SimCLRv2 checkpoints from GCS). The CIFAR notebooks can run locally or on Colab.

---

## SimCLRv2 Checkpoints

Pretrained checkpoints are hosted on Google Cloud Storage:

| Variant | Path |
|---|---|
| Pretrained (linear eval) | `gs://simclr-checkpoints-tf2/simclrv2/pretrained/` |
| Fine-tuned on 1% labels | `gs://simclr-checkpoints-tf2/simclrv2/finetuned_1pct/` |
| Fine-tuned on 10% labels | `gs://simclr-checkpoints-tf2/simclrv2/finetuned_10pct/` |
| Fine-tuned on 100% labels | `gs://simclr-checkpoints-tf2/simclrv2/finetuned_100pct/` |

Example — load ResNet-152 (2×+SK): `gs://simclr-checkpoints-tf2/simclrv2/pretrained/r152_2x_sk1`

---

## License

Notebooks are derived from the [Google SimCLR repository](https://github.com/google-research/simclr) and are licensed under the **Apache License 2.0**.
