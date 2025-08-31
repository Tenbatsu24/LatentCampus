# Consistent View Alignment
## Consistency-Based SSL for 3D Medical Imaging

![Schema](./assets/CVA/Schema.png)

This repository builds on [OpenMind (nnSSL)](link-to-original-repo) and extends it with novel architectures, training strategies, and losses for the **Self-Supervised Learning for 3D Medical Imaging Challenge**.

We (team **The\_Latent\_Campus**) achieved **🥇 1st place in Track Primus-M** and **🥈 2nd place in Track ResEnc-L**, demonstrating the effectiveness of our consistency-based approach.

---

## 🔧 Contributions

Compared to the original repository, we introduce the following key components:

### **Architectures**

* `nnssl/architectures/consis_arch.py` adds:

  * **ResEnc-L** and **Primus-M** with projector–predictor mechanisms.
  * Ability to extract both **volumetric feature maps** and **global pooled feature maps**.

### **Training**
* `nnssl/training/nnsslTrainer/aligned_mae`
  * Implemented a **symmetrized loss** that combines:

    * Contrastive learning
    * Masked autoencoders
    * Our proposed **consistent view alignment** strategy.

* Integrated into the trainer interface, consistent with the repo’s design.

### **Loss**
* `nnssl/training/loss/aligned_mae_loss.py`
  * A **composite loss** that unifies multiple objectives.
  * An **alignment utility** for 3D bounding box–based volumetric alignment.
  * A **reworked NT-Xent loss** capable of handling symmetrization.

---

## 🏆 Challenge Overview

The **Self-Supervised Learning for 3D Medical Imaging Challenge** provides a unified benchmark for evaluating SSL methods in medical imaging.
It addresses fragmentation in the field by standardizing:

* Pre-training datasets
* Model architectures
* Fine-tuning schedules
* Evaluation setups

This ensures fair and reproducible comparisons across SSL approaches.

We participated as **The\_Latent\_Campus** and ranked:

* **1st in Track Primus-M**
* **2nd in Track ResEnc-L**

---

## 📊 Our Testing and Ablations 

![Challenge Results](./assets/CVA/Results.png)

## 📊 Challenge Results
Please find the final challenge results at the link:
* https://ssl3d-challenge.dkfz.de/leaderboard
* We are The_Latent_Campus
  * ResEnc-L Track: Position 2nd
  * Primus-M Track: Position 1st

---

## Usage

Pre-process and format the data as you would for OpenMind as usual. Train the models as below.


```bash
# ResEnc-L
nnssl_train 745 onemmiso -tr AlignedMAEFTTrainer -p nnsslPlans -num_gpus 1 -pretrained_weights ${nnssl_results}/Dataset745_OpenMind/MAETrainer/fold_all/checkpoint_final.pth || true

#Primus-M
nnssl_train 745 onemmiso -tr AlignedMAEFTLR3EvaTrainer -p nnsslPlans -num_gpus 1 -pretrained_weights ${nnssl_results}/Dataset745_OpenMind/MAETrainer/fold_all/checkpoint_final.pth || true
```

----

## 📦 Pretrained Weights

Pretrained weights will be released soon on **Zenodo**. Stay tuned!

---

## 📖 Citation

Citation to our challenge report and paper.

Please also cite the original work this repo builds on:

```bibtex
coming soon!
```

---

## ⚖ License

This repository is released under the [Creative Commons Attribution-NonCommercial 4.0 International Public License](./LICENSE.md).

### Requirements

All requirements are the same as in the original repository, including dependencies for PyTorch, einops, thop, and other libraries.
