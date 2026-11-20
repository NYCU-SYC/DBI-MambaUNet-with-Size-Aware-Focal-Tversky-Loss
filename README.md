# DBI-MambaUNet with Size-Aware Focal Tversky Loss

DBI-MambaUNet segments brain tumors on paired contrast-enhanced T1-weighted and T2-weighted MRI. The model takes two co-registered 3D volumes and returns a tumor mask and voxel-wise probabilities on the input image grid.

**[Download the pretrained model](https://github.com/NYCU-SYC/DBI-MambaUNet-with-Size-Aware-Focal-Tversky-Loss/releases/latest/download/DBI-MambaUNet-model.zip)** · [Model details and SHA-256](https://github.com/NYCU-SYC/DBI-MambaUNet-with-Size-Aware-Focal-Tversky-Loss/releases/latest) · [Training instructions](configs/README.md)

## Installation

The study environment used Ubuntu 20.04, Python 3.10, PyTorch 2.0.1, CUDA 11.8 and an NVIDIA Tesla V100. From the repository root, install the project and compatible CUDA builds of causal-conv1d and Mamba.

```bash
conda create -n dbi-mambaunet python=3.10 -y
conda activate dbi-mambaunet
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install "causal-conv1d>=1.2.0" mamba-ssm --no-cache-dir
pip install -e .
```

## Inference

Unzip the model archive. It contains the fold 0 weights, `plans.json` and `dataset.json` in a ready-to-use model folder. Put the co-registered examinations in an input directory, using the same case identifier for both channels. The study used ANTs for co-registration and omitted skull stripping.

```text
INPUT_FOLDER/
  CASE_0000.nii.gz   # contrast-enhanced T1-weighted MRI
  CASE_0001.nii.gz   # T2-weighted MRI
```

```bash
unzip DBI-MambaUNet-model.zip
nnUNetv2_predict_from_modelfolder -i INPUT_FOLDER -o OUTPUT_FOLDER \
  -m DBI-MambaUNet-model -f 0 -chk checkpoint_best.pth --save_probabilities
```

The pipeline applies per-volume z-score normalization and resamples both channels to **1.5 × 0.5 × 0.5 mm** in depth, height and width. It averages the logits from the original and width-flipped inputs after reversing the flip. The output `.nii.gz` contains the more probable class at each voxel. The compressed `.npz` stores both probability channels, with tumor at index 1.

## Network and loss

The model uses **64 × 224 × 224** patches and has **81,662,618** parameters. The encoder downsamples along all three axes at the second stage and in-plane at the final stage. Its feature-map dimensions in (channels, depth, height, width) order are

```text
(64,64,224,224) → (128,32,112,112) → (256,16,56,56)
→ (512,8,28,28) → (768,4,14,14) → (768,4,7,7)
```

The SA-FTL trainer uses 26-connected tumor components within each patch. Its Tversky coefficients are α = 0.3 and β = 0.7, with a direct focal exponent of 1.33, size exponent of 0.8 and smoothing of 10⁻⁶. Positive voxel weights are normalized within the patch. Background voxels receive weight 1.1 in patches with tumor and 2.1 in all-background patches.

| Implementation | Trainer |
| --- | --- |
| [Size-Aware Focal Tversky](DBI-MambaUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerDBI-MambaUNet_SizeAwareFocalTversky_Loss.py) | `nnUNetTrainerDBIMambaUNet_SizeAwareFocalTversky_Loss` |
| [CE + Dice baseline](DBI-MambaUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainerDBI-MambaUNet_Dice_CE_Loss.py) | `nnUNetTrainerDBIMambaUNet_Dice_CE_Loss` |

[3D network](DBI-MambaUNet/nnunetv2/nets/DBI-MambaUNet_3d.py) · [Trainer implementations](DBI-MambaUNet/nnunetv2/training/nnUNetTrainer)

## Development data

| Stage | Training MRI examinations | Validation MRI examinations | Total |
| --- | ---: | ---: | ---: |
| [Pretraining](configs/pretraining/dataset.json) | 3,944 | 968 | 4,912 |
| [Fine-tuning](configs/fine-tuning/dataset.json) | 4,205 | 1,108 | 5,313 |

Each stage’s `numTraining` counts all MRI examinations in its development partition. The fixed patient-level split assigns the examinations used for gradient updates and validation.

DBI-MambaUNet builds on [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), [U-Mamba](https://github.com/bowang-lab/U-Mamba) and [Mamba](https://github.com/state-spaces/mamba). Please cite the study and the upstream methods when using this work. The source code is available under the [Apache 2.0 license](LICENSE).
