# DBI-MambaUNet with Size-Aware Focal Tversky Loss

DBI-MambaUNet is an nnU-Net v2 based framework for 3D biomedical image segmentation. The project adds a DBI-MambaUNet network under `DBI-MambaUNet/nnunetv2/nets/DBI-MambaUNet_3d.py` and custom nnU-Net trainers for Dice, Tversky, Focal-Tversky, Generalized-Dice, and Size-Aware Focal Tversky losses.

This repository is built on top of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet), and the Mamba installation flow follows [U-Mamba](https://github.com/bowang-lab/U-Mamba), which depends on [Mamba](https://github.com/state-spaces/mamba).

## Installation

Recommended environment, following U-Mamba:

- Ubuntu 20.04 or WSL2/Linux
- CUDA 11.8
- Python 3.10
- PyTorch 2.0.1

```bash
conda create -n dbi-mambaunet python=3.10 -y
conda activate dbi-mambaunet

pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install packaging
pip install "causal-conv1d>=1.2.0"
pip install mamba-ssm --no-cache-dir

pip install -e .
```

Sanity check:

```bash
python - <<'PY'
import torch
import mamba_ssm
print(torch.__version__)
PY
```

Mamba kernels are sensitive to CUDA, PyTorch, and compiler versions. If installation fails, first verify that your CUDA runtime and PyTorch wheel match.

## Data Layout

DBI-MambaUNet uses the standard nnU-Net v2 dataset format:

```text
data/
  nnUNet_raw/
    DatasetXXX_TaskName/
      imagesTr/
        case_0001_0000.nii.gz
      labelsTr/
        case_0001.nii.gz
      imagesTs/
      dataset.json
  nnUNet_preprocessed/
  nnUNet_results/
```

By default, the code uses `<repo>/data/nnUNet_raw`, `<repo>/data/nnUNet_preprocessed`, and `<repo>/data/nnUNet_results`. To use another location, set the nnU-Net environment variables before running commands:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

PowerShell:

```powershell
$env:nnUNet_raw = "D:\path\to\nnUNet_raw"
$env:nnUNet_preprocessed = "D:\path\to\nnUNet_preprocessed"
$env:nnUNet_results = "D:\path\to\nnUNet_results"
```

## Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

Replace `DATASET_ID` with the numeric id of your nnU-Net dataset, for example `701`.

## Training

Default DBI-MambaUNet trainer with Size-Aware Focal Tversky + CE loss:

```bash
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerDBI-MambaUNet_SizeAwareTversky_CE_Loss
```

Example:

```bash
nnUNetv2_train 701 3d_fullres 0 -tr nnUNetTrainerDBI-MambaUNet_SizeAwareTversky_CE_Loss
```

Available DBI-MambaUNet trainers:

- `nnUNetTrainerDBI-MambaUNet_Dice_Loss`
- `nnUNetTrainerDBI-MambaUNet_Tversky_Loss`
- `nnUNetTrainerDBI-MambaUNet_Tversky_CE_Loss`
- `nnUNetTrainerDBI-MambaUNet_FocalTversky_Loss`
- `nnUNetTrainerDBI-MambaUNet_FocalTversky_CE_Loss`
- `nnUNetTrainerDBI-MambaUNet_GeneralizeDice_Loss`
- `nnUNetTrainerDBI-MambaUNet_GeneralizeDice_CE_Loss`
- `nnUNetTrainerDBI-MambaUNet_SizeAwareTversky_Loss`
- `nnUNetTrainerDBI-MambaUNet_SizeAwareTversky_CE_Loss`

## Inference

```bash
nnUNetv2_predict \
  -i INPUT_FOLDER \
  -o OUTPUT_FOLDER \
  -d DATASET_ID \
  -c 3d_fullres \
  -f FOLD \
  -tr nnUNetTrainerDBI-MambaUNet_SizeAwareTversky_CE_Loss \
  --disable_tta
```

## Notes

- The main network implementation is `DBI-MambaUNet/nnunetv2/nets/DBI-MambaUNet_3d.py`.
- Size-aware trainers use `SizeAwareTverskyLoss`, where `gamma` is mapped to the focal term (`focal_gamma`) for Size-Aware Focal Tversky behavior.
- Trained weights, datasets, preprocessing outputs, and large medical image files are intentionally ignored by git.
- This repository does not vendor the Mamba source tree. Install `causal-conv1d` and `mamba-ssm` in the environment instead.
- If AMP produces NaNs in Mamba blocks, try disabling AMP in the trainer, as also noted by U-Mamba.

## Citation

If you use this repository, please cite the DBI-MambaUNet paper once the citation is available. Please also cite the upstream projects that this work builds on:

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet)
- [U-Mamba](https://github.com/bowang-lab/U-Mamba)
- [Mamba](https://github.com/state-spaces/mamba)

## Acknowledgements

We thank the authors of nnU-Net, U-Mamba, and Mamba for releasing their code and documentation to the community.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE).
