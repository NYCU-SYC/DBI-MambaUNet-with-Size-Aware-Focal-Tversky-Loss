# Training DBI-MambaUNet

Training proceeds in two stages. Each stage has its own raw data, fixed split, preprocessing cache and results directory. The dataset IDs 811 and 812 are local recipe identifiers.

| Stage | Directory | Train / validation examinations | Initial learning rate |
| --- | --- | ---: | ---: |
| Pretraining | `Dataset811_DBIPretraining` | 3,944 / 968 | 0.01 |
| Fine-tuning | `Dataset812_DBIFineTuning` | 4,205 / 1,108 | 0.0005 |

## Prepare the two stages

Run from the repository root. Set `nnUNet_raw`, `nnUNet_preprocessed` and `nnUNet_results` to your data directories. Under each raw dataset directory place all development examinations for that stage in `imagesTr` (T1W+C `_0000`, T2W `_0001`) and one integer label map per examination in `labelsTr` (background 0, tumor 1). Co-register the two channel volumes before preprocessing.

Place the original patient-level allocation for each stage at `$nnUNet_preprocessed/DATASET_NAME/splits_final.json`. Fold 0 must have the nnU-Net form `[{"train": [...], "val": [...]}]` with case identifiers and no filename suffixes. The trainer checks examination counts, overlap and coverage. Patient identity and separation across stages must be checked against your private patient manifest. The public package contains configuration files only.

```bash
prepare_stage() (
  set -eu
  stage=$1
  dataset=$2
  id=$3
  dest="$nnUNet_preprocessed/$dataset"
  mkdir -p "$dest"
  cp "configs/$stage/dataset.json" "$nnUNet_raw/$dataset/dataset.json"
  cp "configs/$stage/dataset.json" "$dest/dataset.json"
  cp "configs/$stage/plans.json" "$dest/DBIPlans.json"
  nnUNetv2_extract_fingerprint -d "$id" --verify_dataset_integrity &&
    nnUNetv2_preprocess -d "$id" -plans_name DBIPlans -c 3d_lowres -np 1
)
prepare_stage pretraining Dataset811_DBIPretraining 811
prepare_stage fine-tuning Dataset812_DBIFineTuning 812
```

The fixed plans specify the 3D network configuration. Fingerprint extraction records statistics for your stage data. Each channel uses its own image mean and standard deviation before spatial resampling, so population intensity statistics are unnecessary. Re-running preprocessing rebuilds the stage's `DBIPlans_3d_lowres` cache.

## Train and fine-tune

```bash
trainer=nnUNetTrainerDBIMambaUNet_SizeAwareFocalTversky_Loss
nnUNetv2_train 811 3d_lowres 0 -tr "$trainer" -p DBIPlans
pretrained="$nnUNet_results/Dataset811_DBIPretraining/${trainer}__DBIPlans__3d_lowres/fold_0/checkpoint_best.pth"
nnUNetv2_train 812 3d_lowres 0 -tr "$trainer" -p DBIPlans -pretrained_weights "$pretrained"
```

Both stages use 500 epochs, 250 iterations per epoch, batch size 1, Nesterov SGD (momentum 0.99, weight decay 3 × 10⁻⁵), polynomial decay (power 0.9), and deep supervision. Selection uses the highest exponential moving average of online validation foreground Dice. In this implementation `-pretrained_weights` transfers matching network weights except the decoder segmentation heads. Those heads retain their fresh initialization. Fine-tuning starts a new optimizer and scheduler and keeps all layers trainable.

Training uses spatial flips across three axes for augmentation. Inference averages the original and width-flipped predictions. The model download includes the [fine-tuning plans](fine-tuning/plans.json) and [dataset](fine-tuning/dataset.json). See the [inference command](../README.md#inference).
