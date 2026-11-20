# MRI data layout

Each case uses two co-registered NIfTI volumes and a binary tumor mask. Use the same case identifier in all three filenames.

```text
Dataset812_DBIFineTuning/
  dataset.json
  imagesTr/
    CASE_0000.nii.gz   # contrast-enhanced T1-weighted MRI
    CASE_0001.nii.gz   # T2-weighted MRI
  labelsTr/
    CASE.nii.gz        # 0 background, 1 tumor
```

Use the [pretraining](../configs/pretraining/dataset.json) and [fine-tuning](../configs/fine-tuning/dataset.json) dataset files for the corresponding development partitions. The [training instructions](../configs/README.md) describe the expected patient-level split and directory names. The folders under `data/nnUNet_raw` are upstream nnU-Net layout examples.
