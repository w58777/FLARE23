# Solution of Team mihi_segment for FLARE23 Challenge

This repository provides the solution of team mihi_segment for MICCAI FLARE23 Challenge. Please refer to https://codalab.lisn.upsaclay.fr/competitions/12239 to get more information about the challenge.

## 0 Environments and Requirements

Our work is based on nnUNetv2, so you should meet the requirements of nnUNetv2 and install it. You can install it as below. For more details, please refer to https://github.com/MIC-DKFZ/nnUNet.

```python
git clone https://github.com/w58777/FLARE23.git
cd FLARE23
pip install -e .
```

## 1 Preprocessing

1. We use pseudo organ label provide by best solution of FLARE22, so first you should transfer the pseudo organ labels to the unlabeled organ labels. 
2. As we segment tumor in the fine stage, organ that contains tumor should be crop for training.
3. For pseudo tumor label, we also crop organ that don't contains tumor for generating pseudo tumor label.
4. All step above can be done in /dataset/preprocessing.ipynb.

## 2 Training Organ Segmentation Network

### 2.1 Prepare dataset

Following nnUNetv2, give a DatasetID (e.g. Dataset005) to the 4000 data with label and organize them folowing the requirement of nnUNetv2.

```
nnUNetFrame/DATASET/nnUNet_raw/Dataset005_Organ/
├── dataset.json
├── imagesTr
├── imagesTs
└── labelsTr
```

### 2.2 Conduct automatic preprocessing using nnUNet

Here we do not use the default setting.

```sh
nnUNetv2_plan_and_preprocess -d 5 -c 3d_fullres --verify_dataset_integrity
```

### 2.3 Training organ segmentation network

```sh
for FOLD in 0 1 2 3 4
do
nnUNetv2_train 5 3d_fullres FOLD
done
```

## 3 Training Tumor Segmentation Network

### 3.1 Train tumor segmentation network without pseudo label

As mentioned in 1, we use cropped data to train tumor segmentation network without pseudo label. Dataset should be organized as 2.1 and preprocessing as 2.2, then training as 2.3. For example:
```
nnUNetFrame/DATASET/nnUNet_raw/Dataset001_LiverTumor/
├── dataset.json
├── imagesTr
├── imagesTs
└── labelsTr
```
```sh
nnUNetv2_plan_and_preprocess -d 1 -c 3d_fullres --verify_dataset_integrity
```
```sh
for FOLD in 0 1 2 3 4
do
nnUNetv2_train 1 3d_fullres FOLD
done
```

### 3.2 Generate pseudo tumor labels for cropped data

```sh
nnUNetv2_predict -i INPUTS_FOLDER -o OUTPUTS_FOLDER -d 1 -c 3d_fullres -f all
```

### 3.3 Filter Unsatisfied Pseudo Labels

```sh
python filter_unsatisfed_pseudo_label.py
```

### 3.4 Train tumor segmentation network with pseudo label

The steps are the same as in 3.1, with the addition of pseudo-labeled data.

## 4 Inference

We employ end-to-end inference and utilize multiprocessing techniques for acceleration.

```sh
python inference_multiprocessing.py
```



