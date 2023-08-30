import os
from tqdm import tqdm
import numpy as np
import nibabel as nib
from skimage import measure

raw_pseudo_label_path = "/home/data/raw_pseudo_label_path/"

filt_label_path = "/home/data/filt_label_path/"

organ_pseudo_label_path = "/home/data/organ_pseudo_label_path/"


def crop_by_tumor(mask, tumor):
    # crop data by tumor label's minimum bounding box
    arr = np.nonzero(tumor)
    minA, maxA, minB, maxB, minC, maxC = min(arr[0]), max(arr[0]), min(arr[1]), max(arr[1]), min(arr[2]), max(arr[2]) 

    mask = mask[minA:maxA, minB:maxB, minC:maxC]
    bbox = [minA, maxA, minB, maxB, minC, maxC]

    return mask, bbox

# Liver
for item in os.walk(raw_pseudo_label_path):
    for i in tqdm(range(len(item[2]))):
        # get the original index of file
        digit_filter = filter(str.isdigit, item[2][i])
        file_index = "".join(list(digit_filter))

        if item[2][i][-1] != 'z': continue # exclude json file
        tmp_tumor_pseudo_label = nib.load(os.path.join(raw_pseudo_label_path, item[2][i]).format(i)).get_fdata()
        if tmp_tumor_pseudo_label.sum() < 2048: continue # filt tiny result that likely to be mis segment

        # exclude pseudo tumor labels that are organ indeed
        tmp_organ_pseudo_label = nib.load(os.path.join(organ_pseudo_label_path, item[2][i]).format(i)).get_fdata().astype(np.int16)
        tmp_organ_pseudo_label_crop, tmp_organ_pseudo_label_crop_bbox = crop_by_tumor(tmp_organ_pseudo_label, tmp_tumor_pseudo_label)
        counts = np.bincount(tmp_organ_pseudo_label_crop.flatten())
        counts[0] = 0
        if np.argmax(counts) != 1: continue # not liver, tumor label is label of other organ

        os.system('cp {}{} {}{}'.format(raw_pseudo_label_path, item[2][i], filt_label_path, item[2][i]))
