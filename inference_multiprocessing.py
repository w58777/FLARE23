# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import torch
from skimage import measure
import torch.nn.functional as F
import nibabel as nib
import time

from nnunetv2.paths import nnUNet_results
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data_cascade import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO


def get_args():
    parser = argparse.ArgumentParser(description="Inference tumor")
    parser.add_argument("--gpu", type=int, default="0", metavar="N", help="input visible devices for training (default: 0)",)
    parser.add_argument("-i", type=str, default="/home/data/input/", metavar="N", help="input path")
    parser.add_argument("-o", type=str, default="/home/data/output/", metavar="N", help="output path")
    return parser.parse_args()


def crop_by_organ(image, mask, organ): # crop box by organ segmentation
    arr = np.nonzero(organ)
    minA = max(0, min(arr[0]) - 5)
    maxA = min(len(organ), max(arr[0]) + 5)

    MARGIN = 20
    minB = max(0, min(arr[1]) - MARGIN)
    maxB = min(organ.shape[1], max(arr[1]) + MARGIN)
    minC = max(0, min(arr[2]) - MARGIN)
    maxC = min(organ.shape[2], max(arr[2]) + MARGIN)

    if (maxA - minA) % 8 != 0:
        max_A = 8 * (int((maxA - minA) / 8) + 1)
        gap = int((max_A - (maxA - minA)) / 2)
        minA = max(0, minA - gap)
        maxA = min(len(organ), minA + max_A)
        if maxA == len(organ):
            minA = maxA - max_A

    if (maxB - minB) % 8 != 0:
        max_B = 8 * (int((maxB - minB) / 8) + 1)
        gap = int((max_B - (maxB - minB)) / 2)
        minB = max(0, minB - gap)
        maxB = min(organ.shape[1], minB + max_B)
        if maxB == organ.shape[1]:
            minB = maxB - max_B

    if (maxC - minC) % 8 != 0:
        max_C = 8 * (int((maxC - minC) / 8) + 1)
        gap = int((max_C - (maxC - minC)) / 2)
        minC = max(0, minC - gap)
        maxC = min(organ.shape[2], minC + max_C)
        if maxC == organ.shape[2]:
            minC = maxC - max_C


    image = image[minA:maxA, minB:maxB, minC:maxC]
    mask = mask[minA:maxA, minB:maxB, minC:maxC]

    bbox = [minA, maxA, minB, maxB, minC, maxC]

    return image, mask, bbox


def crop_by_tumor(mask, tumor): # crop data by tumor label's minimum bounding box
    
    arr = np.nonzero(tumor)
    minA, maxA, minB, maxB, minC, maxC = min(arr[0]), max(arr[0]), min(arr[1]), max(arr[1]), min(arr[2]), max(arr[2]) 

    mask = mask[minA:maxA, minB:maxB, minC:maxC]
    bbox = [minA, maxA, minB, maxB, minC, maxC]

    return mask, bbox


def inference(args):

    start_time = time.time()

    # instantiate the nnUNetPredictor
    predictor_liver_tumor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False, # False to disable tta
        perform_everything_on_gpu=True,
        device=torch.device('cuda:{}'.format(args.gpu)),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    predictor_liver_tumor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Load_Your_Model_Weight_Path'),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor_pancreas_tumor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_gpu=True,
        device=torch.device('cuda:{}'.format(args.gpu)),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    predictor_pancreas_tumor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Load_Your_Model_Weight_Path'),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )

    predictor_organ = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_gpu=True,
        device=torch.device('cuda:{}'.format(args.gpu)),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    predictor_organ.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Load_Your_Model_Weight_Path'),
        use_folds=(0,),
        checkpoint_name='checkpoint_best.pth',
    )
    
    print('load:',time.time()-start_time)

    for item in os.walk(args.i):
        for i in range(len(item[2])):

            # read_image
            print('predicting:', item[2][i])
            tmp_image = nib.load(os.path.join(args.i, item[2][i]))
            spacing = tmp_image.header.get_zooms()
            props = {'spacing':[spacing[2],spacing[1],spacing[0]]}
            props_double = {'spacing':[spacing[2]*2,spacing[1],spacing[0]]}
            print(props)

            # get affine and transpose
            affine = tmp_image.affine
            tmp_image = tmp_image.get_fdata().transpose(2, 1, 0)
            tmp_image = np.expand_dims(tmp_image,axis=0)
            print('original shape of image:',tmp_image.shape)
            print('read:',time.time()-start_time)

            # compute length
            z_len = spacing[2] * tmp_image.shape[1]

            if z_len < 600 or tmp_image.shape[1] < 250: # only abdominal
                print('directly predict')
                if tmp_image.shape[1] < 200: # directly predict
                    tmp_label = predictor_organ.predict_single_npy_array(tmp_image, props, None, None, False)
                    tmp_image = np.squeeze(tmp_image)
                    tmp_image = np.squeeze(tmp_image)
                else: # abdominal but narrow spacing
                    if tmp_image.shape[1] % 2 == 0:
                        tmp_image_downsampling = tmp_image[:,::2,:,:]
                        internal_label = predictor_organ.predict_single_npy_array(tmp_image_downsampling, props_double, None, None, False)
                        tmp_label = F.interpolate(torch.from_numpy(internal_label).to(torch.float32).unsqueeze(0).unsqueeze(0), size=(internal_label.shape[0]*2, internal_label.shape[1], internal_label.shape[2]), mode='nearest-exact')
                        tmp_image = np.squeeze(tmp_image)
                        tmp_image = np.squeeze(tmp_image)
                        tmp_label = np.squeeze(tmp_label)
                    else: 
                        tmp_image_downsampling = tmp_image[:,1:tmp_image.shape[1],:,:]
                        tmp_image_downsampling = tmp_image_downsampling[:,::2,:,:]
                        internal_label = predictor_organ.predict_single_npy_array(tmp_image_downsampling, props_double, None, None, False)
                        internal_label = F.interpolate(torch.from_numpy(internal_label).to(torch.float32).unsqueeze(0).unsqueeze(0), size=(internal_label.shape[0]*2, internal_label.shape[1], internal_label.shape[2]), mode='nearest-exact')
                        tmp_image = np.squeeze(tmp_image)
                        tmp_image = np.squeeze(tmp_image)
                        internal_label = np.squeeze(internal_label)
                        tmp_label = np.zeros_like(tmp_image)
                        tmp_label[1:tmp_image.shape[0],:,:] = internal_label
            elif z_len >= 600: # abdominal + lung
                print('crop first')
                tmp_image_crop = tmp_image[:,int(tmp_image.shape[1]*0.25):int(tmp_image.shape[1]*0.75),:,:]
                if tmp_image.shape[1]*0.5 < 200:
                    internal_label = predictor_organ.predict_single_npy_array(tmp_image_crop, props, None, None, False)
                    tmp_image = np.squeeze(tmp_image)
                    tmp_image = np.squeeze(tmp_image)
                    internal_label = np.squeeze(internal_label)
                    tmp_label = np.zeros_like(tmp_image)
                    tmp_label[int(tmp_image.shape[0]*0.25):int(tmp_image.shape[0]*0.75),:,:] = internal_label   
                else: 
                    if tmp_image_crop.shape[1] % 2 == 0:
                        tmp_image_crop_downsampling = tmp_image_crop[:,::2,:,:]
                        internal_label = predictor_organ.predict_single_npy_array(tmp_image_crop_downsampling, props_double, None, None, False)
                        tmp_label_crop = F.interpolate(torch.from_numpy(internal_label).to(torch.float32).unsqueeze(0).unsqueeze(0), size=(internal_label.shape[0]*2, internal_label.shape[1], internal_label.shape[2]), mode='nearest-exact')
                        tmp_image = np.squeeze(tmp_image)
                        tmp_image = np.squeeze(tmp_image)
                        tmp_label_crop = np.squeeze(tmp_label_crop)
                        tmp_label = np.zeros_like(tmp_image)
                        tmp_label[int(tmp_image.shape[0]*0.25):int(tmp_image.shape[0]*0.75),:,:] = tmp_label_crop
                    else:  
                        tmp_image_crop_downsampling = tmp_image_crop[:,1:tmp_image_crop.shape[1],:,:]
                        tmp_image_crop_downsampling = tmp_image_crop_downsampling[:,::2,:,:]
                        internal_label = predictor_organ.predict_single_npy_array(tmp_image_crop_downsampling, props_double, None, None, False)
                        internal_label = F.interpolate(torch.from_numpy(internal_label).to(torch.float32).unsqueeze(0).unsqueeze(0), size=(internal_label.shape[0]*2, internal_label.shape[1], internal_label.shape[2]), mode='nearest-exact')
                        tmp_image = np.squeeze(tmp_image)
                        tmp_image = np.squeeze(tmp_image)
                        internal_label = np.squeeze(internal_label)
                        tmp_label_crop = np.zeros_like(tmp_image_crop)
                        tmp_label_crop = np.squeeze(tmp_label_crop)
                        tmp_label_crop[1:tmp_image.shape[0],:,:] = internal_label
                        tmp_label = np.zeros_like(tmp_image)
                        tmp_label[int(tmp_image.shape[0]*0.25):int(tmp_image.shape[0]*0.75),:,:] = tmp_label_crop

    
            if isinstance(tmp_label, torch.Tensor):
                tmp_label = tmp_label.cpu().numpy()
            pred = np.zeros_like(tmp_label)
            print('organ',time.time()-start_time)
            
            # enhance or plainCT
            Aorta = tmp_label.copy()
            Aorta[Aorta != 5] = 0
            arr_boundary = np.nonzero(Aorta)
            upper_bound = max(arr_boundary[0])
            lower_bound = min(arr_boundary[0])-20
            Aorta_image = tmp_image * Aorta
            aorta_flatten = Aorta_image[arr_boundary]
            aorta_median = np.median(aorta_flatten)
            if aorta_median >= 89 * 5: # enhance
                tumor_list = [13, 2, 4, 1]
                predictor_kidney_tumor = nnUNetPredictor(
                    tile_step_size=0.5,
                    use_gaussian=True,
                    use_mirroring=False,
                    perform_everything_on_gpu=True,
                    device=torch.device('cuda:{}'.format(args.gpu)),
                    verbose=False,
                    verbose_preprocessing=False,
                    allow_tqdm=False
                )
                predictor_kidney_tumor.initialize_from_trained_model_folder(
                    join(nnUNet_results, 'Load_Your_Model_Weight_Path'),
                    use_folds=(0,),
                    checkpoint_name='checkpoint_best.pth',
                )
            else:
                tumor_list = [4, 1]


            for j in tumor_list: 

                if time.time() - start_time > 50: break # time limited

                organ_j = tmp_label.copy()
                organ_j[organ_j != j] = 0
                if np.max(organ_j) == 0: continue
                image_crop, organ_j_crop, bbox = crop_by_organ(tmp_image, organ_j, organ_j)
                
                # predict a single numpy array
                image_crop = np.expand_dims(image_crop, axis=0)
                if j == 2 or j == 13: predictor = predictor_kidney_tumor
                elif j == 4: predictor = predictor_pancreas_tumor
                elif j == 1: predictor = predictor_liver_tumor
                pred_3D = np.zeros_like(tmp_label)
                ret = predictor.predict_single_npy_array(image_crop, props, None, None, False)

                if ret.sum() < 128: continue # filt tiny result that likely to be mis segment

                label_connect, num_connect = measure.label(ret, connectivity=3, background=0, return_num=True)

                for k in range(1, num_connect + 1):
                    tmp_label_connect = label_connect.copy()
                    tmp_label_connect[tmp_label_connect != k] = 0
                    if (tmp_label_connect * organ_j_crop / j / k).sum() < 32: 
                        label_connect[label_connect == k] = 0
                label_connect[label_connect > 1] = 1

                union_sum = (organ_j_crop * label_connect).sum()
                if union_sum/j < 80: continue
                
                pred_3D[bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]] = label_connect
                pred = pred_3D * 14 + pred

        
            if np.max(pred) == 0: pred = tmp_label
            else: 
                pred = pred + tmp_label
                pred[pred > 14] = 14


            final_pred = np.zeros_like(tmp_label)
            final_pred[max(lower_bound, 0): upper_bound, :, :] = pred[max(lower_bound, 0): upper_bound, :, :]

            nib.Nifti1Image(final_pred.transpose(2, 1, 0), affine).to_filename(os.path.join(args.o, "{}{}".format(item[2][i][0:-12], item[2][i][-7:])))

            print('done:',time.time()-start_time)
            start_time = time.time()


if __name__ == '__main__':

    args = get_args()

    inference(args)


