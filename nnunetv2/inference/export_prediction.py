import os
from typing import Union
import numpy as np
import torch

from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice

from nnunetv2.configuration import default_num_processes
from nnunetv2.utilities.label_handling.label_handling import LabelManager
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
import torch.nn.functional as F

def convert_predicted_logits_to_segmentation_with_correct_shape(predicted_logits: Union[torch.Tensor, np.ndarray, str],
                                                                plans_manager: PlansManager,
                                                                configuration_manager: ConfigurationManager,
                                                                label_manager: LabelManager,
                                                                properties_dict: dict,
                                                                return_probabilities: bool = False,
                                                                num_threads_torch: int = default_num_processes):

    old_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads_torch)

    delfile = ''
    if isinstance(predicted_logits, str):
        delfile = predicted_logits
        predicted_logits = torch.from_numpy(np.load(predicted_logits))
        os.remove(delfile)
    
    gpu_device = torch.device("cuda:0")  # 选择第一个 GPU 设备

    predicted_logits = predicted_logits.to(torch.float32)
    print('tensor before upsampling:', predicted_logits.size())
    predicted_logits = torch.unsqueeze(predicted_logits, 0)
    new_size = properties_dict.get('shape_after_cropping_and_before_resampling')
    # print('label size:', new_size)
    predicted_logits = torch.unsqueeze(predicted_logits, 0)
    predicted_logits_new = torch.zeros(1, predicted_logits.shape[1], predicted_logits.shape[2], new_size[0], new_size[1], new_size[2])
    
    if np.prod(new_size) < 20447232:
        for i in range(predicted_logits.shape[2]):
            tmp = predicted_logits[:,:,i,:,:,:].to(gpu_device)
            tmp = F.interpolate(tmp, size=new_size, mode='trilinear', antialias=False)
            predicted_logits_new[:,:,i,:,:,:] = tmp.cpu()

        predicted_logits_new = torch.squeeze(predicted_logits_new, 0)
        predicted_logits_new = torch.squeeze(predicted_logits_new, 0).to(gpu_device)
    else: 
        for i in range(predicted_logits.shape[2]):
            predicted_logits_new[:,:,i,:,:,:] = F.interpolate(predicted_logits[:,:,i,:,:,:], size=new_size, mode='trilinear', antialias=False)

        predicted_logits_new = torch.squeeze(predicted_logits_new, 0)
        predicted_logits_new = torch.squeeze(predicted_logits_new, 0)

    print('tensor after upsampling:', predicted_logits_new.size())
    # predicted_logits = predicted_logits_new
    # predicted_logits = F.interpolate(predicted_logits, size=new_size, mode='nearest-exact', antialias=False)
    # predicted_logits = torch.squeeze(predicted_logits, 0)
    # print('tensor after upsampling:', predicted_logits.size())

    predicted_probabilities = label_manager.apply_inference_nonlin(predicted_logits_new)
    del predicted_logits
    segmentation = label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

    if isinstance(segmentation, torch.Tensor):
        segmentation = segmentation.cpu().numpy()

    segmentation_reverted_cropping = np.zeros(properties_dict['shape_before_cropping'],
                                              dtype=np.uint8 if len(label_manager.foreground_labels) < 255 else np.uint16)

    slicer = bounding_box_to_slice(properties_dict['bbox_used_for_cropping'])
    segmentation_reverted_cropping[slicer] = segmentation
    del segmentation
    # revert transpose
    segmentation_reverted_cropping = segmentation_reverted_cropping.transpose(plans_manager.transpose_backward)
    if return_probabilities: pass
    else:
        torch.set_num_threads(old_threads)
        if not (delfile == ''):
            np.save(delfile, segmentation_reverted_cropping)
            seg_path = delfile
            return seg_path
        else:
            return segmentation_reverted_cropping
