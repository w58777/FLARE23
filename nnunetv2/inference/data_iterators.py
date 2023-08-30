from typing import Union, List

import numpy as np
import torch
from batchgenerators.dataloading.data_loader import DataLoader

from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager

class PreprocessAdapterFromNpy(DataLoader):
    def __init__(self, list_of_images: List[np.ndarray],
                 list_of_segs_from_prev_stage: Union[List[np.ndarray], None],
                 list_of_image_properties: List[dict],
                 truncated_ofnames: Union[List[str], None],
                 plans_manager: PlansManager, dataset_json: dict, configuration_manager: ConfigurationManager,
                 num_threads_in_multithreaded: int = 1, verbose: bool = False):
        preprocessor = configuration_manager.preprocessor_class(verbose=verbose)
        self.preprocessor, self.plans_manager, self.configuration_manager, self.dataset_json, self.truncated_ofnames = \
            preprocessor, plans_manager, configuration_manager, dataset_json, truncated_ofnames

        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if list_of_segs_from_prev_stage is None:
            list_of_segs_from_prev_stage = [None] * len(list_of_images)
        if truncated_ofnames is None:
            truncated_ofnames = [None] * len(list_of_images)

        super().__init__(
            list(zip(list_of_images, list_of_segs_from_prev_stage, list_of_image_properties, truncated_ofnames)),
            1, num_threads_in_multithreaded,
            seed_for_shuffle=1, return_incomplete=True,
            shuffle=False, infinite=False, sampling_probabilities=None)

        self.indices = list(range(len(list_of_images)))

    def generate_train_batch(self):
        idx = self.get_indices()[0]
        image = self._data[idx][0]
        seg_prev_stage = self._data[idx][1]
        props = self._data[idx][2]
        ofname = self._data[idx][3]
        # if we have a segmentation from the previous stage we have to process it together with the images so that we
        # can crop it appropriately (if needed). Otherwise it would just be resized to the shape of the data after
        # preprocessing and then there might be misalignments
        data, seg = self.preprocessor.run_case_npy(image, seg_prev_stage, props,
                                                   self.plans_manager,
                                                   self.configuration_manager,
                                                   self.dataset_json)
        if seg_prev_stage is not None:
            seg_onehot = convert_labelmap_to_one_hot(seg[0], self.label_manager.foreground_labels, data.dtype)
            data = np.vstack((data, seg_onehot))

        data = torch.from_numpy(data)

        return {'data': data, 'data_properites': props, 'ofile': ofname}

