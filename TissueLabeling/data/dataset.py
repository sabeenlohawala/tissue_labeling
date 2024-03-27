#!/.venv/bin/python
# -*- coding: utf-8 -*-
"""
@Author: Matthias Steiner
@Contact: matth406@mit.edu
@File: dataset.py
@Date: 2023/04/03
"""
import glob
import os

import random
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.ndimage import affine_transform
from torchvision import transforms

from TissueLabeling.brain_utils import mapping
from TissueLabeling.data.cutout import Cutout
from TissueLabeling.data.mask import Mask
from TissueLabeling.brain_utils import (
    mapping,
    create_affine_transformation_matrix,
    draw_value_from_distribution,
)


class NoBrainerDataset(Dataset):
    def __init__(self, mode: str, config) -> None:
        """
        Initializes the object with the given `file_dir` and `pretrained` parameters.

        Args:
            mode: The subdirectory where the files are located.
            pretrained: Whether or not to use pretrained models.

        Returns:
            None
        """
        if mode == "val":
            mode = "validation"

        if mode not in ["train", "test", "validation"]:
            raise Exception(
                f"{mode} is not a valid data mode. Choose from 'train', 'test', or 'validation'."
            )
        self.mode = mode

        # Set the model name
        self.model_name = config.model_name

        self.nr_of_classes = config.nr_of_classes

        # Set the pretrained attribute
        self.pretrained = config.pretrained

        self.new_kwyk_data = config.new_kwyk_data
        if config.new_kwyk_data:
            self.images = sorted(glob.glob(f"{config.data_dir}/{mode}/features/*.npy"))
            self.masks = sorted(glob.glob(f"{config.data_dir}/{mode}/labels/*.npy"))
            self.affines = []
        else:
            # Get a list of all the brain image files in the specified directory
            self.images = sorted(glob.glob(f"{config.data_dir}/{mode}/brain*.npy"))

            # Get a list of all the mask files in the specified directory
            self.masks = sorted(glob.glob(f"{config.data_dir}/{mode}/mask*.npy"))

            # Get a list of all the affine matrices for rigid transformations
            self.affines = sorted(glob.glob(f"{config.aug_dir}/{mode}/affine*.npy"))

        # only augment the train, not validation or test
        if self.mode == "train":
            self.augment = config.augment
            self.aug_mask = config.aug_mask
            self.aug_cutout = config.aug_cutout
            self.aug_mask = config.aug_mask
            self.null_half = config.null_half
            self.cutout_obj = Cutout(config.cutout_n_holes, config.cutout_length)
            self.mask_obj = Mask(config.mask_n_holes, config.mask_length)
            self.intensity_scale = (
                transforms.ColorJitter(brightness=0.2, contrast=0.2)
                if config.intensity_scale
                else None
            )
        else:
            self.augment = 0
            self.aug_mask = 0
            self.aug_cutout = 0
            self.aug_mask = 0
            self.null_half = 0
            self.cutout_obj = None
            self.mask_obj = None
            self.intensity_scale = None

        if self.augment or self.intensity_scale:
            print(f"augmenting data!")
            self.images = self.images[:] + self.images[:]
            self.masks = self.masks[:] + self.masks[:]
            self.affines = self.affines[:] + self.affines[:]

        # Limit the number of images and masks to the first 100 during debugging
        if config.debug:
            print("debug mode")
            self.images = self.images[:100]
            self.masks = self.masks[:100]
            self.affines = self.affines[:100]

        # Load the normalization constants from the file directory
        if not config.new_kwyk_data:
            self.normalization_constants = np.load(
                os.path.join(
                    f"{config.data_dir}/{mode}", "..", "normalization_constants.npy"
                )
            )

        if os.path.exists(f"{config.data_dir}/{mode}/keys.npy"):
            self.keys = np.load(f"{config.data_dir}/{mode}/keys.npy")

    def __getitem__(self, idx):
        # returns (image, mask)
        image = torch.from_numpy(np.load(self.images[idx]))
        mask = torch.from_numpy(np.load(self.masks[idx]))

        # randomly augment
        augment_coin_toss = random.randint(0, 1)
        if self.augment and augment_coin_toss == 1:
            # apply affine
            if self.new_kwyk_data:
                affine = torch.from_numpy(self._get_affine_matrix(image))
            else:
                affine = torch.from_numpy(np.load(self.affines[idx]))
            image = affine_transform(image.squeeze(), affine, mode="constant")
            mask = affine_transform(mask.squeeze(), affine, mode="constant", order=0)

            image = torch.from_numpy(image)
            mask = torch.from_numpy(mask)

            # random left/right flip
            flip_coin_toss = random.randint(0, 1)
            if flip_coin_toss == 1:
                image = torch.flip(image, dims=(1,))
                mask = torch.flip(mask, dims=(1,))

            # apply cutout
            if self.aug_cutout == 1:
                image = self.cutout_obj(image)

            # apply mask
            if self.aug_mask == 1:  # TODO: if or elif?
                image, mask = self.mask_obj(image, mask)
            
            # null half
            left_right_null = self.null_half
            if left_right_null == 3: # mix null left/right and null up/down
                left_right_null = random.randint(1,2)
            if left_right_null == 1: # null left/right
                width_half = image.shape[1] // 2
                if random.randint(0, 1):
                    image[:,:width_half] = 0
                    mask[:,:width_half] = 0
                else:
                    image[:,width_half:] = 0
                    mask[:,width_half:] = 0
            elif left_right_null == 2: # null up/down
                height_half = image.shape[1] // 2
                if random.randint(0, 1):
                    image[:height_half,:] = 0
                    mask[:height_half,:] = 0
                else:
                    image[height_half:,:] = 0
                    mask[height_half:,:] = 0

            # resize image to [1,h,w] again
            image = image.unsqueeze(dim=0)
            mask = mask.unsqueeze(dim=0)

        if self.intensity_scale and augment_coin_toss:
            image = self.intensity_scale(image)

        if "unet" in self.model_name:
            image = image[:, 1:161, 1:193]
            mask = mask[:, 1:161, 1:193]

        if not self.new_kwyk_data:
            if self.nr_of_classes == 2:
                mask[mask != 0] = 1
            elif self.nr_of_classes == 7:
                mask = mapping(mask, self.nr_of_classes, original=False)

            # normalize image
            image = (
                image - self.normalization_constants[0]
            ) / self.normalization_constants[1]

        if self.pretrained:
            return image.repeat((3, 1, 1)), mask

        if self.new_kwyk_data:
            image = image.to(torch.float32)
            mask = torch.tensor(
                mapping(np.array(mask), self.nr_of_classes, original=True)
            )

        return image, mask

    def __len__(self):
        return len(self.images)

    def _get_affine_matrix(self, image):
        # which augmentations to perform (based on SynthSeg)
        scaling_bounds = 0.2  # the scaling coefficients will be sampled from U(1-scaling_bounds; 1+scaling_bounds)
        rotation_bounds = 15  # the rotation angles will be sampled from U(-rotation_bounds; rotation_bounds)
        shearing_bounds = 0.012  # the shearing coefficients will be sampled from U(-shearing_bounds; shearing_bounds)
        translation_bounds = False  # no translation is performed, as this is already modelled by the random cropping
        enable_90_rotations = False

        # randomize augmentations
        batchsize = 1
        n_dims = 2
        center = np.array([image.shape[0] // 2, image.shape[1] // 2])

        scaling = draw_value_from_distribution(
            scaling_bounds,
            size=n_dims,
            centre=1,
            default_range=0.15,
            return_as_tensor=False,
            batchsize=batchsize,
        )

        rotation = draw_value_from_distribution(
            rotation_bounds,
            size=1,
            default_range=15.0,
            return_as_tensor=False,
            batchsize=batchsize,
        )

        shearing = draw_value_from_distribution(
            shearing_bounds,
            size=n_dims**2 - n_dims,
            default_range=0.01,
            return_as_tensor=False,
            batchsize=batchsize,
        )
        affine_matrix = create_affine_transformation_matrix(
            n_dims=2,
            scaling=scaling,
            rotation=rotation,
            shearing=shearing,
            translation=None,
        )

        # Translate the center back to the origin
        translation_matrix1 = np.array(
            [[1, 0, -center[0]], [0, 1, -center[1]], [0, 0, 1]]
        )

        # Translate the center to the original position
        translation_matrix2 = np.array(
            [[1, 0, center[0]], [0, 1, center[1]], [0, 0, 1]]
        )

        final_matrix = translation_matrix2 @ affine_matrix @ translation_matrix1
        return final_matrix


def get_data_loader(
    # data_dir: str,
    config,
    num_workers: int = 4 * torch.cuda.device_count(),
):
    # train_dataset = NoBrainerDataset(f"{config.data_dir}/train", config)
    # val_dataset = NoBrainerDataset(f"{config.data_dir}/validation", config)
    # test_dataset = NoBrainerDataset(f"{config.data_dir}/test", config)
    train_dataset = NoBrainerDataset("train", config)
    val_dataset = NoBrainerDataset("validation", config)
    test_dataset = NoBrainerDataset("test", config)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size
    )

    return train_loader, val_loader, test_loader, tuple(train_dataset[0][0].shape[1:])
