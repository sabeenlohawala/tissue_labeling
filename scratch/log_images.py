import argparse
import glob
import json
import os
import sys

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

# from matplotlib.colors import BoundaryNorm, ListedColormap
# import matplotlib.colors as mcolors
from PIL import Image
import cv2
from torch.utils.tensorboard import SummaryWriter

from TissueLabeling.config import Configuration
from TissueLabeling.brain_utils import crop, load_brains, mapping
from TissueLabeling.models.segformer import Segformer
from TissueLabeling.models.original_unet import OriginalUnet
from TissueLabeling.models.attention_unet import AttentionUnet
from TissueLabeling.metrics.metrics import Dice


def load_model(config, checkpoint_path=None):
    """
    Selects the model based on the model name provided in the config file.
    """
    if config.model_name == "segformer":
        model = Segformer(config.nr_of_classes, pretrained=config.pretrained)
    elif config.model_name == "original_unet":
        model = OriginalUnet(image_channels=1, nr_of_classes=config.nr_of_classes)
    elif config.model_name == "attention_unet":
        model = AttentionUnet(
            dim=16,
            channels=1,
            dim_mults=(2, 4, 8, 16, 32, 64),
        )
    else:
        print(f"Invalid model name provided: {config.model_name}")
        sys.exit()

    print(f"{config.model_name} found")
    if checkpoint_path:
        print(f"Loading from checkpoint...")
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(checkpoint_path)["model"])
        else:
            model.load_state_dict(
                torch.load(checkpoint_path, map_location=torch.device("cpu"))["model"]
            )

        # checkpoint path is something like: 'logdir/checkpoint_1000.chkpt'
        config.start_epoch = int(
            checkpoint_path.split("/")[-1].split(".")[0].split("_")[-1]
        )

    return model


def get_config(logdir):
    """
    Gets the config file based on the command line arguments.
    """
    chkpt_folder = os.path.join("results/", logdir)

    config_file = os.path.join(chkpt_folder, "config.json")
    if not os.path.exists(config_file):
        sys.exit(f"Configuration file not found at {config_file}")

    with open(config_file) as json_file:
        data = json.load(json_file)
    assert isinstance(data, dict), "Invalid Object Type"

    dice_list = sorted(glob.glob(os.path.join(chkpt_folder, "checkpoint*")))
    if not dice_list:
        sys.exit("No checkpoints exist to resume training")

    data["checkpoint"] = dice_list[-1]
    # data["start_epoch"] = int(os.path.basename(dice_list[0]).split('.')[0].split('_')[-1])

    args = argparse.Namespace(**data)
    config = Configuration(args, "config_resume.json")

    return config, dice_list


class Log_Images_v2:
    def __init__(
        self,
        # fabric: L.Fabric,
        config,
        writer=None,
    ):
        self.wandb_on = config.wandb_on
        self.pretrained = config.pretrained
        self.model_name = config.model_name
        self.nr_of_classes = config.nr_of_classes
        self.writer = writer
        self.metric = Dice(
            None,
            config,
            is_loss=False,
            class_specific_scores=False,
        )
        if not config.new_kwyk_data and not config.pad_old_data:
            if "unet" in self.model_name:
                self.image_shape = (160, 192)
            else:
                self.image_shape = (162, 194)
        else:
            self.image_shape = (256, 256)

        # color map to get always the same colors for classes
        if config.nr_of_classes in [51, 106, 7, 17, 2, 50, 6, 2, 16]:
            colors = self.__rgb_map_for_data(config.nr_of_classes)
            rgb = colors
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, config.nr_of_classes))
            rgb = colors[:, :-1] * 255
        self.color_range = np.zeros((256, 3))
        self.color_range[: rgb.shape[0], :] = rgb
        # # new plt cmap
        # self.cmap = ListedColormap(colors)
        # # new plt norm
        # bounds = np.arange(0, config.nr_of_classes + 1)
        # self.norm = BoundaryNorm(bounds, self.cmap.N)

        print("Loading brains...")
        # load always the same image from validation set
        image_file = "pac_36_orig.nii.gz"
        mask_file = "pac_36_aseg.nii.gz"
        file_path = "/nese/mit/group/sig/users/matth406/nobrainer_data/data/SharedData/segmentation/freesurfer_asegs/"
        brain, mask, _ = load_brains(image_file, mask_file, file_path)
        mask, _ = mapping(mask, nr_of_classes=self.nr_of_classes)

        self.brain_slices, self.mask_slices = [], []

        # randomly select slices in 3 directions
        self.slice_idx = [125, 150]
        normalization_constants = np.load(
            "/nese/mit/group/sig/users/matth406/nobrainer_data_norm/data_prepared_medium/normalization_constants.npy"
        )
        self.brain_slices = torch.empty(
            (len(self.slice_idx) * 3, 1, self.image_shape[0], self.image_shape[1])
        )
        self.mask_slices = torch.empty(
            (len(self.slice_idx) * 3, 1, self.image_shape[0], self.image_shape[1]),
            dtype=torch.long,
        )

        print("Initializing logging_dict...")
        i = 0
        self.logging_dict = {}
        for d in range(3):
            for slice_id in self.slice_idx:
                if d == 0:
                    brain_slice = crop(
                        brain[slice_id, :, :], self.image_shape[0], self.image_shape[1]
                    )
                    mask_slice = crop(
                        mask[slice_id, :, :], self.image_shape[0], self.image_shape[1]
                    )
                if d == 1:
                    brain_slice = crop(
                        brain[:, slice_id, :], self.image_shape[0], self.image_shape[1]
                    )
                    mask_slice = crop(
                        mask[:, slice_id, :], self.image_shape[0], self.image_shape[1]
                    )
                if d == 2:
                    brain_slice = crop(
                        brain[:, :, slice_id], self.image_shape[0], self.image_shape[1]
                    )
                    mask_slice = crop(
                        mask[:, :, slice_id], self.image_shape[0], self.image_shape[1]
                    )

                self.logging_dict[f"Image d{d} c{slice_id}"] = self.__create_plot(
                    self.wandb_on,
                    brain_slice,
                    caption="Raw Image",  # fig_path=f'/om2/user/sabeen/test_imgs/raw_d{d}_c{slice_id}_fs.png'
                )
                self.logging_dict[f"True Mask d{d} c{slice_id}"] = self.__create_plot(
                    self.wandb_on,
                    mask_slice,
                    caption="True Mask",
                    color_range=self.color_range,
                    # fig_path=f'/om2/user/sabeen/test_imgs/true_mask_d{d}_c{slice_id}_fs.png'
                )
                brain_slice = (
                    brain_slice - normalization_constants[0]
                ) / normalization_constants[1]
                brain_slice = torch.from_numpy(brain_slice).to(torch.float32)
                brain_slice = brain_slice[None, None]
                self.brain_slices[i] = brain_slice

                mask_slice = torch.tensor(mask_slice)[None, None].long()
                self.mask_slices[i] = mask_slice

                i += 1

        # send all slices to device
        if self.pretrained:
            self.brain_slices = self.brain_slices.repeat((1, 3, 1, 1))
        # self.brain_slices = fabric.to_device(self.brain_slices)
        # self.mask_slices = fabric.to_device(self.mask_slices)

    @staticmethod
    def __create_plot(
        wandb_on: bool,
        image: np.array,
        caption: str,
        color_range=None,
        fig_path: str = None,
    ):
        if fig_path is not None and len(fig_path.split(".")) == 1:
            fig_path = fig_path + ".png"

        if color_range is not None:
            image = image.astype(np.uint8)
            channels = [cv2.LUT(image, color_range[:, i]) for i in range(3)]
            new_img = np.dstack(channels)

            if fig_path is not None:
                new_img_bgr = np.dstack([channels[2], channels[1], channels[0]])
                cv2.imwrite(fig_path, new_img_bgr)
            image = Image.fromarray(np.uint8(new_img))
        else:
            img_min = np.min(image)
            img_max = np.max(image)
            new_img = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            if fig_path is not None:
                cv2.imwrite(fig_path, new_img)
            image = Image.fromarray(np.uint8(new_img))
        if wandb_on:
            image = wandb.Image(image, caption=caption)
        return image

    @torch.no_grad()
    def logging(self, model, e: int, commit: bool):
        model.eval()
        logits = model(self.brain_slices)
        probs = logits.argmax(1)
        probs = probs.cpu()
        model.train()

        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, probs.unsqueeze(1), 1)
        logits = logits * one_hot
        logits[logits > 0] = 1

        # mismatch = torch.zeros_like(self.mask_slices)
        # mismatch[probs.unsqueeze(1) != self.mask_slices] = 1
        # mismatch = mismatch.squeeze(1)

        # if logits.shape[2] < 256 or logits.shape[3] < 256:
        #     self.mask_slices = torch.stack([torch.tensor(crop(self.mask_slices[i].squeeze(0), *logits.shape[2:])).unsqueeze(0) for i in range(self.mask_slices.shape[0])])

        i = 0
        logging_dict = {}
        for d in range(3):
            for slice_id in self.slice_idx:
                logging_dict[f"Predicted Mask d{d} c{slice_id}"] = self.__create_plot(
                    self.wandb_on,
                    probs[i].numpy(),
                    caption=f"Epoch {e}",
                    color_range=self.color_range,
                    # fig_path=f'/om2/user/sabeen/test_imgs/predicted_mask_d{d}_c{slice_id}_fs.png'
                )
                # logging_dict[f"Mismatch Mask d{d} c{slice_id}"] = self.__create_plot(
                #     self.wandb_on,
                #     mismatch[i].numpy(),
                #     caption=f"Epoch {e}"
                # )
                print(f"Predicted Mask d{d} c{slice_id}: {self.metric(torch.cat([self.mask_slices[i].long().unsqueeze(0)]*10), torch.cat([logits[i].unsqueeze(0)]*10))}")
                i += 1
        current_logging_dict = self.logging_dict | logging_dict
        if self.wandb_on:
            wandb.log(current_logging_dict, commit=commit)

        if self.writer is not None:
            print("Logging images...")
            for key, img in current_logging_dict.items():
                img = np.array(img)
                if len(img.shape) == 3:
                    self.writer.add_image(
                        key, np.array(img), config.start_epoch, dataformats="HWC"
                    )
                elif len(img.shape) == 2:
                    self.writer.add_image(
                        key, np.array(img), config.start_epoch, dataformats="HW"
                    )
        return current_logging_dict

    def __extract_numbers_names_colors(self, FreeSurferColorLUT=""):
        """
        Extract lists of numbers, names, and colors representing anatomical brain
        regions from FreeSurfer's FreeSurferColorLUT.txt lookup table file.

        Parameters
        ----------
        FreeSurferColorLUT : string
            full path to FreeSurferColorLUT.txt file (else uses local Python file)

        Returns
        -------
        numbers : list of integers
            numbers representing anatomical labels from FreeSurferColorLUT.txt
        names : list of integers
            names for anatomical regions from FreeSurferColorLUT.txt
        colors : list of integers
            colors associated with anatomical labels from FreeSurferColorLUT.txt

        Examples
        --------
        >>> from mindboggle.mio.labels import extract_numbers_names_colors # doctest: +SKIP
        >>> ennc = extract_numbers_names_colors # doctest: +SKIP
        >>> en1,en2,ec = ennc('/Applications/freesurfer/FreeSurferColorLUT.txt') # doctest: +SKIP

        """
        import os
        from io import open

        # from ext.mindboggle.FreeSurferColorLUT import lut_text

        def is_number(s):
            try:
                int(s)
                return True
            except ValueError:
                return False

        # if os.environ['FREESURFER_HOME']:
        #     FreeSurferColorLUT = os.path.join(
        #              os.environ['FREESURFER_HOME'], 'FreeSurferColorLUT.txt')

        if FreeSurferColorLUT and os.path.exists(FreeSurferColorLUT):
            f = open(FreeSurferColorLUT, "r")
            lines = f.readlines()
        # else:
        #     lut = lut_text()
        #     lines = lut.split('\n')

        numbers = []
        names = []
        colors = []
        for line in lines:
            strings = line.split()
            if strings and is_number(strings[0]):
                numbers.append(int(strings[0]))
                names.append(strings[1])
                colors.append([int(strings[2]), int(strings[3]), int(strings[4])])

        return numbers, names, colors

    def __rgb_map_for_data(self, nr_of_classes):
        _, fs_names, fs_colors = self.__extract_numbers_names_colors(
            "/om2/user/sabeen/freesurfer/distribution/FreeSurferColorLUT.txt"
        )

        with open("/om2/user/sabeen/readme", "r") as f:
            voxmorph_label_index = f.read().splitlines()

        # get the last 24 lines of the readme file (format--> id: name)
        if nr_of_classes == 50:
            voxmorph_label_index = [
                item.strip().split(":")
                for item in voxmorph_label_index[200:250]
                if item != ""
            ]  # HACK
        elif nr_of_classes == 106:
            voxmorph_label_index = [
                item.strip().split(":")
                for item in voxmorph_label_index[91:197]
                if item != ""
            ]  # HACK
        elif nr_of_classes == 6:
            voxmorph_label_index = [
                item.strip().split(":")
                for item in voxmorph_label_index[253:259]
                if item != ""
            ]  # HACK
        elif nr_of_classes == 2:
            voxmorph_label_index = [
                item.strip().split(":")
                for item in voxmorph_label_index[262:264]
                if item != ""
            ]  # HACK
        elif nr_of_classes == 16:
            voxmorph_label_index = [
                item.strip().split(":")
                for item in voxmorph_label_index[266:282]
                if item != ""
            ]  # HACK
        else:
            raise Exception(f"coloring for nr_of_classes = {nr_of_classes} not found")

        voxmorph_label_index = [
            [int(item[0]), item[1].strip()] for item in voxmorph_label_index
        ]
        voxmorph_label_index_dict = dict(voxmorph_label_index)
        my_colors = [
            fs_colors[fs_names.index(item)]
            for item in voxmorph_label_index_dict.values()
        ]

        return np.array(my_colors)


logdir = "/om/scratch/tmp/sabeen/results/20240511-null-CBS1-bkgd-shapes-1-grid-1-noise-0-Msegformer\Ldice\Smed\RV0\BC0\C50\B288\LR0.001\PT0\A1"
config, checkpoint_paths = get_config(logdir)

writer = None
# writer = SummaryWriter(f"results/{logdir}")
writer = SummaryWriter(logdir)
print("SummaryWriter created")

# for i in range(len(checkpoint_paths)-1,-1,-1):
#     checkpoint_path = checkpoint_paths[i]
#     model = load_model(config, checkpoint_path)
#     print(f"Epoch {config.start_epoch}")

#     image_logger = Log_Images_v2(config)
#     log = image_logger.logging(model,config.start_epoch,True)

#     print('Logging images...')
#     for key, img in log.items():
#         img = np.array(img)
#         if len(img.shape) == 3:
#             writer.add_image(key, np.array(img), config.start_epoch, dataformats='HWC')
#         elif len(img.shape) == 2:
#             writer.add_image(key, np.array(img), config.start_epoch, dataformats='HW')

checkpoint_path = checkpoint_paths[-1]
model = load_model(config, checkpoint_path)
model.image_dims = (256,256)
print(f"Epoch {config.start_epoch}")

image_logger = Log_Images_v2(config, writer=writer)
log = image_logger.logging(model, config.start_epoch, True)

# print('Logging images...')
# for key, img in log.items():
#     img = np.array(img)
#     if len(img.shape) == 3:
#         writer.add_image(key, np.array(img), config.start_epoch, dataformats='HWC')
#     elif len(img.shape) == 2:
#         writer.add_image(key, np.array(img), config.start_epoch, dataformats='HW')

if writer is not None:
    writer.close()
