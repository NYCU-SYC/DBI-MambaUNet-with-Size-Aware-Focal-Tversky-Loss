import multiprocessing
import os
from time import sleep
from typing import List, Type, Union

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import (
    load_json, join, save_json, isfile, maybe_mkdir_p
)
from tqdm import tqdm

from nnunetv2.imageio.base_reader_writer import BaseReaderWriter
from nnunetv2.imageio.reader_writer_registry import determine_reader_writer_from_dataset_json
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


class DatasetFingerprintExtractor(object):
    def __init__(
        self,
        dataset_name_or_id: Union[str, int],
        num_processes: int = 8,
        verbose: bool = False,
        fixed_xy_size: int = 512
    ):
        """
        extracts the dataset fingerprint used for experiment planning.

        Modified behavior:
        1) DO NOT crop zero regions
        2) Keep full FOV
        3) Force XY to fixed_xy_size for shape statistics
        4) Adjust spacing accordingly so physical FOV remains consistent
        """
        dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
        self.verbose = verbose

        self.dataset_name = dataset_name
        self.input_folder = join(nnUNet_raw, dataset_name)
        self.num_processes = num_processes
        self.fixed_xy_size = int(fixed_xy_size)

        self.dataset_json = load_json(join(self.input_folder, 'dataset.json'))
        self.dataset = get_filenames_of_train_images_and_targets(self.input_folder, self.dataset_json)

        # We don't want to use all foreground voxels because that can accumulate a lot of data (out of memory).
        self.num_foreground_voxels_for_intensitystats = 10e7

    @staticmethod
    def collect_foreground_intensities(
        segmentation: np.ndarray,
        images: np.ndarray,
        seed: int = 1234,
        num_samples: int = 10000
    ):
        """
        images: shape (c, x, y, z) or (c, z, y, x), depending on reader output
        segmentation: shape (1, ...)
        """
        assert images.ndim == 4, f"Expected images.ndim == 4, got {images.ndim}"
        assert segmentation.ndim == 4, f"Expected segmentation.ndim == 4, got {segmentation.ndim}"

        assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values."
        assert not np.any(np.isnan(images)), "Images contains NaN values."

        rs = np.random.RandomState(seed)

        intensities_per_channel = []
        intensity_statistics_per_channel = []

        foreground_mask = segmentation[0] > 0

        for i in range(len(images)):
            foreground_pixels = images[i][foreground_mask]
            num_fg = len(foreground_pixels)

            intensities_per_channel.append(
                rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else np.array([], dtype=images.dtype)
            )

            intensity_statistics_per_channel.append({
                'mean': np.mean(foreground_pixels) if num_fg > 0 else np.nan,
                'median': np.median(foreground_pixels) if num_fg > 0 else np.nan,
                'min': np.min(foreground_pixels) if num_fg > 0 else np.nan,
                'max': np.max(foreground_pixels) if num_fg > 0 else np.nan,
                'percentile_99_5': np.percentile(foreground_pixels, 99.5) if num_fg > 0 else np.nan,
                'percentile_00_5': np.percentile(foreground_pixels, 0.5) if num_fg > 0 else np.nan,
            })

        return intensities_per_channel, intensity_statistics_per_channel

    @staticmethod
    def _compute_effective_shape_and_spacing(
        original_shape: List[int],
        original_spacing: List[float],
        fixed_xy_size: int
    ):
        """
        No cropping.
        Force XY to fixed_xy_size.
        Keep first spatial axis unchanged.

        For 3D:
            original_shape assumed [Z, Y, X] (matching whatever nnU-Net reader returns here)
            effective_shape = [Z, fixed_xy_size, fixed_xy_size]

            spacing is adjusted as:
                new_spacing_y = old_spacing_y * old_y / fixed_xy_size
                new_spacing_x = old_spacing_x * old_x / fixed_xy_size

        This preserves approximate physical FOV.
        """
        original_shape = [int(i) for i in original_shape]
        original_spacing = [float(i) for i in original_spacing]

        if len(original_shape) == 3:
            z, y, x = original_shape
            sz, sy, sx = original_spacing

            effective_shape = [z, int(fixed_xy_size), int(fixed_xy_size)]
            effective_spacing = [
                float(sz),
                float(sy * y / fixed_xy_size),
                float(sx * x / fixed_xy_size)
            ]
        elif len(original_shape) == 2:
            y, x = original_shape
            sy, sx = original_spacing

            effective_shape = [int(fixed_xy_size), int(fixed_xy_size)]
            effective_spacing = [
                float(sy * y / fixed_xy_size),
                float(sx * x / fixed_xy_size)
            ]
        else:
            raise RuntimeError(
                f"Unexpected spatial dimension: {len(original_shape)}. "
                f"original_shape={original_shape}"
            )

        return original_shape, effective_shape, original_spacing, effective_spacing

    @staticmethod
    def analyze_case(
        image_files: List[str],
        segmentation_file: str,
        reader_writer_class: Type[BaseReaderWriter],
        num_samples: int = 10000,
        fixed_xy_size: int = 512
    ):
        rw = reader_writer_class()
        images, properties_images = rw.read_images(image_files)
        segmentation, properties_seg = rw.read_seg(segmentation_file)

        # ------------------------------------------------------------------
        # NO CROPPING
        # ------------------------------------------------------------------
        spacing = properties_images['spacing']
        shape_before_crop = list(images.shape[1:])

        # collect intensity stats on FULL image + segmentation
        foreground_intensities_per_channel, foreground_intensity_stats_per_channel = \
            DatasetFingerprintExtractor.collect_foreground_intensities(
                segmentation, images, num_samples=num_samples
            )

        # ------------------------------------------------------------------
        # EFFECTIVE SHAPE / SPACING FOR PLANNING
        # ------------------------------------------------------------------
        original_shape, effective_shape, original_spacing, effective_spacing = \
            DatasetFingerprintExtractor._compute_effective_shape_and_spacing(
                shape_before_crop, spacing, fixed_xy_size
            )

        # no cropping was performed
        relative_size_after_cropping = 1.0

        return (
            effective_shape,                      # keep returning in "shape_after_crop" slot for planner compatibility
            effective_spacing,                    # adjusted spacing after forced XY size
            foreground_intensities_per_channel,
            foreground_intensity_stats_per_channel,
            relative_size_after_cropping,
            original_shape,
            original_spacing
        )

    def run(self, overwrite_existing: bool = False) -> dict:
        preprocessed_output_folder = join(nnUNet_preprocessed, self.dataset_name)
        maybe_mkdir_p(preprocessed_output_folder)
        properties_file = join(preprocessed_output_folder, 'dataset_fingerprint.json')

        if not isfile(properties_file) or overwrite_existing:
            reader_writer_class = determine_reader_writer_from_dataset_json(
                self.dataset_json,
                self.dataset[self.dataset.keys().__iter__().__next__()]['images'][0]
            )

            num_foreground_samples_per_case = int(
                self.num_foreground_voxels_for_intensitystats // len(self.dataset)
            )

            r = []
            with multiprocessing.get_context("spawn").Pool(self.num_processes) as p:
                for k in self.dataset.keys():
                    r.append(
                        p.starmap_async(
                            DatasetFingerprintExtractor.analyze_case,
                            ((
                                self.dataset[k]['images'],
                                self.dataset[k]['label'],
                                reader_writer_class,
                                num_foreground_samples_per_case,
                                self.fixed_xy_size
                            ),)
                        )
                    )

                remaining = list(range(len(self.dataset)))
                workers = [j for j in p._pool]

                with tqdm(desc=None, total=len(self.dataset), disable=self.verbose) as pbar:
                    while len(remaining) > 0:
                        all_alive = all([j.is_alive() for j in workers])
                        if not all_alive:
                            raise RuntimeError(
                                'Some background worker died.\n'
                                'This could be due to an error or out-of-RAM.'
                            )

                        done = [i for i in remaining if r[i].ready()]
                        for _ in done:
                            pbar.update()
                        remaining = [i for i in remaining if i not in done]
                        sleep(0.1)

            results = [i.get()[0] for i in r]

            # NOTE:
            # shapes_after_crop key is kept for compatibility with downstream nnU-Net planner.
            # But semantically it now means:
            # "shape without cropping, after applying fixed XY planning rule"
            shapes_after_crop = [res[0] for res in results]
            spacings = [res[1] for res in results]
            original_shapes = [res[5] for res in results]
            original_spacings = [res[6] for res in results]

            foreground_intensities_per_channel = [
                np.concatenate([res[2][i] for res in results if len(res[2][i]) > 0])
                for i in range(len(results[0][2]))
            ]

            median_relative_size_after_cropping = float(np.median([res[4] for res in results], 0))

            num_channels = len(
                self.dataset_json['channel_names'].keys()
                if 'channel_names' in self.dataset_json.keys()
                else self.dataset_json['modality'].keys()
            )

            intensity_statistics_per_channel = {}
            for i in range(num_channels):
                if len(foreground_intensities_per_channel[i]) == 0:
                    intensity_statistics_per_channel[i] = {
                        'mean': np.nan,
                        'median': np.nan,
                        'std': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                        'percentile_99_5': np.nan,
                        'percentile_00_5': np.nan,
                    }
                else:
                    intensity_statistics_per_channel[i] = {
                        'mean': float(np.mean(foreground_intensities_per_channel[i])),
                        'median': float(np.median(foreground_intensities_per_channel[i])),
                        'std': float(np.std(foreground_intensities_per_channel[i])),
                        'min': float(np.min(foreground_intensities_per_channel[i])),
                        'max': float(np.max(foreground_intensities_per_channel[i])),
                        'percentile_99_5': float(np.percentile(foreground_intensities_per_channel[i], 99.5)),
                        'percentile_00_5': float(np.percentile(foreground_intensities_per_channel[i], 0.5)),
                    }

            fingerprint = {
                "spacings": spacings,
                "shapes_after_crop": shapes_after_crop,
                "foreground_intensity_properties_per_channel": intensity_statistics_per_channel,
                "median_relative_size_after_cropping": median_relative_size_after_cropping,

                # extra debug / traceability info
                "original_spacings": original_spacings,
                "original_shapes": original_shapes,
                "cropping_was_performed": False,
                "fixed_xy_size_for_planning": int(self.fixed_xy_size),
            }

            try:
                save_json(fingerprint, properties_file)
            except Exception as e:
                if isfile(properties_file):
                    os.remove(properties_file)
                raise e
        else:
            fingerprint = load_json(properties_file)

        return fingerprint


if __name__ == '__main__':
    dfe = DatasetFingerprintExtractor(
        dataset_name_or_id=2,
        num_processes=8,
        verbose=False,
        fixed_xy_size=512
    )
    dfe.run(overwrite_existing=True)