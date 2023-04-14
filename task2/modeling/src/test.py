"""Test quality of denoising."""

import argparse
import logging
import os
import typing as tp
from pathlib import Path
from runpy import run_path

import numpy as np
from tqdm import tqdm

from predict import Denoiser


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default=os.path.join('src', 'configs', 'config.py'),
        type=str,
        help='Path to experiment config file (*.py)',
    )
    parser.add_argument(
        '--ckpt_path',
        required=True,
        type=str,
        help='Path to experiment checkpoint (*.ckpt)',
    )
    parser.add_argument(
        '--data_dir',
        default=os.path.join('src', 'data', 'test'),
        type=str,
        help='Path to directory with test data (must contain clean and noisy subfolders)',
    )
    parser.add_argument(
        '--device_id',
        default=0,
        type=int,
        help='Device id to be use in testing',
    )
    parser.add_argument(
        '--gpu_mem_limit_gb',
        default=6,
        type=int,
        help='GPU memory limit in Gb',
    )
    return parser.parse_args()


def mse(clean_spectrum: np.ndarray, noisy_spectrum: np.ndarray) -> float:
    return np.square(clean_spectrum - noisy_spectrum).mean()


def test(denoiser: Denoiser, data_dir: str):
    metrics = {
        'noise-clean': {'mse': []},
        'clean-clean': {'mse': []}
        }
    noisy_spectrums_paths = sorted(list(Path(data_dir).rglob('*.npy')))

    for noisy_spectrum_path in tqdm(noisy_spectrums_paths):
        noisy_spectrum = np.load(noisy_spectrum_path)

        clean_spectrum_path = str(noisy_spectrum_path).replace('noisy', 'clean')
        clean_spectrum = np.load(clean_spectrum_path)

        denoised_spectrum = denoiser([noisy_spectrum])
        metrics['noise-clean']['mse'].append(mse(clean_spectrum, denoised_spectrum))

        denoised_spectrum = denoiser([clean_spectrum])
        metrics['clean-clean']['mse'].append(mse(clean_spectrum, denoised_spectrum))

    for metric_type, metric in metrics.items():
        for metric_name, metric_values in metric.items():
            logging.info(f'{metric_type.upper()}: MEAN {metric_name.upper()}: {np.mean(metric_values):.4f}')


if __name__ == '__main__':
    args = parse()
    logging.basicConfig(level=logging.INFO)
    config_module = run_path(args.config)
    config = config_module['CONFIG']

    denoiser = Denoiser(
        model_path=args.ckpt_path,
        max_len=config.dataset.max_len,
        n_mels=config.dataset.n_mels,
        device_id=args.device_id,
        gpu_mem_limit_gb=args.gpu_mem_limit_gb,
    )

    test(denoiser, data_dir=args.data_dir)
