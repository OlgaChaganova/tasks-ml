"""Convert model in raw-torch format to ONNX"""

import argparse
import logging
import os
import typing as tp
from runpy import run_path

import numpy as np
import onnxruntime
import torch

from configs.base import Config
from model.model import UNetPredictNoise, UNetRecoverClean


def parse() -> tp.Any:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default=os.path.join('src', 'configs', 'config.py'), type=str, help='Path to config file (*.py)',
    )
    parser.add_argument(
        '--ckpt_path', required=True, type=str, help='Path to experiment checkpoint (*.ckpt)',
    )
    parser.add_argument(
        '--model_type',
        choices=['recover', 'predict_noise'],
        help='`recover`: recovering clean sound from noised; `predict_noise`: predict noise on the sound'
    )
    parser.add_argument(
        '--dir_to_save', type=str, default='weights/', help='Path to directory where .pt model will be saved',
    )
    parser.add_argument(
        '--check', action='store_true', help='Check correctness of converting by shape of output',
    )
    return parser.parse_args()


def convert_from_checkpoint(args: tp.Any, config: Config) -> tp.Tuple[str, torch.tensor, torch.tensor]:
    if args.model_type == 'recover':
        model = UNetRecoverClean.load_from_checkpoint(args.ckpt_path, criterion=config.train.criterion)

    elif args.model_type == 'predict_noise':
        model = UNetPredictNoise.load_from_checkpoint(args.ckpt_path, criterion=config.train.criterion)
    else:
        raise ValueError(f'Available model types are `recover`, `predict_noise`, but got {args.model_type}')

    model.eval()
    model_name = '_'.join(args.ckpt_path.split(os.sep)[-2:]).replace('ckpt', 'onnx')
    model_path = os.path.join(args.dir_to_save, model_name)

    input_sample = torch.randn((10, 1, config.dataset.n_mels, config.dataset.max_len))
    output_sample = model(input_sample)

    model.to_onnx(
        model_path,
        input_sample,
        export_params=True,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
        input_names=['input'],  # the model's input names
        output_names=['output'],  # the model's output names
        training=torch.onnx.TrainingMode.EVAL,
    )

    if os.path.isfile(model_path):
        logging.info(f'Model was successfully saved. File name: {model_path}')
    else:
        raise ValueError('An error was occurred. Check paths and try again.')
    if isinstance(output_sample, tuple):
        output_sample = output_sample[0]
    return model_path, input_sample, output_sample.detach()


def check(model_path: str, input_sample: torch.tensor, output_sample: torch.tensor):
    ort_session = onnxruntime.InferenceSession(model_path)
    ort_inputs = {ort_session.get_inputs()[0].name: input_sample.numpy()}
    output_sample_onnx = ort_session.run(None, ort_inputs)[0]

    atol = 1e-4
    if np.allclose(output_sample_onnx, output_sample.numpy(), atol=atol):
        logging.info('Model can be loaded and outputs look good!')
    else:
        logging.info(f'ONNX: {output_sample_onnx.shape}, torch: {output_sample.numpy().shape}')
        logging.info(f'ONNX: {output_sample_onnx[0]}, torch: {output_sample.numpy()[0]}')
        logging.error('Outputs of the converted model do not match output of the original torch model.')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    args = parse()
    config_module = run_path(args.config)
    exp_config = config_module['CONFIG']
    pt_model_path, input_sample, output_sample = convert_from_checkpoint(args, exp_config)
    if args.check:
        check(pt_model_path, input_sample, output_sample)
