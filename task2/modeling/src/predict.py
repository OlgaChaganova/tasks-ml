import typing as tp

import albumentations as alb
import cv2
import numpy as np
import onnxruntime


class Denoiser(object):
    def __init__(
        self,
        model_path: str,
        device_id: int,
        gpu_mem_limit_gb: int,
        max_len: int,
        n_mels: int,
    ):
        """Initialize Denoiser

        Parameters
        ----------
        model_path: str
            Path to model weights in ONNX format.
        device_id: int
            Device ID.
        gpu_mem_limit_gb: int
           GPU memory limit in GB.
        max_len: int
            Maximum length of the input spectrogram.
        n_mels: int
            Number of mel-filters.

        """
        self._max_len = max_len
        self._n_mels = n_mels

        self._ort_session = onnxruntime.InferenceSession(
            model_path,
            providers=[
                (
                    'CUDAExecutionProvider',
                    {
                        'device_id': device_id,
                        'arena_extend_strategy': 'kNextPowerOfTwo',
                        'gpu_mem_limit': gpu_mem_limit_gb * 1024 * 1024 * 1024,
                        'cudnn_conv_algo_search': 'EXHAUSTIVE',
                        'do_copy_in_default_stream': True,
                    },
                ),
                'CPUExecutionProvider',
            ],
        )

        self._padding = alb.PadIfNeeded(
            min_height=n_mels,
            min_width=max_len,
            border_mode=0,
            value=0,  # silence (no sound)
            always_apply=True,
            position='bottom_left',
        )

    def __call__(self, batch: tp.List[np.ndarray]) -> tp.List[np.ndarray]:
        """Call denoiser on a batch of mel-spectrograms.

        Parameters
        ----------
        batch : tp.List[np.ndarray]
            List with mel-spectrograms in the np.ndarray format.

        Returns
        -------
        tp.List[np.ndarray]
            List with denoised mel-spectrograms in the np.ndarray format and original shape.
        """
        preprocessed_batch, initial_lens = self._preprocess(batch)
        ort_inputs = {
            self._ort_session.get_inputs()[0].name: preprocessed_batch
        }
        denoised_batch = self._ort_session.run(None, ort_inputs)[0]
        return self._postprocess(denoised_batch, initial_lens)

    def _preprocess(self, batch: tp.List[np.ndarray]) -> tp.Tuple[np.ndarray, tp.List[int]]:
        preprocessed_spectrum = []
        initial_lens = []

        for spectrum in batch:
            initial_lens.append(spectrum.shape[0])
            preprocessed_spectrum.append(self._transform(spectrum))

        return np.array(preprocessed_spectrum).astype(np.float32), initial_lens

    def _transform(self, spectrum: np.ndarray) -> np.ndarray:
        spectrum = spectrum.T
        spectrum = spectrum.astype(np.float32)
        spectrum = self._padding(image=spectrum)['image']
        if spectrum.shape != (self._n_mels, self._max_len):
            spectrum = cv2.resize(spectrum, dsize=[self._max_len, self._n_mels])
        return spectrum[np.newaxis, ...]

    def _postprocess(self, denoised_batch: np.ndarray, initial_lens:  tp.List[int]) -> tp.List[np.ndarray]:
        postprocessed_spectrums = []
        for denoised_spectrum, initial_len in zip(denoised_batch, initial_lens):
            if initial_len < self._max_len:
                postprocessed_spectrums.append(denoised_spectrum[0, :, :initial_len].T)
            elif initial_len > self._max_len:
                postprocessed_spectrums.append(cv2.resize(denoised_spectrum[0, :], dsize=[initial_len, self._n_mels]).T)
            else:
                postprocessed_spectrums.append(denoised_spectrum[0, :].T)
        return postprocessed_spectrums
