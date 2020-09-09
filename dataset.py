#!/usr/local/bin/python
# -*- coding:utf-8 -*-
#
# tts/Alternative/dataset.py
#
# NeuraVoice dataset
#

import sys
sys.path.append('../')
import os
import numpy as np
import torch
from tools.libaudio.feature import melspectrogram
from tools.libaudio.utils import utterance_edge_indices, normalize, reshape_with_window
from tools.libaudio.encodes import split_signal, mulaw_encode
from datasets.voice_dataset import VoiceDataset
from mlutils.utils import to_onehot
from models.phoneme import Phoneme43


class NeuraVoiceDataset(VoiceDataset):

    def __init__(
            self, sample_rate=24000, key_name='jsut_ver1.1',
            with_conditions=False, remove_silence=True,
            batch_size=1, window_size=1000,
            n_fft=2048, hop_length=300, power=2.0, verbose=False):

        self.__root_dir__ = f'/diskB/6/Datasets/VoiceData/{key_name}/preprocessed'
        self.__wav_dir__ = f'/diskB/6/Datasets/VoiceData/{key_name}/preprocessed/wav24kHz'
        self.__f0_dir__ = f'{self.__root_dir__}/f0'
        self.__phoneme_dir__ = f'{self.__root_dir__}/phoneme'
        self.__sample_rate__ = sample_rate
        self.__with_conditions__ = with_conditions
        self.__remove_silence__ = remove_silence
        self.__batch_size__ = batch_size
        self.__window_size__ = window_size
        self.__n_fft__ = n_fft
        self.__hop_length__ = hop_length
        self.__power__ = power
        self.verbose = verbose

        self.wav_file_names = os.listdir(self.__wav_dir__)
        self.f0_file_names = os.listdir(self.__f0_dir__)
        self.phonemes_file_names = os.listdir(self.__phoneme_dir__)

    def char_to_wav(self, items):
        """Wavs, Mels, Labels.
        returns:
            - wavs (torch.FloatTensor): wav (N, T) *not used
            - targets (torch.FloatTensor): wav (N, T) *not used
            - mels (torch.FloatTensor): mel spectrogram (N, T, H)
            - labels (torch.FloatTensor): character sequence (N, U)
        """
        wavs = []
        targets = []
        labels = []
        for i, item in enumerate(items):
            # encode wav
            start, end = utterance_edge_indices(item.get('wav'))
            wav_encoded = mulaw_encode(item['wav'][start:end])

            # wav & target
            wav = torch.FloatTensor(wav_encoded[:-1])
            target = torch.FloatTensor(wav_encoded[1:])

            # mel
            T = wav.shape[0]

            # labels
            label = torch.LongTensor(
                to_onehot(item['phonemes'], n_class=len(Phoneme43)))

            # list of Tensors
            wavs += [wav]
            targets += [target]
            labels += [label]

        # list to matrix
        if self.__batch_size__ > 1:
            wavs = torch.FloatTensor(self.pad(wavs))
            targets = torch.FloatTensor(self.pad(targets))
            labels = torch.FloatTensor(self.pad(labels))
        else:
            wavs = torch.stack(wavs, dim=0)
            targets = torch.stack(targets, dim=0)
            labels = torch.stack(labels, dim=0)

        return wavs, targets, labels

    def char_to_mel(self, items):
        """Wavs, Mels, Labels.
        returns:
            - wavs (torch.FloatTensor): wav (N, T) *not used
            - mels (torch.FloatTensor): mel spectrogram (N, T, H)
            - labels (torch.FloatTensor): character sequence (N, U)
        """
        mels = []
        labels = []
        for i, item in enumerate(items):
            mels += [torch.FloatTensor(self.wav_to_mel(item['wav'])).transpose(0, 1)]
            labels += [torch.LongTensor(
                to_onehot(item['phonemes'], n_class=len(Phoneme43)))]
        if self.__batch_size__ > 1:
            mels = torch.FloatTensor(self.pad(mels))
            labels = torch.LongTensor(self.pad(labels))
        else:
            mels = torch.stack(mels, dim=0)
            labels = torch.stack(labels, dim=0)
        if self.verbose:
            print(mels)
            print(labels)
        return mels, labels

    def pad(self, items, constant_value=0):
        """Add padding to items.

        args:
            - items (list of toch.Tensor):
        """
        lens = [item.shape[0] for item in items]
        batch = len(items)
        max_len = np.max(lens)
        dim = len(items[0].shape)
        if self.verbose:
            print(f'max len {max_len}')
            print(f'item shape {items[0].shape}')
        if dim == 2:
            H = items[0].shape[1]
            result = np.zeros((batch, max_len, H))
        else:
            result = np.zeros((batch, max_len))
        if self.verbose:
            print(f'final shape {result.shape}')
        for i, item in enumerate(items):
            if isinstance(item, torch.Tensor):
                item = item.numpy()
            if dim == 1:
                pad_len = max_len - len(item)
                result[i] = np.pad(item, (0, pad_len), mode='constant',
                    constant_values=constant_value)
            if dim == 2:
                pad_len = max_len - item.shape[0]  # (T, H)
                result[i] = np.pad(item, ((0, pad_len), (0, 0)), mode='constant',
                    constant_values=constant_value)
        if self.verbose:
            print(f'result shape {result.shape}')
            print(f'result[0] shape {result[0]}')
            print(f'result[-1] shape {result[-1]}')
        return result

    def mel_to_wav(self, items):
        """Wavs, Mels, Labels.
        returns:
            - wavs (torch.FloatTensor): wav (N, T) *not used
            - targets (torch.FloatTensor): wav (N, T) *not used
            - mels (torch.FloatTensor): mel spectrogram (N, T, H)
            - labels (torch.FloatTensor): character sequence (N, U)
        """
        wavs = []
        targets = []
        mels = []
        for i, item in enumerate(items):
            # encode wav
            start, end = utterance_edge_indices(item.get('wav'))
            wav_raw_trim = item['wav'][start:end]
            wav_encoded = mulaw_encode(wav_raw_trim)

            # wav & target
            wav = torch.FloatTensor(wav_encoded[:-1])
            target = torch.FloatTensor(wav_encoded[1:])

            # mel
            T = wav.shape[0]
            mel = torch.FloatTensor(
                self.upsample(
                    torch.FloatTensor(self.wav_to_mel(wav_raw_trim[:-1])), T))

            # list of Tensors
            wavs += [wav]
            targets += [target]
            mels += [mel]

        # list to matrix
        if self.__batch_size__ > 1:
            wavs = torch.FloatTensor(self.pad(wavs))
            targets = torch.FloatTensor(self.pad(targets))
            mels = torch.FloatTensor(self.pad(mels))
        else:
            wavs = torch.stack(wavs, dim=0)
            targets = torch.stack(targets, dim=0)
            mels = torch.stack(mels, dim=0)

        return wavs, targets, mels

    def wav_to_c_f(self, wav):
        """Wav to Coarse, Fine.

        args:
            - wav (np.array):
        """
        return split_signal(normalize(wav), from_bit=16, to_bit=8)

    def wav_to_mel(self, wav):
        """Wav to Mel.

        args:
            - wav (np.array):
        """
        return melspectrogram(
            wav, sample_rate=self.__sample_rate__,
            n_fft=self.__n_fft__, hop_length=self.__hop_length__,
            power=self.__power__)

    def upsample(self, mel, target_step: int):
        """
        args:
            - mel (torch.FloatTensor): (H, T)
            - target_step:
        returns:
            - upsampled_mel (torch.FloatTensor): (H, T)
        """
        mel_upsample = torch.nn.functional.interpolate(
            mel.unsqueeze(0), target_step, mode='linear', align_corners=True)
        return mel_upsample.squeeze().transpose(0,1)
