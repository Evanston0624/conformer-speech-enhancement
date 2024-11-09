import os
from os.path import sep as filesep
from os.path import join
from scipy.io.wavfile import read as audioread
# from pre_processing import wav2spec
from pre_processing import wav2ibm

import numpy as np
import torch
import random
np.random.seed(0)

class C_Dataset():
    def __init__(self, CleanVoiceFolder, NoisyVoiceFolder, mode):
        self.n_paths = []
        self.c_paths = []
        if isinstance(NoisyVoiceFolder, str):
            NoyWavData = os.listdir(NoisyVoiceFolder)
            for noywavfile in NoyWavData:
                self.n_paths.append(join(NoisyVoiceFolder,noywavfile))
                self.c_paths.append(join(CleanVoiceFolder,noywavfile))

        elif isinstance(NoisyVoiceFolder, list):
            for NVF in NoisyVoiceFolder :
                NoyWavData = os.listdir(NVF)
                for noywavfile in NoyWavData:
                    self.n_paths.append(join(NVF,noywavfile))
                    self.c_paths.append(join(CleanVoiceFolder,noywavfile))

        self.mode = mode
        self.batch_Frame_size = 300

    def _get_waveform_feat(self, Fs, wavData):
        # TmpSpec = wav2spec(wavData)
        TmpSpec = wav2ibm(wavData)
        WavFeat = np.expand_dims(TmpSpec.T, axis=0)
        Dtype = torch.double
        return torch.from_numpy(WavFeat).type(Dtype)

    def _concat_TrFea_to_Batch_frame_size(self):
        _, frms, dims = self.IbmSpec.shape
        # 增加5個frame做為防呆
        while frms <= self.batch_Frame_size+10:
            self.InpFeat = torch.cat((self.InpFeat, self.InpFeat), axis = 1)
            self.ClnSpec = torch.cat((self.ClnSpec, self.ClnSpec), axis = 1)
            self.IbmSpec = torch.cat((self.IbmSpec, self.IbmSpec), axis = 1)
            _, frms, dims = self.IbmSpec.shape

    def _get_fea(self,index):
        # 強制對齊
        Fs, NoyData = audioread(self.n_paths[index].strip())
        Fs2, ClnData = audioread(self.c_paths[index].strip())

        if Fs != Fs2 :
            raise ValueError("Sampling rates of the two audio files do not match.")
        if len(NoyData) > len(ClnData) :
            ClnData = np.pad(ClnData, (0, len(NoyData) - len(ClnData)), 'constant')
        elif len(ClnData) > len(NoyData) :
            NoyData = np.pad(NoyData, (0, len(ClnData) - len(NoyData)), 'constant')

        NoeData=NoyData-ClnData

        self.InpFeat = self._get_waveform_feat(Fs, NoyData/(2.0**15))
        self.ClnSpec = self._get_waveform_feat(Fs, ClnData/(2.0**15))
        # IBM
        self.IbmSpec = self.ClnSpec >= self._get_waveform_feat(Fs, NoeData/(2.0**15))

        self._concat_TrFea_to_Batch_frame_size()

        _, frms, dims = self.IbmSpec.shape

        self.St_Fr = random.randint(0, frms - self.batch_Frame_size)
        self.Ed_Fr = self.St_Fr + self.batch_Frame_size

    def __getitem__(self, index):
        if self.mode == 'train':
            self._get_fea(index)
            return self.InpFeat[:,self.St_Fr:self.Ed_Fr,:], self.IbmSpec[:,self.St_Fr:self.Ed_Fr,:], self.ClnSpec[:,self.St_Fr:self.Ed_Fr,:], self.mode

        elif self.mode == 'valid':
            self._get_fea(index)
            return self.InpFeat[:,self.St_Fr:self.Ed_Fr,:], self.IbmSpec[:,self.St_Fr:self.Ed_Fr,:], self.ClnSpec[:,self.St_Fr:self.Ed_Fr,:], self.mode

    def __len__(self):  # return count of sample we have
        return len(self.n_paths)