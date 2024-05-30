import os
from os.path import sep as filesep
from os.path import join
from scipy.io.wavfile import read as audioread
from pre_processing import wav2spec, spec2wav

import numpy as np
import torch
import random
np.random.seed(0)

class C_Dataset():
    def __init__(self, CleanVoiceFolder, NoisyVoiceFolder, mode):
        self.n_paths = []
        self.c_paths = []

        NoyWavData = os.listdir(NoisyVoiceFolder)
        for noywavfile in NoyWavData:
            clnfilenme = noywavfile.split(filesep)[-1].split('-')[0]
            self.n_paths.append(join(NoisyVoiceFolder,noywavfile))
            # self.c_paths.append(join(CleanVoiceFolder,f'{clnfilenme}.wav'))
            self.c_paths.append(join(CleanVoiceFolder,clnfilenme))

        self.mode = mode
        self.batch_Frame_size = 300
    
    def _get_waveform_feat(self, Fs, wavData):
        TmpSpec = wav2spec(wavData)
        WavFeat = np.expand_dims(TmpSpec.T, axis=0)
        Dtype = torch.double
        return torch.from_numpy(WavFeat).type(Dtype)

    ##
    def _concat_TrFea_to_Batch_frame_size(self,):
        _, frms, dims = self.RCnSpec.shape
        while frms <= self.batch_Frame_size:
            self.InpFeat = torch.cat((self.InpFeat, self.InpFeat), axis = 1)
            self.RCnSpec = torch.cat((self.RCnSpec, self.RCnSpec), axis = 1)
            _, frms, dims = self.InpFeat.shape

    def _get_fea(self,index):
        # print('self.n_paths[index]:',self.n_paths[index])
        # print('self.c_paths[index]:',self.c_paths[index])
        Fs, NoyData = audioread(self.n_paths[index].strip())
        self.InpFeat = self._get_waveform_feat(Fs, NoyData/(2.0**15))

        Fs, ClnData = audioread(self.c_paths[index].strip())
        self.RCnSpec = self._get_waveform_feat(Fs, ClnData/(2.0**15))

        self._concat_TrFea_to_Batch_frame_size()

        _, frms, dims = self.RCnSpec.shape
        self.St_Fr = random.randint(0, frms - self.batch_Frame_size)
        self.Ed_Fr = self.St_Fr + self.batch_Frame_size

    def __getitem__(self, index):

        if self.mode == 'train':
            self._get_fea(index)
            return self.InpFeat[:,self.St_Fr:self.Ed_Fr,:], self.RCnSpec[:,self.St_Fr:self.Ed_Fr,:], self.mode

        elif self.mode == 'valid':
            self._get_fea(index)
            return self.InpFeat[:,self.St_Fr:self.Ed_Fr,:], self.RCnSpec[:,self.St_Fr:self.Ed_Fr,:], self.mode

    def __len__(self):  # return count of sample we have
        return len(self.n_paths)

def collate_fn(data):
    Dtype = torch.double
    mode = data[0][-1]
    bthsize = len(data)

    if mode == 'train':
        _, frms, nydims = data[0][0].shape
        _, frms, rcdims = data[0][1].shape

        InpFeat = torch.zeros(bthsize, frms, nydims, dtype = data[0][0].dtype)
        RCnSpec = torch.zeros(bthsize, frms, rcdims, dtype = data[0][1].dtype)

        for idx, dt in enumerate(data):
            InpFeat[idx] = dt[0]
            RCnSpec[idx] = dt[1]

        RtDta = {'DegFeat': {'inpfeat':InpFeat},
                 'TarSpec': {'rcnspec': RCnSpec}
                }
        return RtDta

    elif mode == 'valid':
        _, frms, nydims = data[0][0].shape
        _, frms, rcdims = data[0][1].shape

        InpFeat = torch.zeros(bthsize, frms, nydims, dtype = data[0][0].dtype)
        RCnSpec = torch.zeros(bthsize, frms, rcdims, dtype = data[0][1].dtype)

        for idx, dt in enumerate(data):
            InpFeat[idx] = dt[0]
            RCnSpec[idx] = dt[1]

        RtDta = {'DegFeat': {'inpfeat':InpFeat},
                 'TarSpec': {'rcnspec': RCnSpec}
                }
        return RtDta