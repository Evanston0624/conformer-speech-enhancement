#!/usr/bin/env python
# coding: utf-8

# # Global setting
# In[ ]:

# ## Environment
# In[ ]:

# ## Packets
# In[ ]:

import os
import torch

import numpy as np

from scipy.io.wavfile import read as audioread
from scipy.io.wavfile import write as audiowrite
from os.path import join
from os.path import sep as filesep

from pesq import pesq
from pystoi import stoi

from torch.nn import functional as F
from torch.utils.data import DataLoader

from pre_processing import wav2spec, spec2wav

def snrcal(InpClnPth, InpNoyPth):
    epsilon = np.finfo(float).eps
    #SNRcal
    (Fs,voice_data)=audioread(InpClnPth)
    (Fs,noisy_data)=audioread(InpNoyPth)

    voice_data=voice_data/1.0
    noisy_data=noisy_data/1.0

    nsypoints=noisy_data.shape[0]

    Err_signal = voice_data[0:nsypoints] - noisy_data

    ref_pow = np.dot(voice_data,voice_data)
    err_pow = np.dot(Err_signal,Err_signal)

    esnr = ref_pow/(err_pow + epsilon * (err_pow == 0).astype('int'))
    return 10 * np.log10(esnr + epsilon * (esnr == 0).astype('int'))

def si_sdr(estimate, reference):
    alpha = np.dot(estimate.T, reference) / (np.dot(estimate.T, estimate) + epsilon)

    molecular = ((alpha * reference) ** 2).sum()  # 分子
    denominator = ((alpha * reference - estimate) ** 2).sum()  # 分母
    return 10 * np.log10((molecular) / (denominator+epsilon))

# device
def get_device():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        print(f'DEVICE: [{torch.cuda.current_device()}] {torch.cuda.get_device_name()}')
    else:
        print(f'DEVICE: CPU')
    device = 'cpu'
    return device

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

def cr_f(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__ == '__main__':
    BatchSize = 60
    Epochs = 40
    # Layer_Size_List = [100, 100, 100, 100]
    # SNR = np.arange(-3,11,1)
    # data
    TrCleanVoiceFolder = "./data/wsj_8k_mv/train/"
    TsCleanVoiceFolder = "./data/wsj_8k_mv/test/"
    VaCleanVoiceFolder = "./data/wsj_8k_mv/valid/"

    for i in range (3, 10) :
        TrNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/train/"

        TsNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/test/"

        VaNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/valid/"

        # model
        ModelPath = f"./model/conformer/L{i}_B4_P7_G2/"
        cr_f(ModelPath)

        CkpotPath = f"./model/conformer/checkpoint/L{i}_B4_P7_G2"
        cr_f(CkpotPath)

        # ClnWavData = os.listdir(TrCleanVoiceFolder)
        # NoeWavData = os.listdir(TrNoiseSignlFolder)

        # # data loader
        # for clnwavfile in ClnWavData:
        #   for noewavfile in NoeWavData:
        #     for snrInd in SNR:

        #         nsywavname = f'{clnwavfile.split(".")[0]}-{noewavfile.split(".")[0]}-{snrInd}.wav'
        #         #mix_noise(join(TrCleanVoiceFolder,clnwavfile),join(TrNoiseSignlFolder,noewavfile),join(TrNoisyVoiceFolder,nsywavname),snrInd)
        
        from load_data import C_Dataset
        TrDataLdr = {
                    'tr' : DataLoader(C_Dataset(TrCleanVoiceFolder, TrNoisyVoiceFolder, 'train'), batch_size=BatchSize, shuffle=True, num_workers=0, collate_fn = collate_fn),
                     'va' : DataLoader(C_Dataset(VaCleanVoiceFolder, VaNoisyVoiceFolder, 'valid'), batch_size=1, num_workers=0, collate_fn = collate_fn)
                    }

        # training
        from conformer_train import conformer_tr
        conformer_tr = conformer_tr(MdlPth=ModelPath, CPTPth=CkpotPath)
        conformer_tr.train(Epochs,TrDataLdr)