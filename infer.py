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

from torch.nn import functional as F
from torch.utils.data import DataLoader

from pre_processing import wav2spec, spec2wav

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
    BatchSize = 1
    Epochs = 10
    Dtype = torch.float

    TsNoisyVoiceFolder = "./data/clean_audio/test_set/"
    TsEnhacVoiceFolder = "./data/enhance_audio/"
    cr_f(TsEnhacVoiceFolder)

    # # model
    ModelPath = f"./model/conformer/mdl.pkl"
    CkpotPath = f"./model/conformer/checkpoint/"

    encoder_dim = 128
    num_encoder_layers = 5
    num_attention_heads = 8

    from conformer_train import conformer_tr
    conformer_tr = conformer_tr(MdlPth=ModelPath, CPTPth=CkpotPath, encoder_dim=encoder_dim, num_encoder_layers=num_encoder_layers, num_attention_heads=num_attention_heads)
    conformer_tr._load_Mdl()

    n_paths = []
    e_paths = []

    NoyWavData = os.listdir(TsNoisyVoiceFolder)
    for noywavfile in NoyWavData:
        clnfilenme = noywavfile.split(filesep)[-1].split('_')[0]

        n_paths.append(join(TsNoisyVoiceFolder,noywavfile))
        e_paths.append(join(TsEnhacVoiceFolder,noywavfile))

    for noyfile, enhfile in zip(n_paths,e_paths):
        (fs,nsyData) = audioread(noyfile)

        TmpSpec, NoyPhas = wav2spec(nsyData/(2.0**15),returnPhase = True)
        NoyFeat = torch.from_numpy(np.expand_dims(TmpSpec.T, axis=0)).type(torch.float)

        mdlout = conformer_tr.eval(NoyFeat)

        enhWav = spec2wav(mdlout[0].detach().cpu().type(Dtype).T, NoyPhas[:, :mdlout[0].shape[0]])

        audiowrite(enhfile, fs, np.round(np.array(enhWav * (2**15))).astype('int16'))
