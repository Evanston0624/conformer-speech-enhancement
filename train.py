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
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    # model path
    ModelPath = "./model"
    cr_f(ModelPath)
    ModelPath = "./model/conformer/"
    cr_f(ModelPath)
    CkpotPath = "./model/conformer/checkpoint/"
    cr_f(CkpotPath)

    # data
    TrCleanVoiceFolder = "./data/wsj_8k_mv/train/"
    VaCleanVoiceFolder = "./data/wsj_8k_mv/valid/"
    TrNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/train/"
    VaNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/valid/"
    encoder_dim = 128
    num_encoder_layers = 5
    num_attention_heads = 8
    Epochs = 150
    BatchSize = 35

    from load_data import C_Dataset
    TrDataLdr = {
                'tr' : DataLoader(C_Dataset(TrCleanVoiceFolder, TrNoisyVoiceFolder, 'train'), batch_size=BatchSize, shuffle=True, num_workers=0, collate_fn = collate_fn),
                 'va' : DataLoader(C_Dataset(VaCleanVoiceFolder, VaNoisyVoiceFolder, 'valid'), batch_size=1, num_workers=0, collate_fn = collate_fn)
                }

    # training
    from conformer_train import conformer_tr
    conformer_tr = conformer_tr(MdlPth=ModelPath, CPTPth=CkpotPath, encoder_dim=encoder_dim, num_encoder_layers=num_encoder_layers, num_attention_heads=num_attention_heads)
    conformer_tr.train(Epochs,TrDataLdr)