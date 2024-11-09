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

def collate_fn(data):
    Dtype = torch.double
    mode = data[0][-1]
    bthsize = len(data)

    if mode == 'train':
        _, frms, nydims = data[0][0].shape
        _, frms, rcdims = data[0][1].shape
        _, frms, cndims = data[0][2].shape
        # print('frms:',frms,'nydims:',nydims,'rcdims:',rcdims)
        InpFeat = torch.zeros(bthsize, frms, nydims, dtype = data[0][0].dtype)
        IbmSpec = torch.zeros(bthsize, frms, rcdims, dtype = data[0][1].dtype)
        RCnSpec = torch.zeros(bthsize, frms, cndims, dtype = data[0][2].dtype)

        for idx, dt in enumerate(data):
            InpFeat[idx] = dt[0]
            IbmSpec[idx] = dt[1]
            RCnSpec[idx] = dt[2]

        RtDta = {'DegFeat': {'inpfeat':InpFeat},
                 'TarSpec': {'ibmspec': IbmSpec, 'rcnspec': RCnSpec},
                }
        return RtDta

    elif mode == 'valid':
        _, frms, nydims = data[0][0].shape
        _, frms, rcdims = data[0][1].shape
        _, frms, cndims = data[0][2].shape

        InpFeat = torch.zeros(bthsize, frms, nydims, dtype = data[0][0].dtype)
        IbmSpec = torch.zeros(bthsize, frms, rcdims, dtype = data[0][1].dtype)
        RCnSpec = torch.zeros(bthsize, frms, cndims, dtype = data[0][2].dtype)

        for idx, dt in enumerate(data):
            InpFeat[idx] = dt[0]
            IbmSpec[idx] = dt[1]
            RCnSpec[idx] = dt[2]

        RtDta = {'DegFeat': {'inpfeat':InpFeat},
                 'TarSpec': {'ibmspec': IbmSpec, 'rcnspec': RCnSpec},
                }
        return RtDta

def cr_f(path):
    if not os.path.isdir(path):
        os.mkdir(path)

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    data_path = 'data'

    # dataset = 'celp'
    # dataset = 'melp'
    # dataset = 'adpcm16kbps'
    dataset = 'fusion'
        
    datasets = ['celp', 'melp', 'adpcm16kbps']

    # model path
    ModelPath = "model"
    cr_f(ModelPath)
    ModelPath = os.path.join(ModelPath, "two_stage")
    cr_f(ModelPath)
    ModelPath = os.path.join(ModelPath, f"wav2ibm")
    # ModelPath = os.path.join(ModelPath, f"wav2spec")
    cr_f(ModelPath)
    CkpotPath = os.path.join(ModelPath, "checkpoint")
    cr_f(CkpotPath)

    # clean data
    TrCleanVoiceFolder = "./data/wsj_8k/train/"
    VaCleanVoiceFolder = "./data/wsj_8k/valid/"

    # noise data
    if dataset == 'fusion' :
        TrNoisyVoiceFolder = []
        VaNoisyVoiceFolder = []
        for ds in datasets :
            TrNoisyVoiceFolder.append(f"./data/{ds}/train/")
            VaNoisyVoiceFolder.append(f"./data/{ds}/valid/")
    else :
        TrNoisyVoiceFolder = f"./data/{dataset}/train/"
        VaNoisyVoiceFolder = f"./data/{dataset}/valid/"

    encoder_dim = 128
    num_encoder_layers = 5
    num_attention_heads = 8
    Epochs = 120
    BatchSize = 12

    from load_data import C_Dataset
    TrDataLdr = {
                'tr' : DataLoader(C_Dataset(TrCleanVoiceFolder, TrNoisyVoiceFolder, 'train'), batch_size=BatchSize, shuffle=True, num_workers=0, collate_fn = collate_fn),
                 'va' : DataLoader(C_Dataset(VaCleanVoiceFolder, VaNoisyVoiceFolder, 'valid'), batch_size=1, num_workers=0, collate_fn = collate_fn)
                }
    # training
    # from conformer_train_ibm import conformer_tr
    from conformer_train import conformer_tr
    conformer_tr = conformer_tr(MdlPth=ModelPath, CPTPth=CkpotPath, encoder_dim=encoder_dim, num_encoder_layers=num_encoder_layers, num_attention_heads=num_attention_heads)
    conformer_tr.train(Epochs,TrDataLdr)