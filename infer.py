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
    # Layer_Size_List = [100, 100, 100, 100]
    # SNR = np.arange(-3,11,1)
    
    TrCleanVoiceFolder = "./data/wsj_8k_mv/train/"
    TsCleanVoiceFolder = "./data/wsj_8k_mv/test/"
    VaCleanVoiceFolder = "./data/wsj_8k_mv/valid/"


    for i in range (3, 10) :
        count = 0
        TrNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/train/"

        TsNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/test/"

        VaNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/valid/"

        TsEnhacVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/enhance/"
        cr_f(TsEnhacVoiceFolder)
        cr_f(TsEnhacVoiceFolder+'conformer')
        # model
        ModelPath = f"./model/conformer/L{i}_B4_P7_G2/mdl.pkl"
        CkpotPath = f"./model/conformer/checkpoint/L{i}_B4_P7_G2"

        # ClnWavData = os.listdir(TrCleanVoiceFolder)01fo0317.wav
        # NoeWavData = os.listdir(TrNoiseSignlFolder)
        # Dtype = torch.float
        # # data loader
        # for clnwavfile in ClnWavData:
        #   for noewavfile in NoeWavData:
        #     for snrInd in SNR:

        #         nsywavname = f'{clnwavfile.split(".")[0]}-{noewavfile.split(".")[0]}-{snrInd}.wav'
        #         #mix_noise(join(TrCleanVoiceFolder,clnwavfile),join(TrNoiseSignlFolder,noewavfile),join(TrNoisyVoiceFolder,nsywavname),snrInd)
        # from load_data import C_Dataset
        # TrDataLdr = {
        #             'tr' : DataLoader(C_Dataset(TrCleanVoiceFolder, TrNoisyVoiceFolder, 'train'), batch_size=BatchSize, shuffle=True, num_workers=0, collate_fn = collate_fn),
        #              'va' : DataLoader(C_Dataset(TsCleanVoiceFolder, TsNoisyVoiceFolder, 'valid'), batch_size=1, num_workers=0, collate_fn = collate_fn)
        #             }

        # training
        from conformer_train import conformer_tr
        conformer_tr = conformer_tr(MdlPth=ModelPath, CPTPth=CkpotPath)
        conformer_tr._load_Mdl()

        n_paths = []
        c_paths = []
        e_paths = []

        NoyWavData = os.listdir(TsNoisyVoiceFolder)
        for noywavfile in NoyWavData:
            clnfilenme = noywavfile.split(filesep)[-1].split('_')[0]

            n_paths.append(join(TsNoisyVoiceFolder,noywavfile))

            e_paths.append(join(TsEnhacVoiceFolder,'conformer',noywavfile))

            c_paths.append(join(TrCleanVoiceFolder,f'{clnfilenme}.wav'))

        for noyfile, enhfile in zip(n_paths,e_paths):
            print('enhfile:',enhfile)
            (fs,nsyData) = audioread(noyfile)
            print('inp:fs:',fs,'|nsyData.shape:',nsyData.shape)

            TmpSpec, NoyPhas = wav2spec(nsyData/(2.0**15),returnPhase = True)
            NoyFeat = torch.from_numpy(np.expand_dims(TmpSpec.T, axis=0)).type(torch.float)

            mdlout = conformer_tr.eval(NoyFeat)

            enhWav = spec2wav(mdlout[0].detach().cpu().type(Dtype).T, NoyPhas[:, :mdlout[0].shape[0]])
            print('enhWav.shape:',enhWav.shape)

            audiowrite(enhfile, 8000, np.round(np.array(enhWav * (2**15))).astype('int16'))

            count += 1
        print(f"./data/L{i}_B4_P7_G2_mv/enhance/ data_count:{count}")