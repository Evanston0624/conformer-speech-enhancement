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
    epsilon = np.finfo(float).eps
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
    BatchSize = 1
    Epochs = 10
    # Layer_Size_List = [100, 100, 100, 100]
    # SNR = np.arange(-3,11,1)

    # save eval path
    eval_path = './eval/'
    cr_f(eval_path)
    eval_path += 'spec/'
    cr_f(eval_path)


    # data
    TrCleanVoiceFolder = "./data/wsj_8k_mv/train/"
    TsCleanVoiceFolder = "./data/wsj_8k_mv/test/"
    VaCleanVoiceFolder = "./data/wsj_8k_mv/valid/"
    for i in range (3, 10) :
        TrNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/train/"

        TsNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/test/"

        VaNoisyVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/valid/"

        TsEnhacVoiceFolder = f"./data/L{i}_B4_P7_G2_mv/enhance/conformer/"

        # model
        ModelPath = f"./model/conformer/L{i}_B4_P7_G2/mdl.pkl"
        CkpotPath = f"./model/conformer/checkpoint/L{i}_B4_P7_G2"

        Dtype = torch.float

        nyclsnr=0
        ehclsnr=0

        nyclsqu=0
        enclsqu=0
        nyclsin=0
        enclsin=0

        NoyWavData = os.listdir(TsEnhacVoiceFolder)
        eval_file = f'{eval_path}L{i}_B4_P7_G2_list.txt'
        with open(eval_file, 'w') as f :
            f.write('file_name\tSNR\tPESQ\tSTOI\n')
            for noywavfile in NoyWavData:
                # print(noywavfile)
                clnfilenme = noywavfile.split(filesep)[-1].split('-')[0]
                fs,clndata = audioread(join(TsCleanVoiceFolder,f'{clnfilenme}'))
                _,noydata = audioread(join(TsNoisyVoiceFolder,noywavfile))
                _,enhdata = audioread(join(TsEnhacVoiceFolder,noywavfile))

                minpoints = np.min([clndata.shape[0],noydata.shape[0],enhdata.shape[0]])
                # print(minpoints)
                nyclsnr += si_sdr(noydata[0:minpoints]*1.0, clndata[0:minpoints]*1.0)
                ehclsnr += si_sdr(enhdata[0:minpoints]*1.0, clndata[0:minpoints]*1.0)
                # print('clndata:',len(clndata))
                # print('noydata:',len(noydata))
                # print('enhdata:',len(enhdata))

                nyclsqu += pesq(fs, clndata[0:minpoints]*1.0, noydata[0:minpoints]*1.0, 'nb')
                enclsqu += pesq(fs, clndata[0:minpoints]*1.0, enhdata[0:minpoints]*1.0, 'nb')


                nyclsin += stoi(clndata[0:minpoints]*1.0, noydata[0:minpoints]*1.0, fs)
                enclsin += stoi(clndata[0:minpoints]*1.0, enhdata[0:minpoints]*1.0, fs)

                f.write(f'{clnfilenme}\t{ehclsnr}\t{enclsqu}\t{enclsin}\n')

        nyclsnr = nyclsnr/len(NoyWavData)
        ehclsnr = ehclsnr/len(NoyWavData)

        nyclsqu = nyclsqu/len(NoyWavData)
        enclsqu = enclsqu/len(NoyWavData)

        nyclsin = nyclsin/len(NoyWavData)
        enclsin = enclsin/len(NoyWavData)

        eval_file = f'{eval_path}L{i}_B4_P7_G2.txt'

        with open(eval_file, 'w') as f :
            f.write(f'Averaged Noisy-Clean SNR: {nyclsnr}\t Averaged Enhanced-Clean SNR: {ehclsnr}\n')
            f.write(f'Averaged Noisy-Clean PESQ score: {nyclsqu}\t Averaged Enhanced-Clean PESQ score: {enclsqu}\n')
            f.write(f'Averaged Noisy-Clean STOI score: {nyclsin}\t Averaged Enhanced-Clean STOI score: {enclsin}\n')
        # print(f'Averaged Noisy-Clean SNR: {nyclsnr}\t Averaged Enhanced-Clean SNR: {ehclsnr}')
        # print(f'Averaged Noisy-Clean PESQ score: {nyclsqu}\t Averaged Enhanced-Clean PESQ score: {enclsqu}')
        # print(f'Averaged Noisy-Clean STOI score: {nyclsin}\t Averaged Enhanced-Clean STOI score: {enclsin}')