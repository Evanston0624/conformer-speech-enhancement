#!/usr/bin/env python
# coding: utf-8

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

def cr_f(path):
    if not os.path.isdir(path):
        os.mkdir(path)

def example():
    BatchSize = 1
    Epochs = 10

    encoder_dim = 128
    num_encoder_layers = 5
    num_attention_heads = 8

    # save eval path
    eval_path = './eval/'
    cr_f(eval_path)
    eval_file = f'{eval_path}ed{encoder_dim}_el{num_encoder_layers}_ah{num_attention_heads}.txt'

    # data
    TsCleanVoiceFolder = "./data/clean_audio/test_set/"
    TsNoisyVoiceFolder = "./data/noisy_audio/test_set/"
    TsEnhacVoiceFolder = "./data/enhance_audio/"

    Dtype = torch.float
    nyclsnr=0
    ehclsnr=0

    nyclsqu=0
    enclsqu=0

    nyclsin=0
    enclsin=0

    NoyWavData = os.listdir(TsEnhacVoiceFolder)

    for noywavfile in NoyWavData:
        clnfilenme = noywavfile.split(filesep)[-1].split('-')[0]
        clnfilenme = clnfilenme.split('_')[0]
        fs,clndata = audioread(join(TsCleanVoiceFolder,f'{clnfilenme}'))
        _,noydata = audioread(join(TsNoisyVoiceFolder,noywavfile))
        _,enhdata = audioread(join(data_file,noywavfile))

        minpoints = np.min([clndata.shape[0],noydata.shape[0],enhdata.shape[0]])
        # print(minpoints)
        nyclsnr += si_sdr(noydata[0:minpoints]*1.0, clndata[0:minpoints]*1.0)
        ehclsnr += si_sdr(enhdata[0:minpoints]*1.0, clndata[0:minpoints]*1.0)

        nyclsqu += pesq(fs, clndata[0:minpoints]*1.0, noydata[0:minpoints]*1.0, 'nb')
        enclsqu += pesq(fs, clndata[0:minpoints]*1.0, enhdata[0:minpoints]*1.0, 'nb')

        nyclsin += stoi(clndata[0:minpoints]*1.0, noydata[0:minpoints]*1.0, fs)
        enclsin += stoi(clndata[0:minpoints]*1.0, enhdata[0:minpoints]*1.0, fs)

    nyclsnr = nyclsnr/len(NoyWavData)
    ehclsnr = ehclsnr/len(NoyWavData)

    nyclsqu = nyclsqu/len(NoyWavData)
    enclsqu = enclsqu/len(NoyWavData)

    nyclsin = nyclsin/len(NoyWavData)
    enclsin = enclsin/len(NoyWavData)

    with open(eval_file, 'w') as f :
        f.write(f'Averaged Noisy-Clean SNR: {nyclsnr}\t Averaged Enhanced-Clean SNR: {ehclsnr}\n')
        f.write(f'Averaged Noisy-Clean PESQ score: {nyclsqu}\t Averaged Enhanced-Clean PESQ score: {enclsqu}\n')
        f.write(f'Averaged Noisy-Clean STOI score: {nyclsin}\t Averaged Enhanced-Clean STOI score: {enclsin}\n')

if __name__ == '__main__':
    encoder_dim = 128
    num_encoder_layers = 5
    num_attention_heads = 8

    model_class = 'fusion'
    # datasets = ['celp']
    # datasets = ['melp']
    # datasets = ['adpcm16kbps']
    datasets = ['celp', 'melp', 'adpcm16kbps']

    for dataset in datasets :
        # save eval path
        eval_path = 'eval'
        cr_f(eval_path)
        eval_file = join(eval_path, f'{model_class}_{dataset}.txt')

        # data
        data_path = 'data'
        TsCleanVoiceFolder = join(data_path, 'wsj_8k', 'test')
        TsNoisyVoiceFolder = join(data_path, dataset, 'test')
        # TsEnhacVoiceFolder = join(data_path, f'em_{dataset}')
        TsEnhacVoiceFolder = join(data_path, f'em_{model_class}', dataset)

        nyclsnr=0
        ehclsnr=0

        nyclsqu=0
        enclsqu=0

        nyclsin=0
        enclsin=0

        NoyWavData = os.listdir(TsEnhacVoiceFolder)

        for noywavfile in NoyWavData:
            clnfilenme = noywavfile.split(filesep)[-1].split('-')[0]
            clnfilenme = clnfilenme.split('_')[0]
            fs,clndata = audioread(join(TsCleanVoiceFolder,f'{clnfilenme}'))
            _,noydata = audioread(join(TsNoisyVoiceFolder,noywavfile))
            _,enhdata = audioread(join(TsEnhacVoiceFolder,noywavfile))

            minpoints = np.min([clndata.shape[0],noydata.shape[0],enhdata.shape[0]])
            # print(minpoints)
            nyclsnr += si_sdr(noydata[0:minpoints]*1.0, clndata[0:minpoints]*1.0)
            ehclsnr += si_sdr(enhdata[0:minpoints]*1.0, clndata[0:minpoints]*1.0)

            nyclsqu += pesq(fs, clndata[0:minpoints]*1.0, noydata[0:minpoints]*1.0, 'nb')
            enclsqu += pesq(fs, clndata[0:minpoints]*1.0, enhdata[0:minpoints]*1.0, 'nb')

            nyclsin += stoi(clndata[0:minpoints]*1.0, noydata[0:minpoints]*1.0, fs)
            enclsin += stoi(clndata[0:minpoints]*1.0, enhdata[0:minpoints]*1.0, fs)

        nyclsnr = nyclsnr/len(NoyWavData)
        ehclsnr = ehclsnr/len(NoyWavData)

        nyclsqu = nyclsqu/len(NoyWavData)
        enclsqu = enclsqu/len(NoyWavData)

        nyclsin = nyclsin/len(NoyWavData)
        enclsin = enclsin/len(NoyWavData)

        with open(eval_file, 'w') as f :
            f.write(f'Averaged Noisy-Clean SNR: {nyclsnr}\t Averaged Enhanced-Clean SNR: {ehclsnr}\n')
            f.write(f'Averaged Noisy-Clean PESQ score: {nyclsqu}\t Averaged Enhanced-Clean PESQ score: {enclsqu}\n')
            f.write(f'Averaged Noisy-Clean STOI score: {nyclsin}\t Averaged Enhanced-Clean STOI score: {enclsin}\n')