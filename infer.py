#!/usr/bin/env python
# coding: utf-8
import os
import torch

import numpy as np

from scipy.io.wavfile import read as audioread
from scipy.io.wavfile import write as audiowrite
from os.path import join
from os.path import sep as filesep

from torch.nn import functional as F
from torch.utils.data import DataLoader

from pre_processing import wav2spec, spec2wav, ibm2wav, wav2ibm

def cr_f(path):
    if not os.path.isdir(path):
        os.mkdir(path)
# def get_ibm():w
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    BatchSize = 1
    Dtype = torch.float
    data_path = 'data'

    model_class = 'ibm'
    # model_class = 'two_stage'

    spec_class = 'ibm2'
    # spec_class = 'wav2ibm'
    # spec_class = 'wav2spec'

    # dataset = 'celp'
    # dataset = 'melp'
    # dataset = 'adpcm16kbps'
    # datasets = ['celp', 'melp', 'adpcm16kbps']
    datasets = ['adpcm16kbps']
    
    threshold = 0.7

    test_path = 'npz_save'
    cr_f(test_path)
    test_path = join(test_path, model_class)
    cr_f(test_path)
    test_path = join(test_path, spec_class)
    cr_f(test_path)
    
    for dataset in datasets :
        TsCleanVoiceFolder = join(data_path, 'wsj_8k', 'test')
        TsNoisyVoiceFolder = join(data_path, dataset, 'test')
        TsEnhacVoiceFolder = join(data_path, f'em_{model_class}')
        cr_f(TsEnhacVoiceFolder)
        TsEnhacVoiceFolder = join(TsEnhacVoiceFolder, spec_class)
        cr_f(TsEnhacVoiceFolder)
        TsEnhacVoiceFolder = join(TsEnhacVoiceFolder, dataset)
        cr_f(TsEnhacVoiceFolder)

        ModelPath = join('model', model_class, spec_class)

        max_epc = -1
        for mdl_name in os.listdir(ModelPath) :
            # if 'mdl.pkl' in mdl_name and int(mdl_name.split('mdl.pkl')[0]) >max_epc :
            if 'mdl.pkl' in mdl_name  :
                max_epc = int(mdl_name.split('mdl.pkl')[0])
                #print(mdl_name.split('mdl.pkl')[0],'|loss:',torch.load(join(ModelPath,mdl_name))['loss'])
                print(mdl_name.split('mdl.pkl')[0],'|loss:',torch.load(join(ModelPath, mdl_name), map_location=torch.device('cpu'))['loss'])
        ModelPath = join(ModelPath, f"{max_epc}mdl.pkl")
        print('ModelPath:',ModelPath)

        CkpotPath = join('model', dataset, 'checkpoint')

        encoder_dim = 128
        num_encoder_layers = 5
        num_attention_heads = 8

        from conformer_train_ibm import conformer_tr
        # from conformer_train import conformer_tr
        conformer_tr = conformer_tr(MdlPth=ModelPath, CPTPth=CkpotPath, encoder_dim=encoder_dim, num_encoder_layers=num_encoder_layers, num_attention_heads=num_attention_heads)
        conformer_tr._load_Mdl()
        c_paths = []
        n_paths = []
        e_paths = []

        NoyWavData = os.listdir(TsNoisyVoiceFolder)
        for noywavfile in NoyWavData:
            c_paths.append(join(TsCleanVoiceFolder,noywavfile))
            n_paths.append(join(TsNoisyVoiceFolder,noywavfile))
            e_paths.append(join(TsEnhacVoiceFolder,noywavfile))

        for clnfile, noyfile, enhfile in zip(c_paths, n_paths, e_paths):
            (fs,clnData) = audioread(clnfile)
            (fs,nsyData) = audioread(noyfile)

            if len(clnData) > len(nsyData) :
                NoeData=nsyData-clnData[:len(nsyData)]
            else :
                NoeData=nsyData[:len(clnData)]-clnData

            # # load audio
            # if spec_class == 'wav2ibm' :
            ClnSpec, ClnPhas = wav2ibm(clnData/(2.0**15),returnPhase = True)
            NoeSpec, NoePhas = wav2ibm(NoeData/(2.0**15),returnPhase = True)
            TmpSpec, NoyPhas = wav2ibm(nsyData/(2.0**15),returnPhase = True)
            # elif spec_class == 'wav2spec' :
            # ClnSpec, ClnPhas = wav2spec(clnData/(2.0**15),returnPhase = True)
            # NoeSpec, NoePhas = wav2spec(NoeData/(2.0**15),returnPhase = True)
            # TmpSpec, NoyPhas = wav2spec(nsyData/(2.0**15),returnPhase = True)

            NoyFeat = torch.from_numpy(np.expand_dims(TmpSpec.T, axis=0)).type(torch.float)

            # real ibm
            ibmSpec = ClnSpec >= NoeSpec

            # # ibm infer
            opt_ibm = conformer_tr.eval(NoyFeat)
            opt_ibm = opt_ibm[0].detach().cpu().type(Dtype)
            
            # binary
            ibm_mdl = np.where(opt_ibm >= threshold, 1, 0).astype(np.float32)

            # ibm_audio = ibm_spec + noy_phas
            enhWav = ibm2wav(ibm_mdl.T, NoyPhas[:, :opt_ibm.shape[0]])
            audiowrite(enhfile.split('.')[0]+'_ibm.wav', fs, np.array(enhWav * (2**15)).astype('int16'))
            (fs,testData) = audioread(enhfile.split('.')[0]+'_ibm.wav')

            # em_audio = mdl_ibm_spec*noy_spec + noy_phas
            cln_mdl = np.multiply(NoyFeat[0], opt_ibm)

            enhWav2 = spec2wav(cln_mdl.T, NoyPhas[:, :cln_mdl.shape[0]])
            audiowrite(enhfile, fs, np.round(np.array(enhWav2 * (2**15))).astype('int16'))

            np.savez(test_path+'/'+enhfile.rsplit('/',1)[1].split('.')[0],cln_mdl=cln_mdl, ibm_mdl=ibm_mdl, ibm=ibmSpec, noy=TmpSpec, cln=ClnSpec)

            # # two_stage infer
            # opt_ibm, opt_cln = conformer_tr.eval(NoyFeat)
            # opt_ibm = opt_ibm[0].detach().cpu().type(Dtype)
            # opt_cln = opt_cln[0].detach().cpu().type(Dtype)
            # print('mdl_ibm.shape:',opt_ibm.shape, '|mdl_cln.shape:',opt_cln.shape)
            # # binary
            # ibm_mdl = np.where(opt_ibm >= threshold, 1, 0).astype(np.float32)

            # # em_audio = mdl_spec + noy_phas
            # if spec_class == 'wav2ibm' :
            #     enhWav = ibm2wav(opt_cln.T, NoyPhas[:, :opt_cln.shape[0]])
            #     audiowrite(enhfile, fs, np.round(np.array(enhWav * (2**15))).astype('int16'))
            #     # em_audio = mdl_spec + noy_phas
            #     enhWav = ibm2wav(ibm_mdl.T, NoyPhas[:, :opt_cln.shape[0]])
            #     audiowrite(enhfile.split('.')[0]+'_ibm.wav', fs, np.round(np.array(enhWav * (2**15))).astype('int16'))
            # elif spec_class == 'wav2spec' :
            #     enhWav = spec2wav(opt_cln.T, NoyPhas[:, :opt_cln.shape[0]])
            #     audiowrite(enhfile, fs, np.round(np.array(enhWav * (2**15))).astype('int16'))
            #     # em_audio = mdl_spec + noy_phas
            #     enhWav = spec2wav(ibm_mdl.T, NoyPhas[:, :opt_cln.shape[0]])
            #     audiowrite(enhfile.split('.')[0]+'_ibm.wav', fs, np.round(np.array(enhWav * (2**15))).astype('int16'))
            # # save npz
            # np.savez(test_path+'/'+enhfile.rsplit('/',1)[1].split('.')[0], cln_mdl=opt_cln, ibm_mdl=ibm_mdl, ibm=ibmSpec, noy=TmpSpec, cln=ClnSpec)
            
