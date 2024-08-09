import numpy as np
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import join, isdir
def cr_f(path):
    if not isdir(path):
        mkdir(path)

np_path = 'npz_save'
# model_class = 'two_stage'
model_class = 'ibm'

# file_path = 'ibm'
# file_path = 'wav2spec'
# file_path = 'wav2ibm'
# file_paths = ['wav2spec', 'wav2ibm']
file_paths = ['ibm2']
# img_path = 'figure'


keys = ['cln_mdl', 'ibm_mdl', 'ibm', 'cln', 'noy']
# keys = ['ibm_mdl', 'ibm', 'cln', 'noy']

# file_path = 'dl_test'
# keys = ['noy', 'ibm', 'cln']
# img_path = 'figure'

for file_path in file_paths :
    img_path = 'figure_'+file_path
    cr_f(img_path)

    file_path = join(np_path, model_class, file_path)
    # 讀取每個檔案並取得 'ibm'、'cln' 和 'noy' 數據
    count = 0
    for file_name in listdir(file_path):
        data = np.load(join(file_path, file_name))
        for key in keys :
            print(key)
            spec = data[key]
            if key in ['ibm_mdl', 'cln_mdl'] :
                spec = spec.T

            print(f'{key}.shape:',spec.shape, type(spec))

            # # 繪製 IBM 數據
            plt.figure(figsize=(10, 4))
            plt.imshow(spec, aspect='auto', origin='lower')  # 使用 .T 轉置數據
            plt.colorbar(label='IBM Value')
            plt.title(f'{key} Data from {file_name}')
            plt.xlabel('Time')
            plt.ylabel('Frequency')
            plt.savefig(f"{img_path}/{file_name.split('.')[0]}_{key}.png")
            plt.close()

        count += 1
        if count > 5 :
            break