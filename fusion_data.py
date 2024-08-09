from os.path import join, isdir
from os import system, listdir, mkdir
def cr_f(path):
    if not isdir(path):
        mkdir(path)

if __name__ == '__main__':
	data_path = 'data'
	datasets = ['celp_fl160_ns5_lo10','melp','adpcm_bit2_small','adpcm_bit2_tiny']
	data_classes = ['train', 'valid']

	save_path = join(data_path,'fusion')
	cr_f(save_path)
	for data_class in data_classes :
		cr_f(join(save_path,data_class))
		for dataset in datasets :
			wav_path = join(data_path, dataset, data_class)
			for wav_name in listdir(wav_path):

				# print(f'cp {join(wav_path, wav_name)} {join(save_path, data_class, wav_name.split(".")[0]+"-"+dataset+"."+wav_name.split(".")[1])}')
				system(f'cp {join(wav_path, wav_name)} {join(save_path, data_class, wav_name.split(".")[0]+"-"+dataset+"."+wav_name.split(".")[1])}')

