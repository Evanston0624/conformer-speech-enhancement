# conformer speech enhancement
## Abstract
This project uses the Conformer architecture from:  
https://github.com/sooftware/conformer  
This project modifies the Conformer CNN module to implement speech enhancement based on Conformer.

## Working environment and dependent projects
OS : ubuntu 22.04  
GPU : GeForce RTX 3080 or 4090,  (you need to adjust the batch size)  
CUDA version : cuda-12.1.0 +
pytorch version : 2.1.1  
python version : 3.9+  

You can use conda :
```
conda create --name myenv python=3.9
conda install nvidia/label/cuda-12.1.0::cuda
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

Use git to pull this project\
```
git clone https://github.com/Evanston0624/conformer-speech-enhancement
cd conformer-speech-enhancement
```

You need to install pip\
Next, on Linux you can run:
```
pip install -r requirements.txt
```
If you're using conda as your package manager, please refer to requirements.sh for installing dependencies.  
## dataset
1. You need a clean speech dataset and a noise dataset.
2. This project was tested with a sample rate of 8000. If your sample rate is different, you may need to adjust some parameters to achieve better results.  

Mix the above data into noisy speech data. The names of the noisy speech data should be the same as the clean data, such as:  
clean_audio/1.wav, 2.wav, ..., n.wav  
noisy_audio/1.wav, 2.wav, ..., n.wav  
Next, split clean_audio and noisy_audio into train, valid, and test sets.  
The train set is used to train the model.  
The valid set evaluates the model performance between each epoch.  
The test set generates enhanced results from the trained model and calculates SNR, PESQ, and STOI.  

In conclusion, you will need the following six folders, paired two by two:  
clean_audio/train_set  
clean_audio/valid_set  
clean_audio/test_set  
noisy_audio/train_set  
noisy_audio/valid_set  
noisy_audio/test_set  

After training, use the data in noisy_audio/test_set to generate enhanced data:  
enhance_audio/  

## training
The following code is included in train.py. You can directly modify train.py.  
First, you need to create a path to save the model:  
```
# model
ModelPath = "./model"
cr_f(ModelPath)
ModelPath = "./model/conformer/"
cr_f(ModelPath)
CkpotPath = "./model/conformer/checkpoint/"
cr_f(CkpotPath)
```
Next, change the file paths to your paths:  
```
TrCleanVoiceFolder = "./data/clean_audio/train_set/"
VaCleanVoiceFolder = "./data/clean_audio/valid_set/"
TrNoisyVoiceFolder = "./data/noisy_audio/train_set/"
VaNoisyVoiceFolder = "./data/noisy_audio/valid_set/"
```

You can set the Conformer hyperparameters and training parameters:  
```
encoder_dim = 128
num_encoder_layers = 5
num_attention_heads = 8
Epochs = 150
BatchSize = 35
```

Create dataloader:  
```
from load_data import C_Dataset
TrDataLdr = {
            'tr' : DataLoader(C_Dataset(TrCleanVoiceFolder, TrNoisyVoiceFolder, 'train'), batch_size=BatchSize, shuffle=True, num_workers=0, collate_fn = collate_fn),
             'va' : DataLoader(C_Dataset(VaCleanVoiceFolder, VaNoisyVoiceFolder, 'valid'), batch_size=1, num_workers=0, collate_fn = collate_fn)
            }
```

Train the model:  
```
# training
from conformer_train import conformer_tr
conformer_tr = conformer_tr(MdlPth=ModelPath, CPTPth=CkpotPath, encoder_dim=encoder_dim, num_encoder_layers=num_encoder_layers, num_attention_heads=num_attention_heads)
conformer_tr.train(Epochs,TrDataLdr)
```
## infer
This part of the code is in infer.py. You can modify it directly.  
Make sure the model path and model hyperparameters are consistent with the settings in train.py.  

## testing
The following code is included in eval.py. You can directly modify eval.py.  
Set the path and file for saving evaluation results:  
```
# save eval path
eval_path = './eval/'
cr_f(eval_path)
eval_file = f'{eval_path}ed{encoder_dim}_el{num_encoder_layers}_ah{num_attention_heads}.txt'
```
The evaluation metrics used in this project are: si_sdr, pesq, and stoi.  

## to-do list
1. Traditional Chinese README
2. The current code still needs to be streamlined, as there are many identical functions existing in different .py files.  
