import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import librosa

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

# input: ? x 42
# output: ? x 768


class lstm2(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_size = 300, num_layers = 2, dropout = 0.1):
        super().__init__()
        self.lstm_list = nn.ModuleList([
        nn.LSTM(input_size = in_dim, hidden_size = out_dim,num_layers = num_layers,
            dropout = dropout
            )
        ])

    def forward(self, x):
        for module in self.lstm_list:
            x, (h_n,c_n) = module(x)
        return x

from train import *
from pytorch_lightning.callbacks import ModelCheckpoint
from train import get_args
PATH = './ckpt/ckpt.ckpt'
import os
from pathlib import Path
from torch import optim
from models.SpeechAutoEncoder import SpeechAutoEncoder
#import torch
from transformers import Data2VecAudioConfig,AutoConfig
from datasets import load_dataset
import datasets
#export PYTHONWARNINGS='ignore:semaphore_tracker:UserWarning'
import torchaudio

MODEL_NAME = 'SAE'
loss = torch.nn.SmoothL1Loss()
light_model = SpeechAutoEncoder
sampling_rate = 16000
base_path = Path(__file__).parent#.parent
# print(base_path)
# print(base_path/"cache"/"configs"/"hifigan"/"config.json")
_model_config = dict(
    encoder_config= None,
    pretrain_encoder_flag = True,
    decoder_config = AutoConfig.from_pretrained(base_path/"cache"/"models"/"hifigan",local_files_only=True, trust_remote_code=True,        
        upsample_rates=[2, 2, 2, 2, 2, 2, 5],
        upsample_initial_channel=768,
        upsample_kernel_sizes=[2, 2, 3, 3, 3, 3, 10],
        model_in_dim=768,
        sampling_rate=sampling_rate),
    optimizer=optim.AdamW,
    loss_fn=loss,
    lr=[0.0000001,0.001],
    sampling_rate = sampling_rate,
)

import torchaudio.transforms as T

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
sample_rate = 16000

mel_spectrogram = T.MelSpectrogram(
    sample_rate=sample_rate,
    n_fft=n_fft,
    win_length=win_length,
    hop_length=hop_length,
    center=True,
    pad_mode="reflect",
    power=2.0,
    norm="slaney",
    onesided=True,
    n_mels=n_mels,
    mel_scale="htk",
)

def power2db(mat):
    return torch.log10(mat)*10

def db2power(mat):
    return 10**torch.multiply(mat,0.1)

class lstm_2_and_SAE(nn.Module):
    def __init__(self,in_dim,out_dim,is_notebook=False,fname_lstm=None,decoding_target_layer=None):
        super().__init__()
        if fname_lstm is not None:
            self.LSTM = torch.load(fname_lstm,map_location = device)
        else:
            self.LSTM = lstm2(in_dim,out_dim)
        self.decoding_target_layer = decoding_target_layer
        os.environ['HF_DATASETS_OFFLINE']="1"
        os.environ['TRANSFORMERS_OFFLINE']="1"
        args = get_args(is_notebook = is_notebook)
        config = importlib.import_module(args.config_path)
        model_config = config.model_config
        train_config = config.train_config
        dataset_config = config.dataset_config
        #model = config.light_model(**model_config)
        SAE = config.light_model.load_from_checkpoint(PATH,
        encoder_config=None,pretrain_encoder_flag=True,
        decoder_config=AutoConfig.from_pretrained('./cache/models/hifigan',local_files_only=True,trust_remote_code=True,
        upsample_rates=[2, 2, 2, 2, 2, 2, 5],
        upsample_initial_channel=768,
        upsample_kernel_sizes=[2, 2, 3, 3, 3, 3, 10],
        model_in_dim=768,
        sampling_rate=sampling_rate),
        loss_fn=torch.nn.SmoothL1Loss(),optimizer=optim.AdamW)
        if decoding_target_layer is not None:
            self.encoder = SAE.encoder
            for p in self.encoder.parameters():
                p.requires_grad = False
        self.decoder = SAE.decoder
        for p in self.decoder.parameters():
            p.requires_grad = False
        self.mel_spectrogram = mel_spectrogram

    def power2db(self,mat,eps = 1e-10):
        '''only one matrix'''
        return 10*torch.log10(torch.maximum(torch.ones_like(mat)*eps,mat))

    def forward(self,x,mel_en = True,return_lat12 = False,return_wf = False):
        x = self.LSTM(x)
        x = x.unsqueeze(0)
        if self.decoding_target_layer is not None:
            NUM_TRANSFORMER_BLOCKS = 12
            for i in range(self.decoding_target_layer,NUM_TRANSFORMER_BLOCKS):
                #print(i,x.shape)
                x = self.encoder.encoder.layers[i](x)[0]

        #print('lstmed:',x.shape)
        #x = x.unsqueeze(1).transpose(-2,-1)#.unsqueeze(0)#.transpose(0,1)
        lat12 = x.transpose(-2,-1)
        x = self.decoder(lat12)
        wf = x
        if return_wf == True:
            return x
        x = self.mel_spectrogram(x)
        #print('meled',x.shape)
        if mel_en == True:
            x = self.power2db(x)
        if return_lat12 == False:
            return {'mel':x,'wf':wf}
        else:
            return {'mel':x,'latent_space':lat12,'wf':wf}
#beginning: torch.Size([72, 42])
#lstmed: torch.Size([72, 768])
#begin_decoder torch.Size([1, 768, 72])
#after_decoder torch.Size([1, 1, 23120])
#meled torch.Size([1, 1, 128, 46])
# can to device

if __name__ == '__main__':
    t = MLP_3_and_SAE(10,10)
    print(t)#OK
