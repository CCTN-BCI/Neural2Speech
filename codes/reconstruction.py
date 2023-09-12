import torchaudio.transforms as T

n_fft = 1024
win_length = None
hop_length = 512
n_mels = 128
sample_rate = 16000
'''
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
'''
mel_basis = {}
hann_window = {}


def mel_spectrogram(y, n_fft=1024, num_mels=128, sampling_rate=16000, hop_size=512, win_size=None, fmin=0, fmax=8000, center=False):
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global mel_basis, hann_window
    if fmax not in mel_basis:
        mel = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[str(fmax)+'_'+str(y.device)] = torch.from_numpy(mel).float().to(y.device)
        hann_window[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[str(y.device)],
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis[str(fmax)+'_'+str(y.device)], spec)
    spec = spectral_normalize_torch(spec)

    return spec


def power2db(mat):
    return torch.log10(mat)*10

def db2power(mat):
    return 10**torch.multiply(mat,0.1)

import matplotlib.pyplot as plt
def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show(block=False)
    
import os

from numpy import dtype
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import time 
import pickle
import numpy as np

import temporal_receptive_field as trf

#subjects_set = ['EC183']
#subject_name = 'EC53'
subjects_set = ['EC183','EC53','EC75','EC85','EC143','EC157','EC182','EC186','EC193','EC196']# all the same as EC183 except EC124
subjects_set = ['EC183']
task = 'english'
'''
feature_names = ['fs_ext', 'fs_proj', 'encoder0', 'encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5', 
                 'encoder6', 'encoder7', 'encoder8', 'encoder9', 'encoder10', 'encoder11', 'encoder12', 
                 'hg', 'an', 'ic', 'spectrogram', 'intensity', 'bin_pitch', 'abs_pitch', 'rel_pitch', 
                 'pitch_change', 'phonetics', 'peakrate', 'onset'] 
'''
feature_names = ['latent_last_encoder','hg']
'''
nn_features = ['fs_ext', 'fs_proj', 'encoder0', 'encoder1', 'encoder2', 'encoder3', 'encoder4', 'encoder5', 
                 'encoder6', 'encoder7', 'encoder8', 'encoder9', 'encoder10', 'encoder11', 'encoder12']
'''
nn_features = ['latent_last_encoder']

import Dataset_fin_multi as Dataset_fin
from torch.utils.data import DataLoader
from model_lstm_and_SAE import *

from torch_stoi import NegSTOILoss

sample_rate = 16000
loss_stoi = NegSTOILoss(sample_rate=sample_rate).to(device)
use_stoi = False
return_wf = False
mel_spectrogram = mel_spectrogram.to(device)

fname_keyfeat = './cache/key_dimensions'
with open(fname_keyfeat,'rb') as f:
    all_key_dimensions = pickle.load(f)['key_dimensions']

for subject_name in subjects_set:
    print('-------\nreconstructing %s' % subject_name)
    #model_dir = './cache/deep_supervision/mel_en_only/'
    model_dir = './cache/deep_supervision/deep_loss/'
    #model_name = 'epoch160,valid_loss=12.522654'
    begin_epoch = 0
    #begin_epoch = int(model_name.split(',')[0][5:])
    #print(begin_epoch)
    #return
    BATCH_SIZE = 1

    #fname_hg_lat = './cache/SAE_res/Concat_SAE_supervised_all_features_EC183'
    fname_hg_lat = './cache_on_subjects/%s/Concat_SAE_supervised_all_features' % subject_name
    fname_wf = './cache/SAE_res/all_wfs_timit_after_SAE'
    fname_mels = './cache/SAE_res/all_mels_specs_timit_after_SAE'
    fname_mlp = './mlp_reconstruction/cache/epoch6,loss=0.005070'
    fname_timepoint = './cache/timepoints_EC183'
    fname_electrodes = './cache/electrodes.mat'

    TrainSet = Dataset_fin.Dataset_fin(fname_hg_lat = fname_hg_lat, fname_wf = fname_wf,fname_mels = fname_mels,subject_name = subject_name,
                               fname_electrodes = fname_electrodes, fname_timepoint = fname_timepoint, train=True,layer=12,return_wf = return_wf)
    TestSet = Dataset_fin.Dataset_fin(fname_hg_lat = fname_hg_lat, fname_wf = fname_wf,fname_mels = fname_mels,subject_name = subject_name,
                               fname_electrodes = fname_electrodes, fname_timepoint = fname_timepoint, train=False,layer=12,return_wf = return_wf)
    TrainLoader = DataLoader(TrainSet, batch_size=BATCH_SIZE,shuffle=False)#, num_workers=8, pin_memory=True )
    TestLoader = DataLoader(TestSet, batch_size=BATCH_SIZE, shuffle=False)#, num_workers=8,pin_memory=True)

    #model

    in_dim = len(TrainSet.all_electrodes[subject_name][0])#TMD
    out_dim = 768
    #print(device)
    model = lstm_2_and_SAE(in_dim=in_dim, out_dim=out_dim, is_notebook=False, fname_lstm=None).to(device)
    #model = model.to(device)
    #model = torch.load(model_dir + 'EC186_epoch549,valid_loss=64.519531',map_location = device)

    #optimization
    loss_function = nn.SmoothL1Loss()
    #loss_function = nn.MSELoss()
    #loss_function = loss_stoi
    LR = 3e-3# 0.0001
    momentum = 0.8
    #optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = LR, betas = (0.9,0.999), eps = 1e-08,weight_decay = 0.001)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, momentum = momentum)
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min')
    #weight_decay = L2 normalization

    #testsample = TrainSet[5]

    #print(model(testsample[0]).shape)# 1 1 128 46#OK
    #testsample = TrainSet[5]
    #print(testsample[0].shape,testsample[1].shape)#(72, 42) torch.Size([23120])#OK

    print('start training...')

    #train and validation
    EPOCHS = 1451#1000
    EPOCH_CHECKPOINT = 90
    train_loss = []
    valid_loss = []
    ret_loss = 0

    from copy import deepcopy
    def get_diffloss(output,label):
        '''
        get the differentiate matrix loss
        hint: output and label share the same shape, here is batch_size*channel*window*time e.g.(1,1,128,52)
        
        '''
        global loss_function
        diff_output = torch.zeros_like(output)
        diff_label = torch.zeros_like(label)
        for i in range(diff_label.shape[-1]):
            diff_output[:,:,:,i] = (output[:,:,:,i] - output[:,:,:,i - 1])
            diff_label[:,:,:,i] = (label[:,:,:,i] - label[:,:,:,i - 1])
        for i in range(diff_label.shape[-2]):
            diff_output[:,:,i,:] = diff_output[:,:,i,:] + (output[:,:,i,:] - output[:,:,i - 1,:])
            diff_label[:,:,i,:] = diff_label[:,:,i,:] + (label[:,:,i,:] - label[:,:,i - 1,:])
        return loss_function(diff_output,diff_label)

    def get_matloss(output,label,
        fname_keyfeat = './cache/key_dimensions',lat_layer = 0):
        '''
        get the differentiate matrix loss
        hint: output and label share the same shape, here is batch_size*channel*window*time e.g.(1,768,86)
        '''
        global loss_function
        global all_key_dimensions
        key_dimensions = all_key_dimensions[lat_layer]
        loss = loss_function(output[:,0,:],label[:,0,:]) * 0# just need a loss object!!!
        for d in key_dimensions:
            loss = loss + loss_function(output[:,d,:],label[:,d,:])
        return loss



    def getloss(dict_output,dict_label,
        brute_compare = False, 
        lambda_encoder12 = 0,
        lambda_diff = 0.1):
        '''
        input: two dicts with the same keys
        here the dicts are: 'mel' and 'latent_space'
        'mel' is the power mel-spectrogram
        'latent_space' is the encoder12 layer in the previous autoencoder

        output: a loss.
        '''
        '''
        with open(fname,'rb') as f:
            dict_keydimensions = pickle.load(f)['key_dinemsions'][0]
        '''
        global loss_function
        if brute_compare == True:
            return loss_function(dict_output['mel'],dict_label['mel'])
        else:
            #print(dict_output[0].shape,dict_output[1].shape)#torch.Size([1, 1, 128, 52]) torch.Size([1, 768, 82])
            #print(dict_output[0].shape,dict_label['mel'].shape)
            #print(dict_output[1].shape,dict_label['latent_space'].shape)
            #print(dict_output.keys())
            mel_loss = loss_function(dict_output['mel'],dict_label['mel'])
            #lat_loss = get_matloss(dict_output['latent_space'],dict_label['latent_space'])
            diff_loss = get_diffloss(dict_output['mel'],dict_label['mel'])
            #print(mel_loss,lat_loss * lambda_encoder12 ,diff_loss * lambda_diff)
            #tensor(7.5423, grad_fn=<SmoothL1LossBackward0>)
            #tensor(0.1189, grad_fn=<SmoothL1LossBackward0>)
            #tensor(8.4650, grad_fn=<SmoothL1LossBackward0>)
            #return lat_loss
            return mel_loss + diff_loss * lambda_diff
            #return lat_loss * lambda_encoder12 + mel_loss + diff_loss * lambda_diff

    fname_trainlog = './cache/trainlog.txt'
    file_trainlog = open(fname_trainlog,'w')
    fname_validlog = './cache/validlog.txt'
    file_validlog = open(fname_validlog,'w')

    for epoch in range(EPOCHS):
        train_loss = []
        valid_loss = []
        '''
        if epoch % EPOCH_CHECKPOINT == 0:
            str_annotation = 'epoch%d,loss=%f' % (epoch,loss.detach())
            print(str_annotation)
            torch.save(model,'./cache/%s' % str_annotation)
        '''
        MARGIN_FRAME = 40
        for batch_idx, (data, label) in enumerate(TrainLoader):
            if batch_idx % 50 == 10:
                pass
                #print('batch_idx:',batch_idx)

            data = data.to(device)
            #label = label.to(device).unsqueeze(0)#.transpose(-1,-2)#torch.Size([32, 1932]) torch.Size([32, 320])
            if use_stoi == True:
                #print('data.shape,label.shape',data.shape,label.shape)
                label = label.to(device)
                output = model(data[0],return_wf = True,return_lat12 = False).squeeze(0)#,mel_en = True
                #print(label.device,output.device)#OK
                #print(output,label)
                #print(loss_stoi(label,output),25 * loss_function(mel_spectrogram(label),mel_spectrogram(output)))
                loss = loss_stoi(label,output)+ 25 * loss_function(mel_spectrogram(label),mel_spectrogram(output))
                #print(loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss.append(loss.detach())
                #break
            else:
                #print('data.shape,label.shape',data.shape,label.shape)
                output = model(data[0],mel_en = True,return_lat12 = True)#,mel_en = True
                '''
                print('----------')
                for k in output.keys():
                    print(k,output[k].shape)
                
                for k in label.keys():
                    print(k,label[k].shape)
                #wf shape does note match 
                #'''
                loss = getloss(output,label)

                optimizer.zero_grad()
                loss.backward()
                #print(loss)
                #print(output['mel'].grad)
                #print(output['wf'].grad)
                print(output['latent_space'].grad)
                print(output.keys())
                print('---')
                print(optimizer)
                print(vars(model.LSTM.lstm_list[0]._parameters))
                print('---')
                print(model.LSTM.lstm_list[0]._parameters)
                print('---')
                output['mel']
                exit(0)                
                optimizer.step()
                train_loss.append(loss.detach())
                #break
        ret_loss = torch.mean(torch.Tensor(train_loss))
        str_log = 'epoch%d loss: %f' % (epoch, ret_loss)
        print(str_log)
        file_trainlog.write(str_log)
        '''
        if epoch == 0:
            break
        #'''

        if epoch % EPOCH_CHECKPOINT == 9:# or epoch < 10:
            #str_annotation = 'epoch%d,loss=%f' % (begin_epoch + epoch,loss.detach())
            str_annotation = 'epoch%d,loss=%f' % (begin_epoch + epoch, ret_loss)
            print(str_annotation)
            with torch.no_grad():
                for batch_idx, (data, label) in enumerate(TestLoader):
                    data = data.to(device)
                    #label = label.to(device).unsqueeze(0)#.transpose(-1,-2)
                    if use_stoi == True:
                        #print('data.shape,label.shape',data.shape,label.shape)
                        label = label.to(device)
                        output = model(data[0],return_wf = True).squeeze(0)#,mel_en = True
                        #print(output,label)
                        val_loss = loss_stoi(label,output)+0.1 * loss_function(mel_spectrogram(label),mel_spectrogram(output))
                        #print(loss)
                        #optimizer.zero_grad()
                        #loss.backward()
                        #optimizer.step()
                        valid_loss.append(val_loss.detach())
                    else:
                        output = model(data[0],mel_en = True,return_lat12 = True)#,mel_en = True
                        #output = output[:,:,MARGIN_FRAME:-MARGIN_FRAME]
                        #label = model.decoder(label)
                        val_loss = getloss(output,label)
                        #scheduler.step(val_loss)
                        valid_loss.append(loss.detach())
                str_annotation = 'epoch%d,valid_loss=%f' % (begin_epoch + epoch,torch.mean(torch.Tensor(valid_loss)))
                print(str_annotation)
                file_validlog.write(str_annotation)
                torch.save(model.LSTM,'./cache/deep_supervision/deep_loss/%s_%s' % (subject_name,str_annotation))
    str_annotation = 'finloss_epoch%d,loss=%f' % (begin_epoch + EPOCHS,loss.detach())
    print(str_annotation)
    #torch.save(model,'./cache/deep_supervision/%s' % str_annotation)

file_trainlog.close()
file_validlog.close()