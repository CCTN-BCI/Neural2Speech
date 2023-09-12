import scipy.io as sio
from numpy import dtype
import torch
from torch.utils.data import Dataset
import scipy.io as sio
import time 
import pickle
import numpy as np
from model_lstm_and_SAE import device

#EC183
#all_electrodes = [36, 37, 38, 39, 52, 53, 54, 55, 56, 57, 69, 70, 71, 72, 73, 85, 86, 87, 88, 89, 102, 103, 104, 105, 106, 117, 118, 119, 120, 133, 134, 135, 136, 150, 151, 166, 167, 182, 196, 233, 250, 251]

#EC53
#all_electrodes = [18, 19, 20, 34, 35, 36, 37, 51, 65, 66, 68, 81, 83, 84, 85, 97, 98, 99, 100, 101, 102, 114, 115, 116, 117, 118, 130, 131, 132, 133, 134, 147, 148, 149, 150, 163, 164, 165, 166, 179, 180, 181, 195, 196, 197, 211, 212, 213, 227, 228]
#print('EC53',len(all_electrodes))

def select_electrodes_for_hg(all_hgs,all_electrodes):
    res = np.zeros((all_hgs.shape[0],len(all_electrodes)))
    #print(all_electrodes)
    for i in range(len(all_electrodes)):
        res[:,i] = all_hgs[:,all_electrodes[i]]
    return res

print('Here en is db, not en is power')

class Dataset_fin(Dataset):
    def __init__(self, fname_hg_lat, fname_wf ,fname_mels, fname_timepoint, subject_name, fname_electrodes, return_wf = False, train=True, layer=None, transform=None, target_transform=None, N=50):
        # neural transformer wf mel
        #print('begin')
        self.all_electrodes = sio.loadmat(fname_electrodes)
        self.train = train
        self.return_wf = return_wf
        self.N = N# for test use
        with open(fname_hg_lat, 'rb') as f:
            feat_mat = pickle.load(f)
        '''
        with open(fname_wf, 'rb') as f:
            self.all_wfs = pickle.load(f)['waveform']
        #'''
        with open(fname_mels, 'rb') as f:
            dic = pickle.load(f)
            #print(dic.keys())
            self.all_mels = dic['mels_en']
            self.all_wfs = dic['wfs']
        self.all_timepoints = sio.loadmat(fname_timepoint)
        #print(all_timepoints.keys())
        #dict_keys(['__header__', '__version__', '__globals__', 'times_begin', 'times_end'])
        #print('finish loading')
        self.transform = transform
        self.target_transform = target_transform
        if layer is not None:
            feat_layer = 'encoder%d' % layer
        else:
            feat_layer = 'latent_last_encoder'
        #print(feat_mat.keys())
        self.all_hgs = select_electrodes_for_hg(feat_mat['hg'],self.all_electrodes[subject_name][0])#TMD
        self.latent_space = feat_mat[feat_layer]
        #print(self.all_hgs.shape)#OK
        #print(len(self.all_wfs))#599
        #dstim = trf.get_dstim(select_electrodes_for_hg(feat_mat['hg'],all_electrodes),delays = trf.get_delays(), add_edges=True)
        #print('finish dstim')
        '''
        if self.train == True:
            self.data = torch.Tensor(dstim)[self.N:]
            self.label = torch.Tensor(feat_mat[feat_layer])[self.N:]
            self.wf = torch.Tensor(all_wf)[self.N:]
        else:
            self.data = torch.Tensor(dstim)[0:self.N]
            self.label = torch.Tensor(feat_mat[feat_layer])[0:self.N]
            self.wf = torch.Tensor(all_wf)[0:self.N]
        '''
        #self.length = self.data.shape[0]
        #print(self.label.shape,self.data.shape,self.wf.shape)
        if self.train == True:
            self.length = len(self.all_mels) - self.N
        else:
            self.length = self.N

    def __len__(self):
        return self.length

    def __getitem__(self,idx):
        if self.train == True:
            idx = idx
        else:
            idx = 0 - idx
        #idx = 0 - idx
        #print(len(self.all_timepoints['times_begin'][0]))
        idx_begin = self.all_timepoints['times_begin'][0][idx]
        idx_end = self.all_timepoints['times_end'][0][idx]
        data = torch.Tensor(self.all_hgs[idx_begin:idx_end])
        #if self.return_wf == True:
        #    return data, torch.Tensor(self.all_wfs[idx])
        #wf = self.wf[idx]
        #return data,wf
        #label = torch.Tensor(self.all_wfs[idx])
        #label = torch.Tensor(self.all_mels[idx])#OK#.transpose(-1,-2)#.unsqueeze(0)
        mel = torch.Tensor(self.all_mels[idx]).unsqueeze(0).to(device)
        latent_space = torch.Tensor(self.latent_space[idx_begin:idx_end]).to(device).transpose(-1,-2)
        #print('shapes:',mel.shape,latent_space.shape)
        #shapes: torch.Size([1, 128, 52]) torch.Size([768])
        return data, {'mel':mel,'latent_space':latent_space,'wf':torch.Tensor(self.all_wfs[idx]).unsqueeze(0)}
        #return data,label
        #print(label.shape)

class Dataset_fin_words(Dataset):
    def __init__(self, fname_hg_lat, fname_wf ,fname_mels, fname_timepoint, subject_name, fname_electrodes, fname_word_marks, return_wf = False, train=True, layer=None, transform=None, target_transform=None, N=50):
        self.all_electrodes = sio.loadmat(fname_electrodes)
        print(self.all_electrodes.keys())
        self.train = train
        self.return_wf = return_wf
        self.N = N# for test use
        with open(fname_hg_lat, 'rb') as f:
            feat_mat = pickle.load(f)
        with open(fname_mels, 'rb') as f:
            dic = pickle.load(f)
            #print(dic.keys())
            self.all_mels = dic['mels_en']
            self.all_wfs = dic['wfs']
        with open(fname_word_marks, 'rb') as f:
            self.word_marks = pickle.load(f)
        self.all_timepoints = sio.loadmat(fname_timepoint)
        self.transform = transform
        self.target_transform = target_transform
        if layer is not None:
            feat_layer = 'encoder%d' % layer
        else:
            feat_layer = 'latent_last_encoder'
        self.all_hgs = select_electrodes_for_hg(feat_mat['hg'],self.all_electrodes[subject_name][0])#TMD
        self.latent_space = feat_mat[feat_layer]
        if self.train == True:
            self.length = len(self.all_mels) - self.N
        else:
            self.length = self.N

    def __len__(self):
        return self.length
    
    def get_wf_speg_len(self, length):
        dic_speglength = {720: 2, 1040: 3, 1360: 3, 1680: 4, 2000: 4, 2320: 5, 2640: 6, 2960: 6, 
        3280: 7, 3600: 8, 3920: 8, 4240: 9, 4560: 9, 4880: 10, 5200: 11, 5520: 11, 5840: 12, 6160: 13, 
        6480: 13, 6800: 14, 7120: 14, 7440: 15, 7760: 16, 8080: 16, 8400: 17, 8720: 18, 9040: 18, 
        9360: 19, 9680: 19, 10000: 20, 10320: 21, 10640: 21, 10960: 22, 11280: 23, 11600: 23, 
        11920: 24, 12240: 24, 12560: 25, 12880: 26, 13200: 26, 13520: 27, 13840: 28, 14160: 28, 
        14480: 29, 14800: 29, 15120: 30, 15440: 31, 15760: 31, 16080: 32, 16400: 33, 16720: 33, 
        17040: 34, 17360: 34, 17680: 35, 18000: 36, 18320: 36, 18640: 37, 18960: 38, 19280: 38, 
        19600: 39, 19920: 39, 20240: 40, 20560: 41, 20880: 41, 21200: 42, 21520: 43, 21840: 43, 
        22160: 44, 22480: 44, 22800: 45, 23120: 46, 23440: 46, 23760: 47, 24080: 48, 24400: 48, 
        24720: 49, 25040: 49, 25360: 50, 25680: 51, 26000: 51, 26320: 52, 26640: 53, 26960: 53, 
        27280: 54, 27600: 54, 27920: 55, 28240: 56, 28560: 56, 28880: 57, 29200: 58, 29520: 58, 
        29840: 59, 30160: 59, 30480: 60, 30800: 61, 31120: 61, 31440: 62, 31760: 63, 32080: 63, 32400: 64}
        wf_length = (length - 80) // 320 * 320 + 80
        return dic_speglength[wf_length]

    def __getitem__(self,idx):
        sr_wf = 16000
        sr_speg = 16000 / 512
        sr_lat = 49
        if self.train == True:
            idx = idx
        else:
            idx = 0 - idx

        total_idx, wf_idx, time_begin, time_end = self.word_marks[idx]
        
        lat_length = int((time_end - time_begin) * sr_lat) + 2 # avoid spectrogram error
        wf_length = int(lat_length * 320 + 80)

        wf_beg = int(time_begin * sr_wf)


        idx_begin = self.all_timepoints['times_begin'][0][wf_idx]
        idx_end = self.all_timepoints['times_end'][0][wf_idx]
        #data = torch.Tensor(self.all_hgs[idx_begin:idx_end])
        #print(wf_length,wf_beg)
        speg_length = self.get_wf_speg_len(wf_length)

        data = torch.Tensor(self.all_hgs[idx_begin + int(sr_lat * time_begin):idx_begin + int(sr_lat * time_begin) + lat_length])

        mel = torch.Tensor(self.all_mels[wf_idx]).unsqueeze(0).to(device)
        mel = mel[:,:,int(sr_speg * time_begin):int(sr_speg * time_begin) + speg_length]

        latent_space = torch.Tensor(self.latent_space[idx_begin:idx_end]).to(device).transpose(-1,-2)
        latent_space = latent_space[:,int(sr_lat * time_begin):int(sr_lat * time_begin) + lat_length]

        wf = torch.Tensor(self.all_wfs[wf_idx][wf_beg : wf_beg + wf_length]).unsqueeze(0)

        return data, {'mel':mel,'latent_space':latent_space,'wf':wf}
        #return data,label
        #print(label.shape)