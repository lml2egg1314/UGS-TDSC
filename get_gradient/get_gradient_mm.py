#!/usr/bin/env python3

import os
import argparse
import numpy as np
import cv2
import logging
import random
import shutil
import time
import math
from PIL import *
import PIL.Image
import scipy.io as sio
from scipy.misc import derivative

from pathlib import Path
import scipy.fftpack as fftpack

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable
# import torch_dct as dct

import train_net
#import jpeg_xu_net



# BB_COVER_DIR = '/data/lml/jpeg_test/BB-cover-resample-256-jpeg-75-dct'
# BOWS_COVER_DIR = '/data/lml/jpeg_test/BOWS2-cover-resample-256-jpeg-75-dct'

BATCH_SIZE = 1



# quality = 75
quant_table_75 = np.array([
    [8, 6, 5, 8, 12, 20, 26, 31],
    [6, 6, 7, 10, 13, 29, 30, 28],
    [7, 7, 8, 12, 20, 29, 35, 28],
    [7, 9, 11, 15, 26, 44, 40, 31],
    [9, 11, 19, 28, 34, 55, 52, 39],
    [12, 18, 28, 32, 41, 52, 57, 46],
    [25, 32, 39, 44, 52, 61, 60, 51],
    [36, 46, 48, 49, 56, 50, 52, 50]
  ], dtype=np.float32)



# quanlity = 95
quant_table_95 = np.array([
    [2, 1, 1, 2, 2, 4, 5, 6],
    [1, 1, 1, 2, 3, 6, 6, 6],
    [1, 1, 2, 2, 4, 6, 7, 6],
    [1, 2, 2, 3, 5, 9, 8, 6],
    [2, 2, 4, 6, 7, 11, 10, 8],
    [2, 4, 6, 6, 8, 10, 11, 9],
    [5, 6, 8, 9, 10, 12, 12, 10],
    [7, 9, 10, 10, 11, 10, 10, 10]
  ], dtype=np.float32)



class ToTensor():
	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		# data = np.expand_dims(data, axis=0)
		data = np.expand_dims(data, axis=1)
		data = data.astype(np.float32)

		new_sample = {
			'data': torch.from_numpy(data),
			'label': torch.from_numpy(label).long(),
		}

		return new_sample



class MyDataset(Dataset):
	def __init__(self, index_path, cover_dir, stego_dir, transform=None):
		self.index_list = np.load(index_path)
		self.transform = transform

		self.bb_cover_path = cover_dir + '/{}.mat'
		# self.bows_cover_path = BOWS_COVER_DIR + '/{}.mat'
		self.stego_path = stego_dir + '/{}.mat'

	def __len__(self):
		return self.index_list.shape[0]

	def __getitem__(self, idx):
		file_index = self.index_list[idx]

		
		cover_path = self.bb_cover_path.format(file_index)
		stego_path = self.stego_path.format(file_index)
		

		cover_data = sio.loadmat(cover_path)['C_COEFFS']
		stego_data = sio.loadmat(stego_path)['S_COEFFS']
		data = np.stack([cover_data, stego_data])
		label = np.array([0, 1], dtype='int32')

		# data = cover_data
		# label = np.array([0], dtype='int32')

		sample = {'data': data, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample


def get_dctmtx(n):
    A = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if(i == 0):
                x = np.sqrt(1/n)
            else:
                x = np.sqrt(2/n)

            A[i][j] = x*np.cos(np.pi*(j+0.5)*i/n)

    return A

# a = get_dctmtx(8)
# print(a)

def get_qtable(QF):
    # standard quantization matrix
    D = np.array([[16,11,10,16,24,40,51,61],
        [12,12,14,19,26,58,60,55],
        [14,13,16,24,40,57,69,56],
        [14,17,22,29,51,87,80,62],
        [18,22,37,56,68,109,103,77],
        [24,35,55,64,81,104,113,92],
        [49,64,78,87,103,121,120,101],
        [72,92,95,98,112,100,103,99]], dtype=np.int64)

    if QF >= 50:
        s = 200 - 2*QF
    else:
        s = 5000/QF

    qtable = np.floor((50 + s*D)/100)

    return qtable


class Bidct(nn.Module):
    def __init__(self, qtable, mtx):
        super(Bidct, self).__init__()

        self.qtable = qtable
        self.mtx = mtx

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], int(x_shape[1]*x_shape[2]/8), 8, x_shape[3])
        x = x.permute(0,1,3,2)
        x = x.reshape(x_shape[0], int(x_shape[1]*x_shape[2]*x_shape[3]/(8*8)), 8, 8)
        x = x.permute(0,1,3,2)
        x = x * self.qtable
        x = torch.matmul(torch.t(self.mtx), x)
        x = torch.matmul(x, self.mtx)
        x = x.permute(0,1,3,2)
        x = x.reshape(x_shape[0], int(x_shape[1]*x_shape[2]/8), x_shape[2], 8)
        x = x.permute(0,1,3,2)
        x = x.reshape(x_shape) + 128

        return x

class Bdct(nn.Module):
    def __init__(self, qtable, mtx):
        super(Bdct, self).__init__()

        self.qtable = qtable
        self.mtx = mtx

    def forward(self, x):
        x_shape = x.shape
        x = x.view(x_shape[0], int(x_shape[1] * x_shape[2] / 8), 8, x_shape[3])
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x_shape[0], int(x_shape[1] * x_shape[2] * x_shape[3] / (8 * 8)), 8, 8)
        x = x.permute(0, 1, 3, 2)
        x = (x - 128) / self.qtable
        x = torch.matmul(self.mtx, x)
        x = torch.matmul(x, torch.t(self.mtx))
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x_shape[0], int(x_shape[1] * x_shape[2] / 8), x_shape[2], 8)
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(x_shape)

        return x




def calSignGrad(pt_path, indexPath, cover_dir, stego_dir, grad_dir, gpu_num, quality):

	print("\tread checkpoint path:", pt_path)
	print("\tsaved grad path:", grad_dir)

	print("\tquality:", quality)


	Path(grad_dir).mkdir(parents=True, exist_ok=True)


	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
	device = torch.device("cuda")
	kwargs = {'num_workers': 1, 'pin_memory': True}


	# PROB_DIR = '/data/lml/jpeg_test/juni_0.4_{}/prob2'.format(quality)
	# STEGO_DIR = '/data/lml/jpeg_test/juni_0.4_{}/stego-dct'.format(quality)

	data_transform = transforms.Compose([
		ToTensor()
	])
	data_dataset = MyDataset(indexPath, cover_dir, stego_dir, transform=data_transform)
	data_loader = DataLoader(data_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	index_list = np.load(indexPath)

	qtable = get_qtable(quality)
	qtable = torch.from_numpy(qtable).float().to(device)

	mtx = get_dctmtx(8)
	mtx = torch.from_numpy(mtx).float().to(device)

	Bdct_layer = Bdct(qtable, mtx)
	Bidct_layer = Bidct(qtable, mtx)
	model = train_net.Net().to(device)
	all_state = torch.load(pt_path)
	model.load_state_dict(all_state['original_state'])
	model.eval()

	#torch.set_printoptions(edgeitems=5)



	for i, sample in enumerate(data_loader):

		file_index = index_list[i]
		#print(str(i+1), "-", file_index, "---------------------")

		data, label = sample['data'], sample['label']
		# print(data.shape)
		shape = list(data.size())
		data = data.reshape(shape[0] * shape[1], *shape[2:])
		label = label.reshape(-1)
		data, label = data.to(device), label.to(device)
		data.requires_grad = True

		spatial_data = Bidct_layer(data)

		output = model(spatial_data)
		pred = output.max(1, keepdim=True)[1]
		criterion = nn.CrossEntropyLoss()
		loss = criterion(output, label)

		#if (loss == 0):
		#	print("index", file_index)
		#	exit(0)

		model.zero_grad()
		loss.backward()

		# grad of dct coefficients
		grad = data.grad.data
		
		cover_grad = grad[0].cpu().numpy().squeeze()
		pred =pred.cpu().numpy()
		
		

		sio.savemat('{}/{}.mat'.format(grad_dir, str(file_index)), mdict={'cover_grad':cover_grad, 'pred':pred})
		


	
