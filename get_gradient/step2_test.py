import os
import argparse
import time
import numpy as np
import scipy.io as sio

import shutil

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable


import train_net_test
import test



# hyperparameters
T = 15

IMAGE_SIZE = 256
TRAIN_BATCH_SIZE = 16
TEST_BATCH_SIZE = 16








def myParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-g',
		'--gpuNum',
		help='Determine which gpu to use',
		type=str,
		choices=['0', '1', '2', '3'],
		required=True
	)

	parser.add_argument(
		'-p',
		'--payLoad',
		help='Determine the payload to embed',
		type=str,
		required=True
	)

	parser.add_argument(
		'-s',
		'--steganography',
		help='Determine the alpha to update cost',
		type=str,
		default = 'uerd'
	)


	parser.add_argument(
		'-q',
		'--quality',
		help='Determine the quality to run jpeg',
		type=str,
		default = '75'
	)

	parser.add_argument(
		'-ln',
		'--list_num',
		help='Determine the list num',
		type=str,
		required=True
	)

	args = parser.parse_args()
	
	return args





class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count





args = myParseArgs()

gpu_num = args.gpuNum
payload = float(args.payLoad)
list_num = int(args.list_num)
quality = int(args.quality)
steganography = args.steganography

print(quality)
# list & path
all_list = './index_list/' + str(list_num) + '/all_list.npy'
step1_list = './index_list/' + str(list_num) + '/train_and_val_list.npy'
step2_list = './index_list/' + str(list_num) + '/test_list.npy'
step1_train_list = './index_list/' + str(list_num) + '/train_list.npy'
step1_val_list = './index_list/' + str(list_num) + '/val_list.npy'
step2_train_list = './index_list/' + str(list_num) + '/retrain_train_list.npy'
step2_test_list = './index_list/' + str(list_num) + '/retrain_test_list.npy'

base_dir = '/data/lml/jpeg_test'

cover_dir = '/data/lml/jpeg_test/BB-cover-resample-256-jpeg-{}'.format(quality)
cover_dct_dir = '/data/lml/jpeg_test/BB-cover-resample-256-jpeg-{}-dct'.format(quality)

sp_dir = '{}_{}_{}'.format(steganography, payload, quality)

stego_dir = '{}/{}/stego'.format(base_dir, sp_dir)
# stego_dir = '{}/{}/stego'.format(base_dir, sp_dir)
stego_dct_dir = '{}/{}/stego-dct'.format(base_dir, sp_dir)

test_stego_dir = '{}/{}/{}/output_UGS2/stego'.format(base_dir, sp_dir, list_num)

output_dir = '../step2_{}/{}/{}'.format(quality, list_num, sp_dir)

os.makedirs(output_dir, exist_ok = True)

pt_name = '{}/0-params.pt'.format(output_dir)

grad_dir = '{}/{}/{}/cover_grad'.format(base_dir, sp_dir, list_num)


print("3 - Calculate and save gradient")
gen_grad_time = AverageMeter()
start_gen_grad = time.time()
acc = train_net_test.testNet(pt_name, cover_dir, stego_dir, test_stego_dir, gpu_num, step1_train_list, step1_val_list, step2_test_list)


with open('noretrain_UGS2.txt', 'a+') as f:
	f.write('{}:{} \n'.format(sp_dir, acc))
gen_grad_time.update(time.time()-start_gen_grad)
print("3 - Calculate and save gradient: {:.3f}s".format(gen_grad_time.val))








