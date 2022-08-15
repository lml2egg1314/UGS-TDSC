import os
import argparse
import time
import numpy as np
import scipy.io as sio
# import matlab.engine
import shutil
import logging
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

import train_net
# import train_net_dct


# hyperparameters
# T = 15

# IMAGE_SIZE = 256
# TRAIN_BATCH_SIZE = 16
# TEST_BATCH_SIZE = 16



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
		help='Determine the steganographic method to use',
		type=str,
		default = 'uerd'
	)


	parser.add_argument(
		'-q',
		'--quality',
		help='Determine the quality factor of jpeg images to run',
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

def setLogger(log_path, mode='a'):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  if not logger.handlers:
    # Logging to a file
    file_handler = logging.FileHandler(log_path, mode=mode)
    file_handler.setFormatter(logging.Formatter('%(asctime)s: %(message)s', '%Y-%m-%d %H:%M:%S'))
    logger.addHandler(file_handler)

    # Logging to console
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(stream_handler)




args = myParseArgs()

gpu_num = args.gpuNum
payload = float(args.payLoad)
list_num = int(args.list_num)
quality = int(args.quality)
steganography = args.steganography


# list & path
all_list = './index_list/' + str(list_num) + '/all_list.npy'
step1_list = './index_list/' + str(list_num) + '/train_and_val_list.npy'
step2_list = './index_list/' + str(list_num) + '/test_list.npy'
step1_train_list = './index_list/' + str(list_num) + '/train_list.npy'
step1_val_list = './index_list/' + str(list_num) + '/val_list.npy'
step2_train_list = './index_list/' + str(list_num) + '/retrain_train_list.npy'
step2_test_list = './index_list/' + str(list_num) + '/retrain_test_list.npy'

base_dir = '/data/lml/jpeg_test'

sp_dir = '{}_{}_{}'.format(steganography, payload, quality)

cover_dir = '/data/lml/jpeg_test/BB-cover-resample-256-jpeg-{}'.format(quality)
cover_dct_dir = '/data/lml/jpeg_test/BB-cover-resample-256-jpeg-{}-dct'.format(quality)

stego_dir = '{}/{}/stego'.format(base_dir, sp_dir)
stego_dct_dir = '{}/{}/stego-dct'.format(base_dir, sp_dir)

output_dir = './step2_{}/{}/{}'.format(quality, list_num, sp_dir)

os.makedirs(output_dir, exist_ok = True)

log_path = output_dir + '/' + 'model-log'

setLogger(log_path, mode='w')

logging.info(args)

###############################################################################################################################
###############################################1111111111111111111111111111####################################################


iteration = 0
# train net and save ckpt
logging.info("2 - Train net in step2")
train_net_time = AverageMeter()
start_train_net = time.time()
pt_name, accuracy = train_net.trainNet(output_dir, cover_dir, stego_dir, iteration, gpu_num, step1_train_list, step1_val_list, step2_list, fiter=True)
# acc_result.append(accuracy)
logging.info("\tRetrain Accuracy:", accuracy)
train_net_time.update(time.time()-start_train_net)
logging.info("2 - Train net in iteration: {:d}, \n\taccuracy: {:.3f}, \n\ttime: {:.3f}s".format(iteration, accuracy, train_net_time.val))









