import os
import argparse
import numpy as np
import cv2
from pathlib import Path
import copy
import logging
import random
import scipy.io as sio
import time
# import matlab.engine

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
from torch.autograd import Variable

from srm_filter_kernel import all_normalized_hpf_list
from MPNCOV.python import MPNCOV



BB_COVER_DIR = '/data/lml/jpeg_test/BB-cover-resample-256-jpeg-75'
# BOSSBASE_COVER_DIR = '/data/lml/jpeg_test/BossBase-1.01-cover-resample-256-jpeg-95'
# BOWS_COVER_DIR = '/data/lml/jpeg_test/BOWS2-cover-resample-256-jpeg-95'

IMAGE_SIZE = 256
BATCH_SIZE = 32 // 2
EPOCHS = 200
#EPOCHS = 2
LR = 0.01
WEIGHT_DECAY = 5e-4
EMBEDDING_RATE = 0.4

TRAIN_FILE_COUNT = 8000
TRAIN_PRINT_FREQUENCY = 100
EVAL_PRINT_FREQUENCY = 1
DECAY_EPOCH = [80, 140, 180]

#FINETUNE_EPOCHS = 1
FINETUNE_EPOCHS = 100


class TLU(nn.Module):
	def __init__(self, threshold):
		super(TLU, self).__init__()

		self.threshold = threshold

	def forward(self, input):
		output = torch.clamp(input, min=-self.threshold, max=self.threshold)

		return output

class HPF(nn.Module):
	def __init__(self):
		super(HPF, self).__init__()

		#Load 30 SRM Filters
		all_hpf_list_7x7 = []

		for hpf_item in all_normalized_hpf_list:
			if hpf_item.shape[0] == 3:
				hpf_item = np.pad(hpf_item, pad_width=((2, 2), (2, 2)), mode='constant')
			elif hpf_item.shape[0] == 5:
				hpf_item = np.pad(hpf_item, pad_width=((1, 1), (1, 1)), mode='constant')
			all_hpf_list_7x7.append(hpf_item)

		hpf_weight = nn.Parameter(torch.Tensor(all_hpf_list_7x7).view(42, 1, 7, 7), requires_grad=False)


		self.hpf = nn.Conv2d(1, 42, kernel_size=7, padding=3, bias=False)
		self.hpf.weight = hpf_weight

		#Truncation, threshold = 5
		self.tlu = TLU(5.0)


	def forward(self, input):

		output = self.hpf(input)
		output = self.tlu(output)


		return output



class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.group1 = HPF()

		self.group2 = nn.Sequential(
			nn.Conv2d(42, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
			# nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
		)

		self.group3 = nn.Sequential(
			nn.Conv2d(32, 32, kernel_size=3, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(),

			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
			# nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
		)

		self.group4 = nn.Sequential(
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),

			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.AvgPool2d(kernel_size=3, padding=1, stride=2)
			# nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
		)

		self.group5 = nn.Sequential(
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),

			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),

			#nn.AvgPool2d(kernel_size=32, stride=1)
		)

		self.fc1 = nn.Linear(int(256 * (256 + 1) / 2), 2)
		#self.fc1 = nn.Linear(1 * 1 * 256, 2)

	def forward(self, input):
		output = input

		output = self.group1(output)
		output = self.group2(output)
		output = self.group3(output)
		output = self.group4(output)
		output = self.group5(output)

		output = MPNCOV.CovpoolLayer(output)
		output = MPNCOV.SqrtmLayer(output, 5)
		output = MPNCOV.TriuvecLayer(output)

		output = output.view(output.size(0), -1)
		output = self.fc1(output)

		return output


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


def train(model, device, train_loader, optimizer, epoch):
	batch_time = AverageMeter() #ONE EPOCH TRAIN TIME
	data_time = AverageMeter()
	losses = AverageMeter()

	model.train()

	end = time.time()

	for i, sample in enumerate(train_loader):

		data_time.update(time.time() - end) 

		data, label = sample['data'], sample['label']

		shape = list(data.size())
		data = data.reshape(shape[0] * shape[1], *shape[2:])
		label = label.reshape(-1)


		# fig = plt.figure(figsize=(10, 10))

		# rows = 2
		# coloums = 6

		# for i in range(1, coloums + 1):
		#	 fig.add_subplot(rows, coloums, i)

		#	 plt.imshow(np.squeeze(data[i + 10]))

		# for i in range(1, coloums + 1):
		#	 fig.add_subplot(rows, coloums, coloums + i)

		#	 plt.imshow(np.squeeze(sc[i + 10]))

		# plt.show()

		# return

		data, label = data.to(device), label.to(device)

		optimizer.zero_grad()

		end = time.time()

		output = model(data) #FP


		criterion = nn.CrossEntropyLoss()
		loss = criterion(output, label)

		losses.update(loss.item(), data.size(0))

		loss.backward()			#BP
		optimizer.step()

		batch_time.update(time.time() - end) #BATCH TIME = BATCH BP+FP
		end = time.time()

		if i % TRAIN_PRINT_FREQUENCY == 0:
			# logging.info('Epoch: [{}][{}/{}] \t Loss {:.6f}'.format(epoch, i, len(train_loader), loss.item()))

			logging.info('Epoch: [{0}][{1}/{2}]\t'
									'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
									'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
									'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
									 epoch, i, len(train_loader), batch_time=batch_time,
									 data_time=data_time, loss=losses))

def adjust_bn_stats(model, device, train_loader):
	model.train()

	with torch.no_grad():
		for sample in train_loader:
			data, label = sample['data'], sample['label']

			shape = list(data.size())
			data = data.reshape(shape[0] * shape[1], *shape[2:])
			label = label.reshape(-1)

			data, label = data.to(device), label.to(device)

			output = model(data)


def evaluate(model, device, eval_loader,best_acc, epoch, optimizer, pt_path):

	model.eval()

	test_loss = 0
	correct = 0

	with torch.no_grad():
		for sample in eval_loader:
			data, label = sample['data'], sample['label']

			shape = list(data.size())
			data = data.reshape(shape[0] * shape[1], *shape[2:])
			label = label.reshape(-1)
			data, label = data.to(device), label.to(device)
			
			output = model(data)
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(label.view_as(pred)).sum().item()

	accuracy = correct / (len(eval_loader.dataset) * 2)

	if accuracy > best_acc and epoch > 180:
		best_acc = accuracy
		all_state = {
		'original_state': model.state_dict(),
		'optimizer_state': optimizer.state_dict(),
		'epoch': epoch
		}
		torch.save(all_state, pt_path)

	logging.info('-' * 8)
	logging.info('Eval accuracy: {:.4f}'.format(accuracy))
	logging.info('Best accuracy: {:.4f}'.format(best_acc))


	logging.info('-' * 8)

	return accuracy

def test(model, device, eval_loader, optimizer):

	model.eval()

	test_loss = 0
	correct = 0

	with torch.no_grad():
		for sample in eval_loader:
			data, label = sample['data'], sample['label']

			shape = list(data.size())
			data = data.reshape(shape[0] * shape[1], *shape[2:])
			label = label.reshape(-1)
			data, label = data.to(device), label.to(device)
			
			output = model(data)
			pred = output.max(1, keepdim=True)[1]
			correct += pred.eq(label.view_as(pred)).sum().item()

	accuracy = correct / (len(eval_loader.dataset) * 2)

	logging.info('-' * 8)
	logging.info('Eval accuracy: {:.4f}'.format(accuracy))


	logging.info('-' * 8)

	return accuracy


def initWeights(module):
	if type(module) == nn.Conv2d:
		if module.weight.requires_grad:
			nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

			# nn.init.xavier_uniform_(module.weight.data)
			# nn.init.constant_(module.bias.data, val=0.2)
		# else:
		#	 module.weight.requires_grad = True

	if type(module) == nn.Linear:
		nn.init.normal_(module.weight.data, mean=0, std=0.01)
		nn.init.constant_(module.bias.data, val=0)


class AugData():
	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		rot = random.randint(0,3)
		data = np.rot90(data, rot, axes=[1, 2]).copy()


		if random.random() < 0.5:
			data = np.flip(data, axis=2).copy()

		new_sample = {'data': data, 'label': label}

		return new_sample


class ToTensor():
	def __call__(self, sample):
		data, label = sample['data'], sample['label']

		data = np.expand_dims(data, axis=1)
		data = data.astype(np.float32)
		# data = data / 255.0

		new_sample = {
			'data': torch.from_numpy(data),
			'label': torch.from_numpy(label).long(),
		}

		return new_sample


class MyDataset(Dataset):

	def __init__(self, cover_dir, stego_dir, index_path, transform=None):
		self.index_list = np.load(index_path)
		self.transform = transform

		self.cover_path = cover_dir + '/{}.jpg'
		# self.bossbase_cover_path = BOSSBASE_COVER_DIR + '/{}.jpg'
		# self.bows_cover_path = BOWS_COVER_DIR + '/{}.jpg'
		self.all_stego_path = stego_dir + '/{}.jpg'

	def __len__(self):
		return self.index_list.shape[0]

	def __getitem__(self, idx):
		file_index = self.index_list[idx]

		cover_path = self.cover_path.format(file_index)	 
		# if file_index <= 10000:
		# 		cover_path = self.bossbase_cover_path.format(file_index)
		# else:
		# 	cover_path = self.bows_cover_path.format(file_index - 10000)
		stego_path = self.all_stego_path.format(file_index)
		#print(stego_path)

		cover_data = cv2.imread(cover_path, -1)		 # spacial jpeg image
		stego_data = cv2.imread(stego_path, -1)

		data = np.stack([cover_data, stego_data])
		label = np.array([0, 1], dtype='int32')

		sample = {'data': data, 'label': label}

		if self.transform:
			sample = self.transform(sample)

		return sample


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





def testNet(pt_path, cover_dir, stego_dir, test_stego_dir, gpu_num, train_index_path, val_index_path, test_index_path):

	log_path = 'test-model-log'
	# pt_path = output_dir + '/' + str(iteration) + '-' + 'params.pt'
	print("\tsaved log path:", log_path)
	print("\tsaved checkpoint path:", pt_path)

	setLogger(log_path, mode='w')

	os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
	device = torch.device("cuda")
	kwargs = {'num_workers': 3, 'pin_memory': True}


	# load data
	train_transform = transforms.Compose([
		AugData(),
		ToTensor()
	])

	eval_transform = transforms.Compose([
		ToTensor()
	])

	train_dataset = MyDataset(cover_dir, stego_dir, index_path=train_index_path, transform=train_transform)
	valid_dataset = MyDataset(cover_dir, stego_dir, index_path=val_index_path, transform=eval_transform)
	test_dataset = MyDataset(cover_dir, test_stego_dir, index_path=test_index_path, transform=eval_transform)

	train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, **kwargs)
	valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)
	test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, **kwargs)


	# load model and params
	model = Net().to(device)
	params = model.parameters()

	params_wd, params_rest = [], []
	for param_item in params:
		if param_item.requires_grad:
			(params_wd if param_item.dim() != 1 else params_rest).append(param_item)

	param_groups = [{'params': params_wd, 'weight_decay': WEIGHT_DECAY},
					{'params': params_rest}]

	optimizer = optim.SGD(param_groups, lr=LR, momentum=0.9)
	logging.info('\nTest set accuracy: \n')
	all_state = torch.load(pt_path)
	original_state = all_state['original_state']
	optimizer_state = all_state['optimizer_state']
	
	model.load_state_dict(original_state)
	optimizer.load_state_dict(optimizer_state)
	

	logging.info('\nTest set accuracy: \n')

	adjust_bn_stats(model, device, train_loader)
	acc = test(model, device, test_loader, optimizer)


	return acc

