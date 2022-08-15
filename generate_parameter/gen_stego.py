
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

import multiprocessing

IMAGE_SIZE = 256
GEN_NUMBER = 100  
ALPHA = 2
GPU_BATCH = 20

def AdjustingCost(grad, sort_grad, sort_rhoP1, sort_rhoM1, prerhoP1, prerhoM1, p1, p2, alpha):
    image_size = grad.size
    positive_grad = grad > 0
    negetive_grad = grad < 0
    

    th_grad = sort_grad[int(np.round(image_size * p1))]
    th_rhoP1 = sort_rhoP1[int(np.round(image_size * p2))]
    th_rhoM1 = sort_rhoM1[int(np.round(image_size * p2))]

    high_grad = abs(grad) > th_grad
    small_prerhoP1 = prerhoP1 < th_rhoP1
    small_prerhoM1 = prerhoM1 < th_rhoM1

    adjust_rhoP1 = high_grad * small_prerhoP1 * positive_grad
    adjust_rhoM1 = high_grad * small_prerhoM1 * negetive_grad

    # rhoP1 = prerhoP1 + adjust_rhoP1 * alpha
    # rhoM1 = prerhoM1 + adjust_rhoM1 * alpha
    # adjusting costs with multiplication
    prerhoP1[adjust_rhoP1] = prerhoP1[adjust_rhoP1] * alpha
    prerhoM1[adjust_rhoM1] = prerhoM1[adjust_rhoM1] * alpha

    return prerhoP1, prerhoM1


def EmbeddingSimulator(rhoP1,rhoM1,m, randChange): # m is the bits that should be embedded

    n = rhoP1.size
    Lambda = calc_lambda(rhoP1,rhoM1,m,n)

    pChangeP1 = (np.exp(-Lambda*rhoP1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))
    pChangeM1 = (np.exp(-Lambda*rhoM1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))

    # randChange = np.random.rand( rhoP1.shape[0],rhoP1.shape[1] )
    modification = np.zeros([rhoP1.shape[0], rhoP1.shape[1]])
    modification[randChange<pChangeP1]=1
    modification[randChange>=1-pChangeM1]=-1
    return modification


def calc_lambda(rhoP1,rhoM1,message_length,n):
    l3 = 1e+3
    m3 = message_length+1
    iterations = 0
    while m3 > message_length:
        l3 = l3*2
        pP1 = (np.exp(-l3*rhoP1))/(1+np.exp(-l3*rhoP1)+np.exp(-l3*rhoM1))
        pM1 = (np.exp(-l3*rhoM1))/(1+np.exp(-l3*rhoP1)+np.exp(-l3*rhoM1))
        m3 = ternary_entropyf(pP1,pM1)
        iterations = iterations+1
        if iterations>10:
            Lambda = l3
            return Lambda

    l1 = 0
    m1 = n
    Lambda = 0
    
    alpha = float(message_length)/n
    while (float(m1-m3)/n>alpha/1000.0) and (iterations<30):
        Lambda = l1+(l3-l1)/2.0
        pP1 = (np.exp(-Lambda*rhoP1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))
        pM1 = (np.exp(-Lambda*rhoM1))/(1+np.exp(-Lambda*rhoP1)+np.exp(-Lambda*rhoM1))
        m2 = ternary_entropyf(pP1,pM1)
        if m2<message_length:
            l3 = Lambda
            m3 = m2
        else:
            l1 = Lambda
            m1 = m2
        iterations = iterations+1
    return Lambda



def ternary_entropyf(pP1,pM1):
    p0=1-pP1-pM1
    p0[p0==0] = 1e-10
    pP1[pP1==0] = 1e-10
    pM1[pM1==0] = 1e-10 
    Ht = -pP1*np.log2(pP1)-pM1*np.log2(pM1)-(p0)*np.log2(p0)
    Ht = np.sum(Ht)
    return Ht


def generate_stego(grad, sort_grad, sort_rhoP1, sort_rhoM1, cover, prerhoP1, prerhoM1, payload, nzAC, p1, p2, alpha, Q, k, randChange):
    rhoP1, rhoM1 = AdjustingCost(grad, sort_grad, sort_rhoP1, sort_rhoM1, prerhoP1, prerhoM1, p1, p2, alpha)
    Modification = EmbeddingSimulator(rhoP1, rhoM1, round(nzAC * payload), randChange)
    stego = cover + Modification
    # print(k)
    # print(stego.shape)
    Q.put({k: stego})


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

def main(args):

    gpu_num = args.gpuNum
    payload = float(args.payLoad)
    list_num = int(args.list_num)
    quality = int(args.quality)
    steganography = args.steganography
    # ALPHA = int(args.alpha)

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

    sp_dir = '{}_{}_{}'.format(steganography, payload, quality)
    cover_dir = '{}/BB-cover-resample-256-jpeg-{}'.format(base_dir, quality)
    cover_dct_dir = '{}-dct'.format(cover_dir)

    cost_dir = '{}/{}/cost'.format(base_dir, sp_dir)
    stego_dir = '{}/{}/stego'.format(base_dir, sp_dir)
    stego_dct_dir = '{}/{}/stego-dct'.format(base_dir, sp_dir)

    filter_dir = '{}/filter-sets-jpeg-{}'.format(base_dir, quality)

    output_dir = '../step2_{}/{}/{}'.format(quality, list_num, sp_dir)
    
    
    params_dir = '{}/{}/{}/UGS'.format(base_dir, sp_dir, list_num)
    # params_dir = 'params_fix_messages'
    
    os.makedirs(params_dir, exist_ok = True)
    # os.makedirs(randChange_dir, exist_ok = True)
    os.makedirs(output_dir, exist_ok = True)

    pt_path = '{}/0-params.pt'.format(output_dir)

    grad_dir = '{}/{}/{}/cover_grad'.format(base_dir, sp_dir, list_num)
    # params_path = '{}/{}_{}.mat'.format(params_dir, sp_dir, list_num)


    print("\tread checkpoint path:", pt_path)
    print("\tsaved grad path:", grad_dir)

    print("\tquality:", quality)



    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_num
    device = torch.device("cuda")
    kwargs = {'num_workers': 1, 'pin_memory': True}

    index_list = np.load(step2_list)

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
    start_time = time.time()
    len = index_list.size
    params = np.zeros([len, 3])
    net_preds = np.zeros([len, GEN_NUMBER])
    residual_dis = np.zeros([len, GEN_NUMBER])
    for i in range(len):
        if (i+1)%100 == 0:
            print('{} images have done with average escaped time {}s'.format(i+1, (time.time() - start_time)/(i+1)))
            # strat_time = time.time()
        index = index_list[i]
        cover_path = '{}/{}.mat'.format(cover_dct_dir, index)
        stego_path = '{}/{}.mat'.format(stego_dct_dir, index)
        grad_path = '{}/{}.mat'.format(grad_dir, index)
        filter_path = '{}/{}.mat'.format(filter_dir, index)
        cost_path = '{}/{}.mat'.format(cost_dir, index)
        
        params_path = '{}/{}.mat'.format(params_dir, index)

        cover_mat = sio.loadmat(cover_path)
        stego_mat = sio.loadmat(stego_path)
        grad_mat = sio.loadmat(grad_path)
        filter_mat = sio.loadmat(filter_path)
        cost_mat = sio.loadmat(cost_path)


        cover_dct = cover_mat['C_COEFFS']
        cover_dct = cover_dct.astype(np.float32)
        stego_dct = stego_mat['S_COEFFS']
        stego_dct = stego_dct.astype(np.float32)
        grad = grad_mat['cover_grad']
        grad = grad.astype(np.float32)
        filters = filter_mat['filter']
        f1 = filters[0][0]
        f2 = filters[1][0]
        fs = np.stack([f1,f2])
        prerhoP1, prerhoM1 = cost_mat['rhoP1'], cost_mat['rhoM1']    

        ms = np.ones([IMAGE_SIZE,IMAGE_SIZE])
        ms[0:IMAGE_SIZE:8,0:IMAGE_SIZE:8] = 0
        ms[cover_dct==0] = 0
        nzAC = np.sum(ms)

        flat_grad = np.reshape(abs(grad), -1)
        sort_grad = np.sort(flat_grad)

        flat_rhoP1 = np.reshape(prerhoP1, -1)
        sort_rhoP1 = np.sort(flat_rhoP1)

        flat_rhoM1 = np.reshape(prerhoM1, -1)
        sort_rhoM1 = np.sort(flat_rhoM1)

        # p = np.random.rand(GEN_NUMBER, 1)
        # stegos = np.zeros([GEN_NUMBER, 1, IMAGE_SIZE, IMAGE_SIZE])

        # for j in range(GEN_NUMBER):
        #     p1 = p[j]
        #     p2 = p[j]
        #     alpha = ALPHA
        #     stegos[j] = generate_stego(grad, sort_grad, sort_rhoP1, sort_rhoM1, cover_dct, payload, nzAC, p1, p2, alpha)
        
        # start_time = time.time()
        randChange = np.random.rand(prerhoP1.shape[0], prerhoP1.shape[1])
        # sio.savemat(randChange_path, {'randChange': randChange})
        p1 = np.random.rand(GEN_NUMBER, 1)
        p2 = np.random.rand(GEN_NUMBER, 1)
        stegos = np.zeros([GEN_NUMBER, 1, IMAGE_SIZE, IMAGE_SIZE])
        spatial_stegos = np.zeros([GEN_NUMBER, 1, IMAGE_SIZE, IMAGE_SIZE])
        alpha = ALPHA
        # alpha = np.round(np.random.rand(GEN_NUMBER, 1) * 9 + 1)
        pool = multiprocessing.Pool(processes = 10)
        Q = multiprocessing.Manager().Queue()
        for k in range(GEN_NUMBER):
            pool.apply_async(generate_stego, (grad, sort_grad, sort_rhoP1, sort_rhoM1, cover_dct, \
                prerhoP1, prerhoM1, payload, nzAC, p1[k], p2[k], alpha, Q, k, randChange))
        pool.close()
        pool.join()
        # print(time.time() - start_time)
        result_dict = {}

        # start_time1 = time.time()
        while (not Q.empty()):
            result_dict.update(Q.get())
        # print(time.time() - start_time1)
        # for ii in range(GEN_NUMBER):
        #     stegos[ii] = result_dict[ii]
        for key in result_dict.keys():
            stegos[key] = result_dict[key]
        stegos[0] = stego_dct
        p1[0] = 0
        p2[0] = 0
        # alpha[0] = 0

        stegos = stegos.astype(np.float32)
        stegos = torch.from_numpy(stegos)
        spatial_stegos = spatial_stegos.astype(np.float32)
        spatial_stegos = torch.from_numpy(spatial_stegos)
        spatial_stegos = spatial_stegos.to(device)

        pred = torch.zeros(GEN_NUMBER, 1)
        for j in range(int(GEN_NUMBER/GPU_BATCH)):
            partial_stegos = stegos[j*GPU_BATCH:(j+1)*GPU_BATCH].to(device)
            spatial_stegos[j*GPU_BATCH:(j+1)*GPU_BATCH] = Bidct_layer(partial_stegos)

            partial_output = model(spatial_stegos[j*GPU_BATCH:(j+1)*GPU_BATCH])
            pred[j*GPU_BATCH:(j+1)*GPU_BATCH] = partial_output.max(1, keepdim=True)[1]

        
        redidual_dis = torch.zeros(GEN_NUMBER, 1)
        cover_dct = torch.from_numpy(cover_dct)
        cover_dct = torch.reshape(cover_dct, [1,1,IMAGE_SIZE, IMAGE_SIZE])
        cover_spatial = Bidct_layer(cover_dct.to(device))

        fs = np.expand_dims(fs, 1)
        fs = fs.astype(np.float32)
        fs = torch.from_numpy(fs)
        fs = fs.to(device)
        cover_residual = torch.conv2d(cover_spatial, fs)
        stego_residual = torch.conv2d(spatial_stegos, fs)

        distance = torch.sum(torch.abs(stego_residual - cover_residual), [1,2,3])

        # select_index = pred < 1
        # select_dis = distance[select_index]
        # min_index = torch.argmin(select_dis)
        sio.savemat(params_path, {'randChange':randChange, 'p1':p1, 'p2':p2, 'alpha':alpha, \
            'pred':pred.cpu().numpy(), 'residual_dis': distance.cpu().numpy()})   
        
        
        
        
        # select_index = pred < 1

        # select_p = p[select_index]
        # select_stego = spatial_stegos[select_index]
        # # print(select_stego.shape)
        # if select_stego.shape[0] == 0:
        #     continue
        # select_stego = torch.reshape(select_stego.contiguous(), [select_stego.shape[0], 1, select_stego.shape[1],select_stego.shape[2]])

        # cover_dct = torch.from_numpy(cover_dct)
        # cover_dct = torch.reshape(cover_dct, [1,1,IMAGE_SIZE, IMAGE_SIZE])
        # cover_spatial = Bidct_layer(cover_dct.to(device))

        # fs = np.expand_dims(fs, 1)
        # fs = fs.astype(np.float32)
        # fs = torch.from_numpy(fs)
        # fs = fs.to(device)
        # cover_residual = torch.conv2d(cover_spatial, fs)
        # stego_residual = torch.conv2d(select_stego, fs)

        # distance = torch.sum(torch.abs(stego_residual - cover_residual), [1,2,3])

        # min_index = torch.argmin(distance)

        # final_p = select_p[min_index]

        
        # params[i] = final_p
        
        
    
    # sio.savemat(params_path, {'randChange':randChange, 'pred':pred, 'residual_dis': residual_dis})   




def myParseArgs():
	parser = argparse.ArgumentParser()

	parser.add_argument(
		'-g',
		'--gpuNum',
		help='Determine which gpu to use',
		type=str,
		choices=['0', '1', '2', '3'],
		default= '0'
		# required=True
	)

	parser.add_argument(
		'-p',
		'--payLoad',
		help='Determine the payload to embed',
		type=str,
		default= '0.4'
		# required=True
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
		default= '1'
		# required=True
	)
	# parser.add_argument(
	# 	'-a',
	# 	'--alpha',
	# 	help='Determine the list num',
	# 	type=str,
	# 	default= '2'
	# 	# required=True
	# )

	args = parser.parse_args()
	
	return args

if __name__ == '__main__':
    args = myParseArgs()
    main(args)

