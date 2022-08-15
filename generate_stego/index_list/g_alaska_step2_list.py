import numpy as np
import os 
import scipy.io as sio

# all_list = np.arange(1,80006)
    
# ln = np.arange(10000)

for i in range(1,4):
    # list_mat = sio.loadmat('{}/alaska_4w_test_list.mat'.format(i))
    step1 = np.load('{}/alaska_2w_retrain_train_list.npy'.format(i))
    step2 = np.load('{}/alaska_2w_retrain_test_list.npy'.format(i))
    # all_list = np.load('{}/alaska_4w_test_list.npy'.format(i))
    step1_train = step1[:16000]
    step1_valid = step1[16000:20000]
    step2_train = step2[:10000]
    step2_test = step2[10000:20000]
    step2_list = step2
    np.save('{}/step1_list_2w.npy'.format(i), step1)
    np.save('{}/step1_train_16k.npy'.format(i), step1_train)
    np.save('{}/step1_valid_4k.npy'.format(i), step1_valid)
    np.save('{}/step2_train_1w.npy'.format(i), step2_train)
    np.save('{}/step2_test_1w.npy'.format(i), step2_test)
    np.save('{}/step2_list_2w.npy'.format(i), step2_list)

    # step2_list = all_list[:20000]
    # train_list = all_list[:10000]
    # test_list = all_list[10000:20000]
    # np.save('{}/alaska_1w_retrain_train_list.npy'.format(i), train_list)
    # np.save('{}/alaska_1w_retrain_test_list.npy'.format(i), test_list)
    # sio.savemat('{}/alaska_2w_test_list.mat'.format(i), mdict = {'training':train_list, 'test':test_list, 'step2_list':step2_list})
    