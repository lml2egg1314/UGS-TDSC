import numpy as np

#index = np.arange(1,20001)
#np.random.shuffle(index)

#train_list = index[:8000]
#ival_list = index[8000:10000]
train_and_val_list = np.load('./1/train_and_val_list.npy')
test_list = np.load('./1/test_list.npy')
#retrain_train_list = test_list[:5000]
#retrain_test_list = test_list[5000:]
all_list = np.append(train_and_val_list, test_list)

#save_file_train = './3/train_list.npy'
#save_file_val = './3/val_list.npy'
#save_file_train_and_val = './3/train_and_val_list.npy'
#save_file_test = './3/test_list.npy'
#save_file_re_train = './3/retrain_train_list.npy'
#save_file_re_test = './3/retrain_test_list.npy'
save_file_all = './1/all_list.npy'

#np.save(save_file_train, train_list)
#np.save(save_file_val, val_list)
#np.save(save_file_train_and_val, train_and_val_list)
#np.save(save_file_test, test_list)
#np.save(save_file_re_train, retrain_train_list)
#np.save(save_file_re_test, retrain_test_list)
np.save(save_file_all, all_list)
