import numpy as np
import os
from scipy import io


'''
index = np.arange(1,20001)
np.random.shuffle(index)

train_list = index[:8000]
val_list = index[8000:10000]
train_and_val_list = index[:10000]
test_list = index[10000:20001]
retrain_train_list = test_list[:5000]
retrain_test_list = test_list[5000:]
all_list = np.append(train_and_val_list, test_list)

save_file_train = './3/train_list.npy'
save_file_val = './3/val_list.npy'
save_file_train_and_val = './3/train_and_val_list.npy'
save_file_test = './3/test_list.npy'
save_file_re_train = './3/retrain_train_list.npy'
save_file_re_test = './3/retrain_test_list.npy'
save_file_all = './3/all_list.npy'

np.save(save_file_train, train_list)
np.save(save_file_val, val_list)
np.save(save_file_train_and_val, train_and_val_list)
np.save(save_file_test, test_list)
np.save(save_file_re_train, retrain_train_list)
np.save(save_file_re_test, retrain_test_list)
np.save(save_file_all, all_list)
'''



filePath = '/data/stt/adv_spa_bl_p0.4/params/'
namelist = os.listdir(filePath)

mylist = []
for i in namelist:
	order = int(i[:-4])
	mylist.append(order)
print(len(mylist))

np.save('./test/test_image.npy', mylist)

'''



mat = np.load('./3/test_list.npy')
 
io.savemat('./3/test_list.mat', {'index': mat})

'''


