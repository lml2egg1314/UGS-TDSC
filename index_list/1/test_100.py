import numpy as np

#INDEX_FILE = '/home/chenbolin/steganalysis/index_list/bossbase_and_bows_train_index.npy'
INDEX_FILE = './all_list.npy'
order_list = np.load(INDEX_FILE)

list_100 = order_list[:100]

save_list_100 = './first_100.npy'

np.save(save_list_100, list_100)
