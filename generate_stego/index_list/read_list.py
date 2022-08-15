import numpy as np

#INDEX_FILE = '/home/chenbolin/steganalysis/index_list/bossbase_and_bows_train_index.npy'
INDEX_FILE = './1/test_list.npy'
order_list = np.load(INDEX_FILE)
#print (len(order_list))
print (order_list[0])
#print order_list[0].type
#for i in range(len(order_list)):
#  print order_list[i]
#print order_list[0].type
print (order_list[9999])
#print (order_list[19999])
