import numpy as np

filter_class_1 = [
  np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 1],
    [0, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [1, -1, 0],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 1],
    [0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 0],
    [1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 0],
    [0, 1, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
  ], dtype=np.float32)
]


filter_class_2 = [
  np.array([
    [1, 0, 0],
    [0, -2, 0],
    [0, 0, 1]
  ], dtype=np.float32),
  np.array([
    [0, 1, 0],
    [0, -2, 0],
    [0, 1, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 1],
    [0, -2, 0],
    [1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0],
    [1, -2, 1],
    [0, 0, 0]
  ], dtype=np.float32),
]


filter_class_3 = [
  np.array([
    [-1, 0, 0, 0, 0],
    [0, 3, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, -1, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, -1],
    [0, 0, 0, 3, 0],
    [0, 0, -3, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, -3, 3, -1],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 0, 3, 0],
    [0, 0, 0, 0, -1]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, -3, 0, 0],
    [0, 0, 3, 0, 0],
    [0, 0, -1, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, -3, 0, 0],
    [0, 3, 0, 0, 0],
    [-1, 0, 0, 0, 0]
  ], dtype=np.float32),
  np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [-1, 3, -3, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
  ], dtype=np.float32)
]
filter_class_4 = [
	np.array([
	  [0,0,0,0,0],
	  [0,0,0,0,0],
	  [1,-4,6,-4,1],
	  [0,0,0,0,0],
	  [0,0,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [1,0,0,0,0],
	  [0,-4,0,0,0],
	  [0,0,6,0,0],
	  [0,0,0,-4,0],
	  [0,0,0,0,1]
  ], dtype=np.float32),
	np.array([
	  [0,0,1,0,0],
	  [0,0,-4,0,0],
	  [0,0,6,0,0],
	  [0,0,-4,0,0],
	  [0,0,1,0,0]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,0,1],
	  [0,0,0,-4,0],
	  [0,0,6,0,0],
	  [0,-4,0,0,0],
	  [1,0,0,0,0]
  ], dtype=np.float32)
]
filter_class_5 = [
	np.array([
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [-1,5,-10,10,-5,1,0],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [-1,0,0,0,0,0,0],
	  [0,5,0,0,0,0,0],
	  [0,0,-10,0,0,0,0],
	  [0,0,0,10,0,0,0],
	  [0,0,0,0,-5,0,0],
	  [0,0,0,0,0,1,0],
	  [0,0,0,0,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,-1,0,0,0],
	  [0,0,0,5,0,0,0],
	  [0,0,0,-10,0,0,0],
	  [0,0,0,10,0,0,0],
	  [0,0,0,-5,0,0,0],
	  [0,0,0,1,0,0,0],
	  [0,0,0,0,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,0,0,0,-1],
	  [0,0,0,0,0,5,0],
	  [0,0,0,0,-10,0,0],
	  [0,0,0,10,0,0,0],
	  [0,0,-5,0,0,0,0],
	  [0,1,0,0,0,0,0],
	  [0,0,0,0,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [0,1,-5,10,-10,5,-1],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,0,0,0,0],
	  [0,1,0,0,0,0,0],
	  [0,0,-5,0,0,0,0],
	  [0,0,0,10,0,0,0],
	  [0,0,0,0,-10,0,0],
	  [0,0,0,0,0,5,0],
	  [0,0,0,0,0,0,-1]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,0,0,0,0],
	  [0,0,0,1,0,0,0],
	  [0,0,0,-5,0,0,0],
	  [0,0,0,10,0,0,0],
	  [0,0,0,-10,0,0,0],
	  [0,0,0,5,0,0,0],
	  [0,0,0,-1,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,1,0],
	  [0,0,0,0,-5,0,0],
	  [0,0,0,10,0,0,0],
	  [0,0,-10,0,0,0,0],
	  [0,5,0,0,0,0,0],
	  [-1,0,0,0,0,0,0]
  ], dtype=np.float32)
]
filter_class_6 = [
	np.array([
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [1,-6,15,-20,15,-6,1],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0],
	  [0,0,0,0,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [1,0,0,0,0,0,0],
	  [0,-6,0,0,0,0,0],
	  [0,0,15,0,0,0,0],
	  [0,0,0,-20,0,0,0],
	  [0,0,0,0,15,0,0],
	  [0,0,0,0,0,-6,0],
	  [0,0,0,0,0,0,1]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,1,0,0,0],
	  [0,0,0,-6,0,0,0],
	  [0,0,0,15,0,0,0],
	  [0,0,0,-20,0,0,0],
	  [0,0,0,15,0,0,0],
	  [0,0,0,-6,0,0,0],
	  [0,0,0,1,0,0,0]
  ], dtype=np.float32),
	np.array([
	  [0,0,0,0,0,0,1],
	  [0,0,0,0,0,-6,0],
	  [0,0,0,0,15,0,0],
	  [0,0,0,-20,0,0,0],
	  [0,0,15,0,0,0,0],
	  [0,-6,0,0,0,0,0],
	  [1,0,0,0,0,0,0]
  ], dtype=np.float32)
]
square_2x2 = np.array([
  [1,-1,0],
  [-1,1,0],
  [0,0,0],
], dtype=np.float32)
square_3x3 = np.array([
  [1, -2, 1],
  [-2, 4, -2],
  [1, -2, 1]
], dtype=np.float32)
square_4x4 = np.array([
  [1,-3,3,-1,0],
  [-3,9,-9,3,0],
  [3,-9,9,-3,0],
  [-1,3,-3,1,0],
  [0,0,0,0,0]
], dtype=np.float32)
square_5x5 = np.array([
  [1,-4,6,-4,1],
  [-4,16,-24,16,-4],
  [6,-24,36,-24,6],
  [-4,16,-24,16,-4],
  [1,-4,6,-4,1],
], dtype=np.float32)
square_6x6 = np.array([
  [1,-5,10,-10,5,-1,0],
  [-5,25,-50,50,-25,5,0],
  [10,-50,100,-100,50,-10,0],
  [-10,50,-100,100,-50,10,0],
  [5,-25,50,-50,25,-5,0],
  [-1,5,-10,10,-5,1,0],
  [0,0,0,0,0,0,0]
], dtype=np.float32)
square_7x7 = np.array([
  [1,-6,15,-20,15,-6,1],
  [-6,36,-90,120,-90,36,-6],
  [15,-90,225,-300,255,-90,15],
  [-20,120,-300,400,-300,120,-20],
  [15,-90,225,-300,255,-90,15],
  [-6,36,-90,120,-90,36,-6],
  [1,-6,15,-20,15,-6,1],
], dtype=np.float32)
all_hpf_list = filter_class_1 + filter_class_2 + filter_class_3 + filter_class_4 + filter_class_5 + filter_class_6 + [square_2x2, square_3x3, square_4x4, square_5x5, square_6x6, square_7x7]

normalized_filter_class_2 = [hpf / 2 for hpf in filter_class_2]
normalized_filter_class_3 = [hpf / 3 for hpf in filter_class_3]
normalized_filter_class_4 = [hpf / 6 for hpf in filter_class_4]
normalized_filter_class_5 = [hpf / 10 for hpf in filter_class_5]
normalized_filter_class_6 = [hpf / 20 for hpf in filter_class_6]

normalized_square_3x3 = square_3x3 / 4
normalized_square_4x4 = square_4x4 / 9
normalized_square_5x5 = square_3x3 / 36
normalized_square_6x6 = square_5x5 / 100
normalized_square_7x7 = square_3x3 / 400

normalized_square_list = [square_2x2] + [normalized_square_3x3] + [normalized_square_4x4] + [normalized_square_5x5] + [normalized_square_6x6] + [normalized_square_7x7]

all_normalized_hpf_list = filter_class_1 + normalized_filter_class_2 + normalized_filter_class_3 + normalized_filter_class_4 + normalized_filter_class_5 + normalized_filter_class_6 +  normalized_square_list