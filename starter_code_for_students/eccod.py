import numpy  as np
def encod(y):
	arr = np.array([0,0,0,0,0,0,0,0,0,0])
	arr[y] +=1
	print (arr)

encod(5)