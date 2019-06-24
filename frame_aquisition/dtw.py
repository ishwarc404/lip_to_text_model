m = 1
MASTER_COORDS = []
while(m<6):

	fname = "v"+str(m)+".txt"
	with open(fname) as f:
	    content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [x.strip() for x in content] 

	x_coords  = []
	y_coords = []
	for i in content:
		x_coords.append(int(i[0:3]))
		y_coords.append(int(i[4:7]))

	#combining x and y
	coords  = []
	length = len(x_coords)

	for i in range(length):
		coords.append([x_coords[i],y_coords[i]])

	MASTER_COORDS.append(coords)
	m = m+1

print(len(MASTER_COORDS))



import numpy as np
from scipy.spatial.distance import euclidean

from fastdtw import fastdtw

g1 = np.array(MASTER_COORDS[0])
g2 = np.array(MASTER_COORDS[1])
g3 = np.array(MASTER_COORDS[2])
g4 = np.array(MASTER_COORDS[3])
tg = np.array(MASTER_COORDS[4]) #user input 

distance1, path = fastdtw(g1,g2, dist=euclidean)

distance2, path = fastdtw(g1,tg, dist=euclidean)

distance3, path = fastdtw(g2,tg, dist=euclidean)

distance4, path = fastdtw(g3,tg, dist=euclidean)

distance5, path = fastdtw(g4,tg, dist=euclidean)




print("acc=",(distance2-distance1)/distance2)
print("acc=",(distance3-distance1)/distance3)
print("acc=",(distance4-distance1)/distance1)
print("acc=",(distance1-distance5)/distance1)
