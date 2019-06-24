# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import argparse
import dlib
import imutils
'''import Centroid'''
facial_features_cordinates = {}


coordinates_values  = []

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_INDEXES = OrderedDict([
	("Mouth", (48, 68)),
	("Right_Eyebrow", (17, 22)),
	("Left_Eyebrow", (22, 27)),
	("Right_Eye", (36, 42)),
	("Left_Eye", (42, 48)),
	("Nose", (27, 35)),
	("Jaw", (0, 17))
])

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())
mouth_coordinates=[]

def shape_to_numpy_array(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coordinates = np.zeros((68, 2), dtype=dtype)

	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coordinates[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coordinates
x=[]
y=[]
mouthlist=[]
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()

	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
				  (168, 100, 168), (158, 163, 32),
				  (163, 38, 32), (180, 42, 220)]

	# loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_INDEXES.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_INDEXES[name]
		pts = shape[j:k]
		facial_features_cordinates[name] = pts

		# check if are supposed to draw the jawline
		if name == "Jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)

		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	# return the output image
	#print(facial_features_cordinates["Mouth"])
	#for i in range(len(facial_features_cordinates)):
	mouth_coordinates=facial_features_cordinates["Mouth"]
  
	for i in range(len(mouth_coordinates)):
		mouthlist.append(mouth_coordinates[i].tolist())
 
	#print("centroid is:", Centroid.centroid(mouthlist))
	for i in mouthlist:
		x.append(i[0])
		y.append(i[1])
	
	centroid_x=sum(x)/len(x)
	centroid_y=sum(y)/len(y)
	
	#print (centroid_x, centroid_y)
	coordinates_values.append([centroid_x, centroid_y]) ;
	
	
	
	
# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# load the input image, resize it, and convert it to grayscale


n=0
#impath='C:\Users\mridu\OneDrive\Desktop\dotslash'
fno = 1


while (n<250):
	image = cv2.imread("Database/v5/frame"+str(n)+".jpg") 
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	n = n+1

	  

# detect faces in the grayscale image
	rects = detector(gray, 1)

# loop over the face detections
	for (i, rect) in enumerate(rects):
		 # determine the facial landmarks for the face region, then
		 # convert the landmark (x, y)-coordinates to a NumPy array
		 shape = predictor(gray, rect)
		 shape = shape_to_numpy_array(shape)

	output = visualize_facial_landmarks(image, shape)
	print(coordinates_values)
	with open('v5.txt', 'w') as f:
		for item in coordinates_values:
			for sub_item in item:
				f.write("%d " % sub_item)
			f.write("\n")
		

	#cv2.imshow("Image", output)
	cv2.waitKey(0)



	
	
	
	
	
	