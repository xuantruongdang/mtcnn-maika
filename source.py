# face detection with mtcnn on a photograph
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN
import cv2
import math 
import numpy as np

# draw an image with detected objects
# def draw_image_with_boxes(filename, result_list):
# 	# load the image
# 	data = pyplot.imread(filename)
# 	# plot the image
# 	pyplot.imshow(data)
# 	# get the context for drawing boxes
# 	ax = pyplot.gca()
# 	# plot each box
# 	for result in result_list:
# 		# get coordinates
# 		x, y, width, height = result['box']
# 		# create the shape
# 		rect = Rectangle((x, y), width, height, fill=False, color='red')
# 		# draw the box
# 		ax.add_patch(rect)
# 	# show the plot
# 	pyplot.show()

def distance_to_camera(r, focalLength, R):
	# compute and return the distance from the maker to the camera
	return (r * focalLength) / R 

def calculate_R_bbox(width_bbox, height_bbox):
    return math.sqrt(width_bbox**2 + height_bbox**2) / 2

KNOW_WIDTH = 16.5
KNOW_HEIGHT = 27
r = math.sqrt(KNOW_WIDTH**2 + KNOW_HEIGHT**2) / 2
focalLength = 350.0

video_capture = cv2.VideoCapture(0)

# create the detector, using default weights
detector = MTCNN()

while True:
    ret, frame = video_capture.read()

    h, w, _ = frame.shape
    # preprocess img acquired
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 480)) 
    # img_mean = np.array([127, 127, 127])
    # img = (img - img_mean) / 128
    # img = np.transpose(img, [2, 0, 1])
    # img = np.expand_dims(img, axis=0)
    # img = img.astype(np.float32)

    faces = detector.detect_faces(img)

    for index, i in enumerate(faces):
        x, y, width, height = i['box']
        x1 = int(x) 
        x2 = int(x) + int(width)
        y1 = int(y)
        y2 = int(y) +int(height)
        # distance to camera
        R = calculate_R_bbox(width, height)
        cm = distance_to_camera(r, focalLength, R)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (80,18,236), 2)
        cv2.rectangle(frame, (x1, y2 - 20), (x2, y2), (80,18,236), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        text = f"face: {index}"
        cv2.putText(frame, text, (x1 + 6, y2 - 6), font, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "%.2fcm" % (cm),
		    (w - 200, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
		    1.0, (0, 255, 0), 3)

    cv2.imshow('Video', frame)
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()