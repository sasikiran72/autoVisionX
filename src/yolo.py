
'''
You can run the file by using 

python yolo.py --image images/ipl.jpeg
'''


# importing the necessary packages
import numpy as np
import argparse
import time
import cv2
import os

# construct the argument parse
argument_parse = argparse.ArgumentParser()
argument_parse.add_argument("-i", "--image", required=True,
	help="path to input image")
argument_parse.add_argument("-c", "--confidence", type=float, default=0.9,
	help="minimum probability to filter weak detections, IoU threshold")
argument_parse.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applying non-maxima suppression")
args = vars(argument_parse.parse_args())


labelsPath = 'yolo-coco\\coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

# listing colors to the labels
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")

# coco dataset weights
weights_path = 'yolo-coco\\yolov3.weights'
config_path = 'yolo-coco\\yolov3.cfg'

# loading the 80 classes in our dataset
D_net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

# loading image
image = cv2.imread(args["image"])
(H, W) = image.shape[:2]


layer_names = D_net.getLayerNames()

output_layer_indices = D_net.getUnconnectedOutLayers()


output_layer_names = []
for i in output_layer_indices:
    layer_name=[]
    layer_name = layer_names[i - 1]
    output_layer_names.append(layer_name)



blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
D_net.setInput(blob)
layerOutputs = D_net.forward(output_layer_names)

# initializing our lists of detected bounding boxes

boxes = []
confidences = []
classIDs = []

# loop over each of the layer outputs
for output in layerOutputs:
	for detection in output:
		scores = detection[5:]
		classID = np.argmax(scores)
		confidence = scores[classID]

		if confidence > args["confidence"]:
			box = detection[0:4] * np.array([W, H, W, H])
			(centerX, centerY, width, height) = box.astype("int")


			x = int(centerX - (width / 2))
			y = int(centerY - (height / 2))

			boxes.append([x, y, int(width), int(height)])
			confidences.append(float(confidence))
			classIDs.append(classID)

# appling non-maxima suppression to suppress weak, overlapping bounding boxes
idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
	args["threshold"])

# ensure at least one detection exists
if len(idxs) > 0:
	# loop over the indexes we are keeping
	for i in idxs.flatten():
		# extract the bounding box coordinates
		(x, y) = (boxes[i][0], boxes[i][1])
		(w, h) = (boxes[i][2], boxes[i][3])

		# draw a bounding box rectangle and label on the image
		color = [int(c) for c in COLORS[classIDs[i]]]
		cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
		text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
		cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
			0.5, color, 2)

# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)
