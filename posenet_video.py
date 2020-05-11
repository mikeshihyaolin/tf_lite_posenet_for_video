'''
Shih-Yao (Mike) Lin
Date: 2020-05-11
Email: mike.lin@ieee.org
'''
import math
import time
from enum import Enum

import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import matplotlib.pyplot as plt

WIDTH = 257
HEIGHT = 257

class BodyPart(Enum):
	NOSE = 0,
	LEFT_EYE = 1,
	RIGHT_EYE = 2,
	LEFT_EAR = 3,
	RIGHT_EAR = 4,
	LEFT_SHOULDER = 5,
	RIGHT_SHOULDER = 6,
	LEFT_ELBOW = 7,
	RIGHT_ELBOW = 8,
	LEFT_WRIST = 9,
	RIGHT_WRIST = 10,
	LEFT_HIP = 11,
	RIGHT_HIP = 12,
	LEFT_KNEE = 13,
	RIGHT_KNEE = 14,
	LEFT_ANKLE = 15,
	RIGHT_ANKLE = 16,

class Position:
	def __init__(self):
		self.x = 0
		self.y = 0

class KeyPoint:
	def __init__(self):
		self.bodyPart = BodyPart.NOSE
		self.position = Position()
		self.score = 0.0

class Person:
	def __init__(self):
		self.keyPoints = []
		self.score = 0.0

class PoseNet:
	def __init__(self, model_path):
		self.input_mean = 127.5
		self.input_std = 127.5
		self.image_width = 0
		self.image_height = 0
		self.interpreter = tf.lite.Interpreter(model_path=model_path)
		self.interpreter.allocate_tensors()
		self.input_img = None
		self.input_details = self.interpreter.get_input_details()
		self.output_details = self.interpreter.get_output_details()
		print('input_details : ', self.input_details)
		print('output_details : ', self.output_details)

	def sigmoid(self, x):
		return 1. / (1. + math.exp(-x))

	def img_parsing(self,img):
		height, width = self.input_details[0]['shape'][1], self.input_details[0]['shape'][2]
		self.image_width, self.image_height, channels = img.shape
		# print('width, height = (', self.image_width, ',', self.image_height, ')')
		resize_image = cv2.resize(img, (width, height), interpolation = cv2.INTER_AREA)
		self.input_img = resize_image.copy()
		resize_image = np.asarray(resize_image)
		return np.expand_dims(resize_image, axis=0)

	def estimate_pose(self, img):
		input_data = self.img_parsing(img)

		if self.input_details[0]['dtype'] == type(np.float32(1.0)):
			input_data = (np.float32(input_data) - self.input_mean) / self.input_std

		self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

		self.interpreter.invoke()

		heat_maps = self.interpreter.get_tensor(self.output_details[0]['index'])
		offset_maps = self.interpreter.get_tensor(self.output_details[1]['index'])
		# print('heat_maps shape=', heat_maps.shape)
		# print('offset_maps shape=', offset_maps.shape)

		height = len(heat_maps[0])
		width = len(heat_maps[0][0])
		num_key_points = len(heat_maps[0][0][0])

		key_point_positions = [[0] * 2 for i in range(num_key_points)]
		for key_point in range(num_key_points):
			max_val = heat_maps[0][0][0][key_point]
			max_row = 0
			max_col = 0
			for row in range(height):
				for col in range(width):
					heat_maps[0][row][col][key_point] = self.sigmoid(heat_maps[0][row][col][key_point])
					if heat_maps[0][row][col][key_point] > max_val:
						max_val = heat_maps[0][row][col][key_point]
						max_row = row
						max_col = col
			key_point_positions[key_point] = [max_row, max_col]

		x_coords = [0] * num_key_points
		y_coords = [0] * num_key_points
		confidenceScores = [0] * num_key_points
		for i, position in enumerate(key_point_positions):
			position_y = int(key_point_positions[i][0])
			position_x = int(key_point_positions[i][1])
			y_coords[i] = (position[0] / float(height - 1) * self.image_height +
			               offset_maps[0][position_y][position_x][i])
			x_coords[i] = (position[1] / float(width - 1) * self.image_width +
			               offset_maps[0][position_y][position_x][i + num_key_points])
			confidenceScores[i] = heat_maps[0][position_y][position_x][i]
			# print("confidenceScores[", i, "] = ", confidenceScores[i])

		person = Person()
		key_point_list = []
		for i in range(num_key_points):
			key_point = KeyPoint()
			key_point_list.append(key_point)
		total_score = 0
		for i, body_part in enumerate(BodyPart):
			key_point_list[i].bodyPart = body_part
			key_point_list[i].position.x = x_coords[i]
			key_point_list[i].position.y = y_coords[i]
			key_point_list[i].score = confidenceScores[i]
			total_score += confidenceScores[i]

		person.keyPoints = key_point_list
		person.score = total_score / num_key_points

		return person

def run_video(posenet, body_joints):

	cap = cv2.VideoCapture(0)

	MIN_CONFIDENCE = 0.40
	# Initialize frame rate calculation
	frame_rate_calc = 1
	freq = cv2.getTickFrequency()

	while(True):
		# Start timer (for calculating frame rate)
		t1 = cv2.getTickCount()

		# Capture frame-by-frame
		ret, frame = cap.read()

		height, width, _ = frame.shape
		h_r = height/257
		w_r = width/257

		res = frame.copy()

		frame = cv2.resize(frame,(257,257))

		person = posenet.estimate_pose(frame)

		for line in body_joints:
			if person.keyPoints[line[0].value[0]].score > MIN_CONFIDENCE and person.keyPoints[line[1].value[0]].score > MIN_CONFIDENCE:
				start_point_x, start_point_y = int(person.keyPoints[line[0].value[0]].position.x), int(person.keyPoints[line[0].value[0]].position.y)
				end_point_x, end_point_y = int(person.keyPoints[line[1].value[0]].position.x), int(person.keyPoints[line[1].value[0]].position.y)


				start_point_x = int(start_point_x*w_r)
				start_point_y = int(start_point_y*h_r)
				end_point_x = int(end_point_x*w_r)
				end_point_y = int(end_point_y*h_r)

				cv2.line(res,(start_point_x,start_point_y),(end_point_x,end_point_y),(0,0,255),5)

		for key_point in person.keyPoints:
			if key_point.score > MIN_CONFIDENCE:
				key_point.position.x = int(key_point.position.x*w_r)
				key_point.position.y = int(key_point.position.y*h_r)
				cv2.circle(res,(int(key_point.position.x),int(key_point.position.y)), 10, (0,255,0), -1)

		# Calculate framerate
		t2 = cv2.getTickCount()
		time1 = (t2-t1)/freq
		frame_rate_calc= 1/time1

		# Draw framerate in corner of frame
		cv2.putText(res,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)

		cv2.imshow("res", res)
		
		# Display the resulting frame
		# cv2.imshow('frame',frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()
def main():
	body_joints = [[BodyPart.LEFT_WRIST, BodyPart.LEFT_ELBOW],
	               [BodyPart.LEFT_ELBOW, BodyPart.LEFT_SHOULDER],
	               [BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER],
	               [BodyPart.RIGHT_SHOULDER, BodyPart.RIGHT_ELBOW],
	               [BodyPart.RIGHT_ELBOW, BodyPart.RIGHT_WRIST],
	               [BodyPart.LEFT_SHOULDER, BodyPart.LEFT_HIP],
	               [BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP],
	               [BodyPart.RIGHT_HIP, BodyPart.RIGHT_SHOULDER],
	               [BodyPart.LEFT_HIP, BodyPart.LEFT_KNEE],
	               [BodyPart.LEFT_KNEE, BodyPart.LEFT_ANKLE],
	               [BodyPart.RIGHT_HIP, BodyPart.RIGHT_KNEE],
	               [BodyPart.RIGHT_KNEE, BodyPart.RIGHT_ANKLE]]
	
	posenet = PoseNet(model_path="./models/posenet_mobilenet_v1_100_257x257_multi_kpt_stripped.tflite")

	run_video(posenet, body_joints)

if __name__ == '__main__':
	main()
	
