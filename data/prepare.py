import os
import sys

def prepare():
	data_path = "_data/"
	data_seq = "_data/frame_sequence.txt"


	running_video = os.listdir(data_path + "running/")
	print(running_video)