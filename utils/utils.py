import os
import glob


def most_recently_modified_directory(directory):
	files = glob.glob(os.path.join(directory, '*'))
	files.sort(key=os.path.getmtime)
	return files[-1]


def most_recently_modified_policy(directory):
	files = glob.glob(os.path.join(f'{directory}/policies', '*'))
	files = [file for file in files if 'policy' in file]
	files.sort(key=os.path.getmtime)
	return files[-1]
