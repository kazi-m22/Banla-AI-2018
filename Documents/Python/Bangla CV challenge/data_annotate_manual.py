import cv2
import os
import matplotlib.pyplot as plt
import pandas as pd
files = os.listdir('./input/testing-g/')
numbers = []
names = []

for i in files:
	img = plt.imread('./input/testing-g/'+i)
	names.append(i)
	plt.imshow(img)
	plt.show()
	p = input('enter number ')
	numbers.append(p)

for i in range(len(numbers)):
	numbers[i]=[names[i], numbers[i]]
data = pd.DataFrame(numbers)
data.to_csv('1.csv', index=False, header = False)