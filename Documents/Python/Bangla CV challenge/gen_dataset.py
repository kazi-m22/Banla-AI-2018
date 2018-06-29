import numpy as np
import cv2
import os
import random
from scipy import ndimage


base_path = './input/'
print(os.getcwd())
input_folder = base_path+'training-b/'
output_folder = base_path+'training-h/'
# os.mkdir(output_folder)
# os.chdir(base_path+input_folder+'/')
files = os.listdir(input_folder)
# print(files)
for img in files:
    i = cv2.imread(input_folder+img,0)
    (thresh, im_bw) = cv2.threshold(i, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    index = np.where(im_bw == 0)
    coordinates = zip(index[0], index[1])
    coordinates = list(coordinates)
    rand_cor = random.sample(coordinates,3)
    counter =0
    for cor in rand_cor:
        if counter == 0:
            cv2.rectangle(i,(cor[1],cor[0]),(cor[1]+13,cor[0]+13),(0,0,0),-1)
        elif counter == 1:
            cv2.rectangle(i,(cor[1],cor[0]),(cor[1]+14,cor[0]+14),(0,0,0),-1)
        elif counter == 2:
            cv2.rectangle(i,(cor[1],cor[0]),(cor[1]+20,cor[0]+15),(0,0,0),-1)
        counter +=1

    counter = 0
    cv2.imwrite(output_folder+img, i)


input_folder = base_path+'training-i/'
output_folder = base_path + 'training-i/'
# os.mkdir('./input/training-i')
angles = [-45,-40,-30,-20,20,30,40,50]
files2=os.listdir(input_folder)
from PIL import Image
blur_cor = [7,10,15]
for img in files2:
    # image = cv2.imread(input_folder+img)
    # blur = cv2.blur(image,(random.choice(blur_cor),random.choice(blur_cor)))
    # cv2.imwrite(output_folder+img, blur)
    image = Image.open(input_folder+img)
    img2 = image.rotate(random.choice(angles))
    img2.save(output_folder+img)


# print(files)