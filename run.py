
# Part 1 - Import dog images

from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import os

# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

cwd = os.path.dirname(__file__) 
print ('Running in:' + cwd)

# load train, test, and validation datasets
train_files, train_targets = load_dataset(cwd + os.sep +'dogImages'+ os.sep +'train')
valid_files, valid_targets = load_dataset(cwd + os.sep +'dogImages'+ os.sep +'valid')
test_files, test_targets = load_dataset(cwd + os.sep +'dogImages'+ os.sep +'test')

# load list of dog names
dog_names = [item[20:-1] for item in sorted(glob("dogImages"+ os.sep +"train"+ os.sep +"*"+ os.sep +""))]

# print statistics about the dataset
print('There are %d total dog categories.' % len(dog_names))
print('There are %s total dog images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training dog images.' % len(train_files))
print('There are %d validation dog images.' % len(valid_files))
print('There are %d test dog images.'% len(test_files))

print("Part 2 - Import Human Dataset")

import random
random.seed(8675309)

# load filenames in shuffled human dataset
# human_files = np.array(glob(cwd + os.sep +"lfw" + os.sep +"*"+ os.sep +"*"))
human_files = np.array(glob(cwd + "C:/AI_Engineer/Term2/Project1_dog-project/dog-project/lfw/*/*"))

imgs = []
path = cwd + os.sep +"lfw" + os.sep 
valid_images = [".jpg",".gif",".png",".tga"]

print("Files:" + ''.join(os.listdir(path)))

for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue
    imgs.append(os.path.join(path,f))
    # imgs.append(Image.open(os.path.join(path,f))

# random.shuffle(imgs)
random.shuffle(human_files)

# print statistics about the dataset
print('There are %d total human images.' % len(human_files))
print('In the second array, There are %d total human images.' % len(imgs))

# ################################################################
print("\nStep 1 - Detect Humans")

import cv2                
import matplotlib.pyplot as plt                        
# %matplotlib inline                               

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier(cwd + os.sep +'haarcascades'+ os.sep +'haarcascade_frontalface_alt.xml')

# load color (BGR) image
img = cv2.imread(human_files[3])
#img = cv2.imread(imgs[3])
# convert BGR image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# find faces in image
faces = face_cascade.detectMultiScale(gray)

# print number of faces detected in the image
print('Number of faces detected:', len(faces))

# get bounding box for each detected face
for (x,y,w,h) in faces:
    # add bounding box to color image
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    
# convert BGR image to RGB for plotting
cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display the image, along with bounding box
plt.imshow(cv_rgb)
plt.show()