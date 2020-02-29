import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def addImages():
    images = []
    for i in range (9):
        filePath = "car_" + str(i) + ".npy"
        image = np.load(filePath)
        images.append(image)
    images = np.asarray(images)
    return images


images = addImages()
print("Images: ")
print(images.shape)

def pixelSum():
    sum = 0
    for i in range(images.shape[0]):
        sum += np.sum(images[i, :, :])
    return sum


print("Suma valorilor pixelilor tuturor imaginilor: ")
print(pixelSum())


def pixelSumEachImage():
    sum = 0
    for i in range(images.shape[0]):
        print("Imaginea " + str(i))
        print(np.sum(images[i, :, :]))


print("Suma valorilor pixelilor pentru friecare imagine in parte: ")
pixelSumEachImage()

def maxPixelSum():
    sum = []
    for i in range(images.shape[0]):
        sum.append(np.sum(images[i, :, :]))
    return sum


sum = maxPixelSum()
print("Imaginea " + str(np.argmax(sum)) + " are suma pixelilor = " + str(np.max(sum)))


def pixelAverage():
    return np.mean(images, axis=0)

mean_image = pixelAverage()
io.imshow(mean_image.astype(np.uint8))
io.show()

std = np.std(images)
print(std)
normalization = (images-mean_image)/std
io.imshow(normalization[0].astype(np.uint8))
io.show()

for i in range(images.shape[0]):
    io.imshow(images[i, 200:300, 280:400])
    io.show()
