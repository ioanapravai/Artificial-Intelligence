import numpy as np
from skimage import io
import matplotlib

def save_image ():  #a
    images = [] #lista de vectori
    for i in range (9):
        print("car_" + str(i) + ".npy")
        file_path = "car_" + str(i) + ".npy" #concatenam pt a citi
        images.append([])
        images[i] = np.load(file_path)
    print("Matricea tuturor imaginilor!!")
    print(images)
    return images

def sum_matrix_pixels (images): #b
    sum_all = np.sum(images) #functie pt calcularea sumei
    print("Suma pixeli tuturor imaginilor: ",sum_all)
    sum_image_alone = [0 for i in range(9)] #initializeaza vectorul cu 0 pt sume

    #c
    for i in range (9):
        sum_image_alone[i] = np.sum(images[i])
    print("Suma pixeli pt fiecare imagine: ",sum_image_alone)

    #d
    max = sum_image_alone[0]
    index_max = 0
    for i in range (1,9):
        if max <= sum_image_alone[i]:
           max = sum_image_alone[i]
           index_max = i
    print("Indexul imaginii cu suma maxima este: ",index_max)



    #e
    sum  = np.full((400,600), 0)
    for i in range (9):
        sum = np.add(sum, images[i])
    sum = sum/9
    io.imshow(sum.astype(np.uint8))
    io.show()
    return sum


def standard_dev (images, image_med):
    #f
    deviatia_standard = np.std(images)
    print("Deviatia standard: ", deviatia_standard)

    #e
    for i in range(9):
        images[i] = (images[i] - image_med)/deviatia_standard
    print("Normalizare", images)


def decupare (images): #h
    for i in range(9):
        from skimage import io
        io.imshow(images[i][200:300, 280:400].astype(np.uint8))
        io.show()

images = []
images = save_image ()
image_med = sum_matrix_pixels(images)
standard_dev(images, image_med)
decupare(images)