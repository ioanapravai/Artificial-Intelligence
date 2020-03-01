import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from skimage import io


file_path_train_img = "C:\\ANUL2\\Semestrul2\\IA\\laborator3\\data\\train_images.txt"
file_path_train_labels = "C:\\ANUL2\\Semestrul2\\IA\\laborator3\\data\\train_labels.txt"
file_path_test_img = "C:\\ANUL2\\Semestrul2\\IA\\laborator3\\data\\test_images.txt"
file_path_test_labels = "C:\\ANUL2\\Semestrul2\\IA\\laborator3\\data\\test_labels.txt"
train_images = np.loadtxt(file_path_train_img)
test_images = np.loadtxt(file_path_test_img)
train_labels = np.loadtxt(file_path_train_labels)
test_labels = np.loadtxt(file_path_test_labels)


#2
def values_to_bins(images, num_bins):
    bins = np.linspace(start=0, stop=255, num=num_bins)
    return np.digitize(images, bins) - 1

print("3: ")
#3
images_to_bins = values_to_bins(train_images, 5)
test_images_to_bins = values_to_bins(test_images, 5)
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(images_to_bins, train_labels)
naive_bayes_model.predict(test_images_to_bins)
print(naive_bayes_model.score(test_images_to_bins, test_labels))

print("4: ")
#4
for i in [3, 5, 7, 9, 11]:
    images_to_bins = values_to_bins(train_images, i)
    test_images_to_bins = values_to_bins(test_images, i)
    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(images_to_bins, train_labels)
    naive_bayes_model.predict(test_images_to_bins)
    print(naive_bayes_model.score(test_images_to_bins, test_labels))

print("5: ")
#5
images_to_bins = values_to_bins(train_images, 7)
test_images_to_bins = values_to_bins(test_images, 7)
naive_bayes_model = MultinomialNB()
naive_bayes_model.fit(images_to_bins, train_labels)
predicted = naive_bayes_model.predict(test_images_to_bins)
j = 0
for i in range(len(test_labels)):
    if test_labels[i] != predicted[i]:
        io.imshow(np.reshape(test_images[i], (28, 28)).astype(np.uint8))
        io.show()
        print(predicted[i])
        j += 1
        if j == 3:
            break

confusion_matrix = np.zeros((10, 10))
for i in range(10):
    for j in range(10):
        counter = 0
        for k in range(len(predicted)):
            if train_labels[k] == i and predicted[k] == j:
                counter += 1
        confusion_matrix[i,j] = counter
print(confusion_matrix)