import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB


heights = [160, 165, 155, 172, 175, 180, 177, 190]
labels = ["F", "F", "F", "F", "B", "B", "B", "B", "B"]

bins = np.linspace(start=150, stop=190, num=4)
x_to_bins = np.digitize(heights, bins)
print(x_to_bins)
probabilityF = 0
probabilityB = 0
for i in range (9):
    if labels[i] == "F":
        probabilityF += 1
    else:
        probabilityB += 1
probabilityF /= 9
probabilityB /= 9
print(probabilityF)
print(probabilityB)
prob_x_f = []
prob_x_b = []
for i in range (1, 5):
    f = 0
    b = 0
    for j in range (len(x_to_bins)):
        if labels[j] == "F":
            f += 1
        else:
            b += 1
    prob_x_f.append(f / np.count_nonzero(x_to_bins == i)) #P(xi|F)
    prob_x_b.append(b / np.count_nonzero(x_to_bins == i)) #P(xi|B)

x = np.digitize(178, bins)
pcf = probabilityF * prob_x_f[x - 1]
pcb = probabilityB * prob_x_b[x - 1]
if pcf > pcb:
    print("F " + str(pcf))
else:
    print("B " + str(pcb))



