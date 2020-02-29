y_pred = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]
y_true = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
n = len(y_pred)
def accuracy_score(y_pred, y_true):
    s = 0
    for it in range(n):
        if y_pred[it] == y_true[it]:
            s += 1
    return s/n

accuracy = accuracy_score(y_pred, y_true)
print("accuracy: ")
print(accuracy)

def precision_recall_score(y_pred, y_true):
    s = [0, 0, 0]
    for it in range(len(y_pred)):
        if y_pred[it] == 1 and y_true[it] == 1:
            s[0] += 1
        if y_pred[it] == 1 and y_true[it] == 0:
            s[1] += 1
        if y_pred[it] == 0 and y_true[it] == 1:
            s[2] += 1
    precision = s[0]/(s[0] + s[1])
    recall = s[0]/(s[0] + s[2])
    return precision, recall

prec, rec = precision_recall_score(y_pred, y_true)
print("precision: ")
print(prec)
print("recall: ")
print(rec)

def mse(y_true, y_pred):
    x = 0
    z = 0
    for it in range(len(y_pred)):
        x = y_pred[it] - y_true[it]
        x = x * x
        z += x
    return z/n
m = mse(y_true, y_pred)
print("mse:")
print(m)

def mae(y_true, y_pred):
    y = 0
    for it in range(len(y_pred)):
        y = y + abs(y_pred[it] - y_true[it])
    return y/n

ma = mae(y_true, y_pred)
print("mae")
print(ma)



