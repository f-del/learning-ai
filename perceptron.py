import random

#input 1
# X00
# 0X0
# 00X
# label / y = 1

#input 2
#00X
#0X0
#X00
# label / y = -1

inputs = [[1,0,0,
          0,1,0,
          0,0,1],
         [0,0,1,
          0,1,0,
          0,0,1]
        ]
labels = [1, -1]

nn_weight = [random.uniform(-1, 1) for _ in range(len(inputs[0]))]


def computeW(X, w, y_true, y_pred):
    w_updated = [0] * len(w)

    for i, _w in enumerate(w):
        w_updated[i] = _w + (y_true - y_pred) * X[i]

    return w_updated

def computeY(X, w):
    y = 0
    for i, x in enumerate(X):
        y += x * w[i]
    
    return max(-1, min(y, 1))


epochs = 100
for epoch in range(epochs):

    for i, input in enumerate(inputs):
        y_pred= computeY(input, nn_weight)

        y_true = labels[i]
        if y_pred != y_true:
            print("bad prediction", y_pred, "vs", y_true)
            nn_weight = computeW(input, nn_weight, y_true, y_pred)
            print(nn_weight)
        else:
            print("good prediction")