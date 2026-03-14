import math;
import numpy as np; 
# Зададим массивы ориентиров и истиную геометрическую позицию объекта
# предположим, что у нас 5 ориентиров
landmarks = [(2, 1), (10, 4),(5, 7),(-19, 10),(0, 15)]

p_true = (0, 0)

r_true = []
r_measured = []
sigma = 0.5
for obj in range(len(landmarks)):
    temp = math.sqrt((p_true[0] - landmarks[obj][0])**2 +
                            (p_true[1] - landmarks[obj][1])**2)
    r_true.append(temp)
    #Зададим гауссов шум
    eps = np.random.normal(0, sigma)
    r_measured.append(r_true[obj] + eps)
    

#Выберем первый ориентир в качестве опорного
(x1, y1) = landmarks[0]
r1 = r_measured[0]

#Зададим матрицу коэфицентов и вектор значений
A = np.zeros((len(landmarks)-1, 2))
B = np.zeros((len(landmarks)-1,))

for i in range(1, len(landmarks)):
    #Заполняем матрицу по формуле, но я домножил на -1
    A[i-1, 0] = 2 * (landmarks[i][0] - x1)
    A[i-1, 1] = 2 * (landmarks[i][1] - y1)

    #Заполняем вектор
    B[i-1] = r1**2 - r_measured[i]**2 + (landmarks[i][0]**2 - x1**2) + (landmarks[i][1]**2 - y1**2)

object_ = tuple(np.linalg.lstsq(A, B, rcond=None)[0])

print("Оценка позиии", object_)
print("Реальная позиция", p_true)