import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


def pose_to_matrix(x, y, theta):
    #Превращает координаты и угол в гомогенную матрицу 3x3 (SE2)
    return np.array([
        [np.cos(theta), -np.sin(theta), x],
        [np.sin(theta),  np.cos(theta), y],
        [0,              0,             1]
    ])

def matrix_to_pose(T):
    #Извлекает x, y, theta из матрицы 3x3
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    return x, y, theta

def exp_map(arr):
    return pose_to_matrix(arr[0], arr[1], arr[2])

def log_map(T):
    x, y, theta = matrix_to_pose(T)

    #Нормализация угла в диапазон [-pi, pi]
    theta = (theta + np.pi) % (2 * np.pi) - np.pi
    return np.array([x, y, theta])

#Настройка симуляции
np.random.seed(42)
n_steps = 40 
step_dist = 1.0
#Шум: по x и y небольшой, по углу (theta) — критичный для передвижения
noise_std = [0.03, 0.03, 0.02] 

gt_poses = [pose_to_matrix(0, 0, 0)]
noisy_poses = [pose_to_matrix(0, 0, 0)]
measurements = [] #ребра графа (Z_ij)

#Цикл движения робота
for i in range(n_steps):
    # Идеальный поворот на 90 градусов каждые 10 шагов
    d_theta = np.pi/2 if (i+1) % 10 == 0 else 0
    delta_T_ideal = pose_to_matrix(step_dist, 0, d_theta)
    
    gt_poses.append(gt_poses[-1] @ delta_T_ideal)
    
    #Добавляем шум Z_ij = delta_T * exp(eta)
    eta = np.random.normal(0, noise_std)
    Z_ij = delta_T_ideal @ exp_map(eta)
    measurements.append((i, i+1, Z_ij))
    
    #Накапливаем шумную траекторию
    noisy_poses.append(noisy_poses[-1] @ Z_ij)


Z_loop = pose_to_matrix(0, 0, 0)
measurements.append((n_steps, 0, Z_loop))

#Оптимизация PGO
#Задаем веса 
#Вес 50.0 для угла заставляет оптимизатор сохранять форму квадрата
weights = np.array([1.0, 1.0, 50.0]) 

def residuals_weighted(params):
    current_poses = params.reshape(-1, 3)
    all_errors = []
    
    #Фиксируем первую точку в (0,0,0), Без этого вся траектория может вращаться вокруг своей оси
    first_pose_error = (current_poses[0] - np.array([0, 0, 0])) * 100.0
    all_errors.extend(first_pose_error)
    
    for i, j, Z_ij in measurements:
        Ti = pose_to_matrix(*current_poses[i])
        Tj = pose_to_matrix(*current_poses[j])
        
        #r_ij = log(Z_ij^-1 * (T_i^-1 * T_j))
        error_mat = np.linalg.inv(Z_ij) @ (np.linalg.inv(Ti) @ Tj)
        r_ij = log_map(error_mat)
        
        #Взвешиваем ошибку
        all_errors.extend(r_ij * weights)
        
    return np.array(all_errors)

#Начальное приближение — наша кривая траектория
initial_guess = np.array([matrix_to_pose(T) for T in noisy_poses]).flatten()

#Запуск метода Левенберга-Марквардта
opt_result = least_squares(residuals_weighted, initial_guess, method='lm')
optimized_coords = opt_result.x.reshape(-1, 3)

#--ВИЗУАЛИЗАЦИЯ--
gt_coords = np.array([matrix_to_pose(T) for T in gt_poses])
noisy_coords = np.array([matrix_to_pose(T) for T in noisy_poses])

plt.figure(figsize=(10, 8))
plt.plot(gt_coords[:, 0], gt_coords[:, 1], 'g--', alpha=0.6, label='Без шума')
plt.plot(noisy_coords[:, 0], noisy_coords[:, 1], 'r-o', markersize=3, alpha=0.3, label='с Шумом')
plt.plot(optimized_coords[:, 0], optimized_coords[:, 1], 'b-s', markersize=4, label='Оптимизация PGO')

plt.legend()
plt.axis('equal')
plt.grid(True)
plt.title("Визуализация восстановленой траектории по относительному движению")
plt.show()