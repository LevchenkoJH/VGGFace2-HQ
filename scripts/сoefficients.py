#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import kendalltau
import time



# In[2]:


def load_image(path, title='', show=False):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if show:
        show_image(image, path)
    return image


# In[3]:


def show_image(image, title='', size=(15, 15)):
    plt.figure(figsize = size)
    if title != '':
        plt.title(title)
    plt.imshow(image)
    plt.show()
    plt.clf()


# # Коэффициент Танимото

# In[4]:


# От -1 до 1
def tanimoto_coefficient(image1, image2, show=False, prin=False):
    # Нужно перейти из RGB в яркостное представление
    # Возможно для тензоров нельзя использовать методы OpenCV
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    if show:
        show_image(image1, "gray image1")
        show_image(image2, "gray image2")
    
    S1 = np.sum(image1 * image2)
    S2 = np.sum((image1 - image2) ** 2)
    St = S1 / (S1 + S2)
    if prin:
        print("Коэффициент Танимото ->", St)
    return St


# # Кросс-корреляция изображений

# In[5]:


# От -1 до 1
def cross_correlation_coefficient(image1, image2, show=False, prin=False):
    # Нужно перейти из RGB в яркостное представление
    # Возможно для тензоров нельзя использовать методы OpenCV
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    if show:
        show_image(image1, "gray image1")
        show_image(image2, "gray image2")
    
    # Объем локальной выборки
    h, w = image1.shape
    m = h * w
#     print("m ->", m)
    # Оценкка математического ожидания изображения
    M1 = np.sum(image1) / m
    M2 = np.sum(image2) / m
#     print("M1 ->", M1)
#     print("M2 ->", M2)
    # Оценка дисперсии изображения
    D1 = np.sqrt(np.sum((image1 - M1) ** 2) / m)
    D2 = np.sqrt(np.sum((image2 - M2) ** 2) / m)
#     print("D1 ->", D1)
#     print("D2 ->", D2)
    
    r = np.sum((image1 - M1) * (image2 - M2)) / (m * D1 * D2)
    
    if prin:
        print("Кросс-корреляция ->", r)
    return r


# # Коэффициент ранговой корреляции Кендалла

# In[6]:


def kendall_coefficient(image1, image2, show=False, prin=False):
    # Нужно перейти из RGB в яркостное представление
    # Возможно для тензоров нельзя использовать методы OpenCV
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    if show:
        show_image(image1, "gray image1")
        show_image(image2, "gray image2")
        
        
    image1 = image1.reshape(image1.shape[0]*image1.shape[1])
    image2 = image2.reshape(image2.shape[0]*image2.shape[1])
    
    # Количество инверсий
    R = 0
    
    
    
    coef, _ = kendalltau(image1, image2)
    
    if prin:
        print('Коэффициент ранговой корреляции Кендалла ->', coef)
    
    return coef


# # Одна пара изображений

# In[7]:


# def main():
#     IN_PATH = "/home/jasmine/Tanimoto_Coefficient/input"
#
#     file_names = sorted(os.listdir(IN_PATH))
#
#     print(file_names)
#
#     image_path_1 = os.path.join(IN_PATH, file_names[0])
#     image_path_2 = os.path.join(IN_PATH, file_names[2])
#     print(image_path_1)
#     print(image_path_2)
#
#
#     image_1 = load_image(image_path_1)
#     image_2 = load_image(image_path_2)
#
#     print('\n')
#     time_start = time.time()
#     buf1 = tanimoto_coefficient(image_1, image_2, prin=True)
#     time_end = time.time()
#     print("Время работы:", time_end - time_start, '\n')
#
#     time_start = time.time()
#     buf2 = cross_correlation_coefficient(image_1, image_2, prin=True)
#     time_end = time.time()
#     print("Время работы:", time_end - time_start, '\n')
#
#     time_start = time.time()
#     buf3 = kendall_coefficient(image_1, image_2, prin=True)
#     time_end = time.time()
#     print("Время работы:", time_end - time_start, '\n')
#
# main()


# # Множество пар изображений

# In[8]:


# def many_test(IN_PATH, file_names):
#     previous_file = ""
#
#     points1 = np.array([0])
#     points2 = np.array([0])
#     points3 = np.array([0])
#
#     for file in file_names:
#         if (previous_file != ""):
# #             print(previous_file + " " + file)
#             image1 = load_image(path=os.path.join(IN_PATH, previous_file), show=False)
#             image2 = load_image(path=os.path.join(IN_PATH, file), show=False)
#
#             buf1 = tanimoto_coefficient(image1, image2, show=False)
#             buf2 = cross_correlation_coefficient(image1, image2)
#             buf3 = kendall_coefficient(image1, image2)
#
#             points1 = np.append(points1, buf1)
#             points2 = np.append(points2, buf2)
#             points3 = np.append(points3, buf3)
#
#
#
#         previous_file = file
#
#
#     plt.figure(figsize = (15, 5))
#     plt.axis([0, 228, 0.6, 1.0])
#     plt.plot(range(len(file_names)), points1, 'r')
#     plt.plot(range(len(file_names)), points2, 'g')
#     plt.plot(range(len(file_names)), points3, 'b')
#     plt.legend(['Коэффициент Танимото',
#                 'Кросс-корреляция изображений',
#                 'Коэффициент ранговой корреляции Кендалла'], loc=0)
#     plt.show()


# In[9]:


# def main():
#     IN_PATH = "/home/jasmine/Tanimoto_Coefficient/_CuZqXrhEZI_5.mp4222"
#     file_names = sorted(os.listdir(IN_PATH))
#     many_test(IN_PATH, file_names)
# main()


# In[10]:


# def main():
#     IN_PATH = "/home/jasmine/Tanimoto_Coefficient/_CuZqXrhEZI_5.mp4"
#     file_names = sorted(os.listdir(IN_PATH))
#     many_test(IN_PATH, file_names)
# main()


# In[ ]:




