'''
Author: Naiyuan liu
Github: https://github.com/NNNNAI
Date: 2021-11-16 15:34:14
LastEditors: Naiyuan liu
LastEditTime: 2021-11-19 16:14:01
Description:
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import argparse
import cv2
import glob
from tqdm import tqdm
from insightface_func.face_detect_crop_ffhq_newarcAlign import Face_detect_crop
import argparse
import numpy as np

from сoefficients import tanimoto_coefficient
from сoefficients import cross_correlation_coefficient
from сoefficients import kendall_coefficient

def video_to_frames(input_dir, output_dir):
    dirs = sorted(os.listdir(input_dir))
    print(dirs)

    for video_path in tqdm(dirs):
        print("Обработка", video_path)

        # Расшариваем видео в буфере (frames)
        video_path_tmp = os.path.join(input_dir, video_path)
        videoCapture = cv2.VideoCapture()
        videoCapture.open(video_path_tmp)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Буфер изображений
        # Из него выбираем кадры, которые будут сохранены для датасета
        buf = np.array([])

        # Структура
        # [коэффициент корреляции, индекс первого изображения, индекс второго изображения]
        coef_pair_frame_array = np.array([])
        for i in range(int(frames)):
            ret, frame = videoCapture.read()
            # Заносим кадры в буффер для дальнейшей обработки
            if (len(buf) == 0):

                # Если изображение слишком маленькое
                if frame.shape[0] < 250 or frame.shape[1] < 250:
                    print("------------------------------------Изображение слишком маленькое------------------------------------", i)
                    frame = []
                else:
                    # Выравниваем кадр
                    frame = app.get(frame, crop_size, mode=mode)
                # Проверка
                if len(frame) == 0:
                    print("------------------------------------Пустое изображение------------------------------------", i)
                    # Для избавления от возни с индексами, в буфер отправляется черная картинка
                    frame = [np.zeros((crop_size, crop_size, 3), dtype=np.uint8)]
                buf = np.expand_dims(frame[0], axis=0)
            else:
                # Если изображение слишком маленькое
                if frame.shape[0] < 250 or frame.shape[1] < 250:
                    print(
                        "------------------------------------Изображение слишком маленькое------------------------------------", i)
                    frame = []
                else:
                    # Выравниваем кадр
                    frame = app.get(frame, crop_size, mode=mode)

                # Проверка
                if len(frame) == 0:
                    print("------------------------------------Пустое изображение------------------------------------", i)
                    # Для избавления от возни с индексами, в буфер отправляется черная картинка
                    frame = [np.zeros((crop_size, crop_size, 3), dtype=np.uint8)]

                print(buf.shape)
                print(np.expand_dims(frame[0], axis=0).shape)

                buf = np.concatenate((buf, np.expand_dims(frame[0], axis=0)), axis=0)

                # Вычисляем корреляцию
                # Нужно проверить, что обе картинки не черные
                if buf[i - 1][buf[i - 1][:] != 0].shape[0] != 0 and buf[i][buf[i][:] != 0].shape[0] != 0:
                    # Тогда вычисляем коэффициент корреляции
                    coef_pair_frame = np.array([kendall_coefficient(buf[i - 1], buf[i]), i - 1, i])
                else:
                    # Иначе коэффициент корреляции равен -1
                    coef_pair_frame = np.array([-1, i - 1, i])

                # Записываем корреляцию для дальнейшего анализа
                if (len(coef_pair_frame_array) == 0):
                    coef_pair_frame_array = np.expand_dims(coef_pair_frame, axis=0)
                else:
                    coef_pair_frame_array = np.concatenate((coef_pair_frame_array, np.expand_dims(coef_pair_frame, axis=0)), axis=0)

        # Нужно найти пары кадров с подходящей корреляцией
        # К какой корреляции должны стремиться, выбираемые кадры
        need_coef = 0.985 # Нужно указывать в стартовых параметрах

        # Сколько пар кадров берем из одного видео
        need_count = 5
        # На столько частей нам нужно поделить видео
        # и выбрать в каждой части пару наиболее близкую к need_coef

        # Проходим по участкам одинаковой длины
        for i in range(need_count):
            # Вычисляем длинну участка
            frames_range_a = len(coef_pair_frame_array) // need_count + 1
            frames_range_b = frames_range_a
            # Проверяем что не выходим за границы массива
            if i * frames_range_b + frames_range_b - 1 > len(coef_pair_frame_array) - 1:
                # Вычисляем новую длину
                frames_range_b += len(coef_pair_frame_array) - i * frames_range_b - frames_range_b
                # print("NEW frames_range ->", frames_range_b)

            print(f"({i * frames_range_a}, { i * frames_range_a + frames_range_b - 1 })")

            # Нужный срез
            frames_coef = coef_pair_frame_array[i * frames_range_a : i * frames_range_a + frames_range_b]

            # Находим запись с корреляцией, абсолютное значение разности которого, с нужным коэффициентом - минимально
            condition = np.abs(frames_coef[:, 0] - need_coef)
            frames_coef = frames_coef[condition == np.min(condition)]

            # Нужно учесть что в срезе может не быть ни одной пары
            if frames_coef[0][0] != -1:
                # Сохраняем выделенную пару в соответствующей папке
                direction = os.path.join(output_dir, video_path, str(i).zfill(3) + " coef:" + str(frames_coef[0][0])[:6])
                if not os.path.exists(direction):
                    os.makedirs(direction)

                # Сохраняем пару кадров
                print(os.path.join(direction, str(int(frames_coef[0][1])).zfill(5)))
                print(os.path.join(direction, str(int(frames_coef[0][2])).zfill(5)))
                cv2.imwrite(os.path.join(direction, str("%s.png" % str(int(frames_coef[0][1])).zfill(5))), buf[int(frames_coef[0][1])])
                cv2.imwrite(os.path.join(direction, str("%s.png" % str(int(frames_coef[0][2])).zfill(5))), buf[int(frames_coef[0][2])])

# Для выравнивания датасета
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',type=str,default = '../Data/VGGface2/train')
    parser.add_argument('--output_dir_ffhqalign',type=str,default = '../Data/VGGface2_FFHQalign')
    # parser.add_argument('--crop_size',type=int,default = 256)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--mode',type=str,default = 'ffhq',choices=['ffhq','newarc','both'])

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir_ffhqalign = args.output_dir_ffhqalign
    crop_size = args.crop_size
    mode = args.mode

    app = Face_detect_crop(name='antelope', root='../insightface_func/models')

    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(320,320))

    dirs = sorted(os.listdir(input_dir))

    ready_video = sorted(os.listdir(output_dir_ffhqalign))

    video_to_frames(input_dir, output_dir_ffhqalign)