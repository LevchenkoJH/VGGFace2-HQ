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


# На вход название обрабатываемого видео
def align_image_dir(dir_name_tmp):
    # Расположение видео
    ori_path_tmp = os.path.join(input_dir, dir_name_tmp)
    print("Путь к видео ->", ori_path_tmp)

    # Где будет сохранен результат выравнивания (Видео)
    save_dir_ffhqalign = os.path.join(output_dir_ffhqalign, dir_name_tmp)
    print("save_dir_ffhqalign ->", save_dir_ffhqalign)

    if not os.path.exists(output_dir_ffhqalign):
        os.makedirs(output_dir_ffhqalign)

    # Расшариваем видео в буфере (frames)
    video_path_tmp = ori_path_tmp # os.path.join(input_dir, video_path)
    videoCapture = cv2.VideoCapture()
    videoCapture.open(video_path_tmp)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    out = cv2.VideoWriter(save_dir_ffhqalign,
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (crop_size, crop_size))

    for i in range(int(frames)):
        ret, frame = videoCapture.read()

        # Если изображение слишком маленькое
        if frame.shape[0]<250 or frame.shape[1]<250:
            continue

        # Выравниваем кадр
        frame = app.get(frame, crop_size, mode=mode)

        # Сохранение
        if len(frame) == 0:
            continue
        else:
            print(frame[0].shape)
            out.write(frame[0])

    out.release()

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

    #
    # print("do DIRS", input_dir)
    dirs = sorted(os.listdir(input_dir))
    # print("posle Dirs", dirs)




    norm_images = sorted(os.listdir(output_dir_ffhqalign))
    # print(norm_images)





    # Список папок с изображениями
    # Нужно чтобы в место папок были видео
    handle_dir_list = dirs



    # Перебор по папкам
    # Нужен перебор по видео
    for handle_dir_list_tmp  in tqdm(handle_dir_list):
        print("Выравнивание видео ->", handle_dir_list_tmp)

        if handle_dir_list_tmp in norm_images:
            print("Видео уже обработано")
        else:
            align_image_dir(handle_dir_list_tmp)