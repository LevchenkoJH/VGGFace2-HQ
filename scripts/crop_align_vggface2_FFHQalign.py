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
    print("------------------------------------------------------align_image_dir------------------------------------------------------")
    # Расположение видео
    ori_path_tmp = os.path.join(input_dir, dir_name_tmp)
    print("Путь к видео ->", ori_path_tmp)








    # # Расположения изображений
    # # Нужно заменить на разложение видео на кадры и их хранение в оперативной памяти
    # image_filenames = glob.glob(os.path.join(ori_path_tmp,'*'))
    # print("image_filenames ->", image_filenames)







    # Где будет сохранен результат выравнивания (Папка)
    # Папку заменяем на видео
    save_dir_ffhqalign = os.path.join(output_dir_ffhqalign, dir_name_tmp)
    print("save_dir_ffhqalign ->", save_dir_ffhqalign)

    if not os.path.exists(output_dir_ffhqalign):
        os.makedirs(output_dir_ffhqalign)







    ##############################################################################################################
    # Расшариваем видео в буфере (frames)
    video_path_tmp = ori_path_tmp # os.path.join(input_dir, video_path)
    videoCapture = cv2.VideoCapture()
    videoCapture.open(video_path_tmp)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

    # output_video_dir = os.path.join(output_dir, video_path)
    # if not os.path.exists(output_video_dir):
    #     os.makedirs(output_video_dir)












    # Буфер
    buf = np.array([])
    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        # print(i)
        # print("frames.shape ->", type(frame))

        # Выравниваем кадр
        # Если изображение слишком маленькое
        if frame.shape[0]<250 or frame.shape[1]<250:
            continue

        # Выравнивание изображения
        frame = app.get(frame, crop_size, mode=mode)



        # Заносим кадры в буффер
        if (len(buf) == 0):
            buf = np.expand_dims(frame, axis=0)
            # print("IOK")
        else:
            buf = np.concatenate((buf, np.expand_dims(frame, axis=0)), axis=0)
        # print(i + 1, "->", buf.shape)
    ##############################################################################################################

    out = cv2.VideoWriter(save_dir_ffhqalign,
                          cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, (crop_size, crop_size))

    for i in range(len(buf)):
        print(buf[i][0].shape)
        out.write(buf[i][0])
    out.release()












    # # Где будет сохранен результат выравнивания (Папка)
    # # Папку заменяем на видео
    # save_dir_ffhqalign = os.path.join(output_dir_ffhqalign, dir_name_tmp)
    # print("save_dir_ffhqalign ->", save_dir_ffhqalign)
    #
    # # НУЖНО БУДЕТ ЗАКОММЕНТИРОВАТЬ
    # if not os.path.exists(save_dir_ffhqalign):
    #     os.makedirs(save_dir_ffhqalign)
    #
    #
    # for file in image_filenames:
    #     # file = ../Data/VGGface2/train/_CuZqXrhEZI_5.mp4/00005.png
    #     print("-------------------------------------------------------------------------")
    #
    #     # Название обрабатываемог оизображения
    #     image_file = os.path.basename(file)
    #     print("image_file =", image_file)
    #
    #     # Где будет сохранен результат выравнивания
    #     image_file_name_ffhqalign  = os.path.join(save_dir_ffhqalign, image_file)
    #     print("image_file_name_ffhqalign =", image_file_name_ffhqalign)
    #
    #     # Если результат обработки уже есть в папке
    #     if os.path.exists(image_file_name_ffhqalign):
    #         continue
    #     print("cont")
    #
    #     print("file =", file)
    #     face_img = cv2.imread(file)
    #
    #     # Если изображение слишком маленькое
    #     if face_img.shape[0]<250 or face_img.shape[1]<250:
    #         continue
    #
    #     # Выравнивание изображения
    #     ret = app.get(face_img,crop_size,mode=mode)
    #
    #
    #
    #     # Сохранение
    #     if len(ret)!=0 :
    #         cv2.imwrite(image_file_name_ffhqalign, ret[0])
    #     else:
    #         continue

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
    print("do DIRS", input_dir)
    dirs = sorted(os.listdir(input_dir))
    print("posle Dirs", dirs)

    # Список папок с изображениями
    # Нужно чтобы в место папок были видео
    handle_dir_list = dirs


    # Перебор по папкам
    # Нужен перебор по видео
    for handle_dir_list_tmp  in tqdm(handle_dir_list):
        print("Выравнивание видео ->", handle_dir_list_tmp)
        align_image_dir(handle_dir_list_tmp)

