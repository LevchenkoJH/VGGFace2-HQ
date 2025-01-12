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

def align_image_dir(dir_name_tmp):
    print("------------------------------------------------------align_image_dir------------------------------------------------------")
    ori_path_tmp = os.path.join(input_dir, dir_name_tmp)
    print("ori_path_tmp ->", ori_path_tmp)
    image_filenames = glob.glob(os.path.join(ori_path_tmp,'*'))
    print("image_filenames ->", image_filenames)
    save_dir_ffhqalign = os.path.join(output_dir_ffhqalign,dir_name_tmp)
    if not os.path.exists(save_dir_ffhqalign):
        os.makedirs(save_dir_ffhqalign)


    for file in image_filenames:
        image_file = os.path.basename(file)

        image_file_name_ffhqalign  = os.path.join(save_dir_ffhqalign, image_file)
        if os.path.exists(image_file_name_ffhqalign):
            continue

        face_img = cv2.imread(file)
        if face_img.shape[0]<250 or face_img.shape[1]<250:
            continue
        ret = app.get(face_img,crop_size,mode=mode)
        if len(ret)!=0 :
            cv2.imwrite(image_file_name_ffhqalign, ret[0])
        else:
            continue

# Для выравнивания датасета
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir',type=str,default = '../Data/VGGface2/train')
    parser.add_argument('--output_dir_ffhqalign',type=str,default = '../Data/VGGface2_FFHQalign')
    # parser.add_argument('--crop_size',type=int,default = 256)
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--mode',type=str,default = 'ffhq',choices=['ffhq','newarc','both'])

    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir_ffhqalign = args.output_dir_ffhqalign
    crop_size = args.crop_size
    mode      = args.mode

    app = Face_detect_crop(name='antelope', root='../insightface_func/models')

    app.prepare(ctx_id= 0, det_thresh=0.6, det_size=(320,320))

    #
    print("do DIRS", input_dir)
    dirs = sorted(os.listdir(input_dir))
    print("posle Dirs")
    handle_dir_list = dirs
    for handle_dir_list_tmp  in tqdm(handle_dir_list):
        print("handle_dir_list_tmp ->", handle_dir_list_tmp)
        align_image_dir(handle_dir_list_tmp)

