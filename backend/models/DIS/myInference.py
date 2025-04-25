import os
import time
import numpy as np
from skimage import io
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

from models import ISNetDIS  
import cv2
import argparse

from mask_to_image import *
import time
import psutil

def print_memory_usage():
    process = psutil.Process(os.getpid())
    ram_usage = process.memory_info().rss / 1024**3 
    print(f"RAM usage: {ram_usage:.2f} GB")
    
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / 1024**3
        vram_max = torch.cuda.max_memory_allocated() / 1024**3
        print(f"VRAM usage: {vram_alloc:.2f} GB (Peak: {vram_max:.2f} GB)")

def get_arguments():
    parser = argparse.ArgumentParser(description='Image Segmentation Inference')
    parser.add_argument('--dataset_path', type=str, default="",
                        help='dataset path')
    parser.add_argument('--model_path', type=str, default="",
                        help='model path')
    parser.add_argument('--result_path', type=str, default="",
                        help='mask path')
    parser.add_argument('--img_result_path', type=str, default="",
                        help='original image path')
    parser.add_argument('--input_size', type=int, nargs=2, default=[1024, 1024],
                        help='input size')
    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.perf_counter()
    initial_ram = psutil.virtual_memory().used / 1024**3


    args = get_arguments()
    dataset_path = args.dataset_path
    model_path = args.model_path
    result_path = args.result_path
    img_result_path = args.img_result_path  
    input_size = args.input_size

    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    
    net.eval()  

    im_list = glob(dataset_path+"/*.jpg")+glob(dataset_path+"/*.JPG")+glob(dataset_path+"/*.jpeg")+glob(dataset_path+"/*.JPEG")+glob(dataset_path+"/*.png")+glob(dataset_path+"/*.PNG")+glob(dataset_path+"/*.bmp")+glob(dataset_path+"/*.BMP")+glob(dataset_path+"/*.tiff")+glob(dataset_path+"/*.TIFF")

    os.makedirs(result_path, exist_ok=True)
    os.makedirs(img_result_path,exist_ok=True)

    with torch.no_grad():  
        try:
            for i, im_path in tqdm(enumerate(im_list), total=len(im_list)):
                im = cv2.imread(im_path)  
                
                if im is None:
                    print(f"can't read image: {im_path}")
                    continue

                if len(im.shape) < 3:  
                    im = im[:, :, np.newaxis]

                im_shp = im.shape[0:2]  
                
                im_tensor = torch.tensor(im.copy(), dtype=torch.float32).permute(2, 0, 1)
                im_tensor = F.upsample(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
                image = torch.divide(im_tensor, 255.0)
                image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

                if torch.cuda.is_available():
                    image = image.cuda()
                
                result = net(image)
                
                result = torch.squeeze(F.upsample(result[0][0], im_shp, mode='bilinear'), 0)
                ma = torch.max(result)
                mi = torch.min(result)
                result = (result - mi) / (ma - mi)
                im_name = os.path.basename(im_path).split('.')[0]
                io.imsave(os.path.join(result_path, im_name + ".png"),
                          (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8))
            restore_original_colors(result_path,dataset_path,img_result_path)
            total_time = time.perf_counter() - start_time
            final_ram = psutil.virtual_memory().used / 1024**3
    
            print("\n" + "="*50)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"RAM consumption: {final_ram - initial_ram:.2f} GB")
            print_memory_usage()
        except Exception as e:
            print(f"error: {e}")
