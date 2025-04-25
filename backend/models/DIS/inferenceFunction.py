import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models import ISNetDIS
from mask_to_image import restore_original_colors
import uuid

def run_dis_model_on_image(image_path: str, model_path: str, input_size=(1024, 1024)) -> str:
    net = ISNetDIS()
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    im = cv2.imread(image_path)
    if im is None:
        raise ValueError(f"Không thể đọc ảnh: {image_path}")
    
    if len(im.shape) < 3:
        im = im[:, :, np.newaxis]

    original_shape = im.shape[0:2]

    im_tensor = torch.tensor(im.copy(), dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(torch.unsqueeze(im_tensor, 0), input_size, mode="bilinear").type(torch.uint8)
    image = torch.divide(im_tensor, 255.0)
    image = normalize(image, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        result = net(image)
        result = torch.squeeze(F.interpolate(result[0][0], original_shape, mode='bilinear'), 0)
        result = (result - result.min()) / (result.max() - result.min())

    base_name = os.path.basename(image_path).split('.')[0]
    output_name = f"{base_name}_{uuid.uuid4().hex[:8]}.png"
    mask_output_path = os.path.join("masks", output_name)
    os.makedirs("masks", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    result_img = (result * 255).permute(1, 2, 0).cpu().data.numpy().astype(np.uint8)
    cv2.imwrite(mask_output_path, result_img)

    restore_original_colors("masks", os.path.dirname(image_path), "images")

    final_image_path = os.path.join("images", output_name)
    return final_image_path
