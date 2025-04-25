import numpy as np
from PIL import Image
import os
import cv2
import torch
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from models.DIS.models import ISNetDIS
from models.DIS.mask_to_image import restore_original_colors
import uuid



def removing_bg(image_path, model_path="/home/minh/Desktop/fastapi_removebg/backend/models/DIS/isnet-general-use.pth", input_size=(512, 512)) -> str:
    net = ISNetDIS()

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net = net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location="cpu"))
    net.eval()

    im = Image.open(image_path).convert("RGB")
    if im is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    im_np = np.array(im)

    if len(im_np.shape) < 3:
        raise ValueError(f"Image does not have 3 channels: {image_path}")

    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2, 0, 1) 
    im_tensor = F.interpolate(im_tensor.unsqueeze(0), size=input_size, mode="bilinear").type(torch.float32) / 255.0

    image = normalize(im_tensor, [0.5, 0.5, 0.5], [1.0, 1.0, 1.0])

    if torch.cuda.is_available():
        image = image.cuda()

    with torch.no_grad():
        result = net(image)
        result = torch.squeeze(F.interpolate(result[0][0], im_np.shape[:2], mode='bilinear'), 0)
        result = (result - result.min()) / (result.max() - result.min())

    output_name = f"{uuid.uuid4().hex[:8]}.png"
    result_img = (result * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    mask_output_path = os.path.join("masks", output_name)
    os.makedirs("masks", exist_ok=True)
    cv2.imwrite(mask_output_path, result_img)

    return mask_output_path  