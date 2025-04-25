import requests
import io
from PIL import Image

BACKEND_URL = "http://127.0.0.1:8000"

def dis_api(image_path: str):
    try:
        image = Image.open(image_path)
        img_name = image_path.split("/")[-1]
        
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        url = f"{BACKEND_URL}/rembg/DIS"
        
        files = {'file_upload': (img_name, img_byte_arr, 'image/jpeg')}
        headers = {'accept': 'application/json'}

        response = requests.post(url, headers=headers, files=files)

        if response.status_code == 200:
            image_relative_path = response.json()['data']  
            image_url = f"{BACKEND_URL}/{image_relative_path}"  
            
            response_image = requests.get(image_url)
            if response_image.status_code == 200:
                image_with_result = Image.open(io.BytesIO(response_image.content))
                return image_with_result
            else:
                return f"Error downloading result image: {response_image.status_code}"
        else:
            return f"Error: API request failed ({response.status_code})"

    except Exception as e:
        return f"Error: {str(e)}"
