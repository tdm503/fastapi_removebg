from fastapi import APIRouter, File, UploadFile, status, HTTPException
from controllers import removing_bg
from schemas import REMOVE_Response
from PIL import Image

router = APIRouter()

@router.post("/rembg/DIS")
async def predict_ocr(file_upload: UploadFile = File(...)):
    try:
        image_bytes = await file_upload.read()  
        image_path = "/tmp/temp_image.jpg"  
        
        with open(image_path, "wb") as f:
            f.write(image_bytes)  

        removebg_result_path = removing_bg(image_path)
        
        return REMOVE_Response(
            data=removebg_result_path,
            status_code=status.HTTP_200_OK,
            image_path=removebg_result_path
        )
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Error processing image: {str(e)}")

@router.post("rembg/traceb7")
async def predict(file_upload:UploadFile = File(...)):
    res = "a"
    return res


@router.post("birefnet")
async def predict(file_upload:UploadFile = File(...)):
    res = "a"
    return res