from pydantic import BaseModel
from typing import List



class REMOVE_Response(BaseModel):
    data : str
    image_path: str
    status_code: int