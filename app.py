```python
from fastapi import FastAPI, HTTPException
from mangum import Mangum

from pydantic import BaseModel
import base64
import numpy as np
from PIL import Image
import io
import os
from supabase import create_client, Client
from pyzbar.pyzbar import decode
from dotenv import load_dotenv

load_dotenv()

url: str = os.getenv("SUPABASE_URL")
key: str = os.getenv("SUPABASE_KEY")

if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(url, key)

app = FastAPI()

class ImageRequest(BaseModel):
    image: str
    format: str = "image/jpeg"

class LanguageUpdateRequest(BaseModel):
    uid: str
    language: str

def read_barcode(image):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    
    try:
        pil_image = Image.fromarray(image)
    except:
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image, 'RGB')
        else:
            pil_image = Image.fromarray(image, 'L')
    
    try:
        decoded_objects = decode(pil_image)
        if decoded_objects:
            barcodes = [obj.data.decode("utf-8") for obj in decoded_objects]
            if barcodes:
                return barcodes[0]
    except:
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            pil_image = Image.fromarray(gray, 'L')
            try:
                decoded_objects = decode(pil_image)
                if decoded_objects:
                    barcodes = [obj.data.decode("utf-8") for obj in decoded_objects]
                    if barcodes:
                        return barcodes[0]
            except:
                pass
    
    return None

def get_student_by_uid(uid: str):
    """
    Fetch student data by UID from Supabase.
    """
    response = (
        supabase.table("Database")
        .select("*")
        .eq("UID", uid)
        .execute()
    )
    return response.data

def update_student_language(uid: str, language: str):
    """
    Update student language preference in Supabase.
    """
    response = (
        supabase.table("Database")
        .update({"Language": language})
        .eq("UID", uid)
        .execute()
    )
    return response.data

@app.post("/api/read-barcode")
async def read_barcode_endpoint(request: ImageRequest):
    try:
        image_data = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_data))
        image_np = np.array(image)
        
        barcode = read_barcode(image_np)
        
        if barcode:
            # Fetch student details using the barcode as UID
            student_data = get_student_by_uid(barcode)
            first_name = None
            if student_data and len(student_data) > 0:
                # Assuming Name column exists
                name = student_data[0].get("Name", "Student")
                first_name = name.split(" ")[0] if name else "Student"
            
            return {
                "success": True,
                "barcode": barcode,
                "firstName": first_name,
                "message": "Barcode detected"
            }
        else:
            return {
                "success": False,
                "barcode": None,
                "message": "No barcode detected"
            }
    except Exception as e:
        print(f"Error processing barcode: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/student-profile/{uid}")
async def get_student_profile(uid: str):
    try:
        data = get_student_by_uid(uid)
        if data and len(data) > 0:
            student = data[0]
            return {
                "success": True,
                "uid": student.get("UID"),
                "firstName": student.get("Name", "").split(" ")[0],
                "fullName": student.get("Name"),
                "number": student.get("Number"),
                "language": student.get("Language", "English"),
                "message": "Student found"
            }
        else:
            return {
                "success": False,
                "message": "Student not found"
            }
    except Exception as e:
        print(f"Error fetching profile: {e}")
        return {"success": False, "error": str(e)}

@app.post("/api/update-language")
async def update_language(request: LanguageUpdateRequest):
    try:
        data = update_student_language(request.uid, request.language)
        return {"success": True, "data": data}
    except Exception as e:
        print(f"Error updating language: {e}")
        return {"success": False, "error": str(e)}

handler = Mangum(app)
```
