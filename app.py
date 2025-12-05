from fastapi import FastAPI, HTTPException
from mangum import Mangum

from pydantic import BaseModel
import base64
import numpy as np
from PIL import Image
import io
import os
import logging
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    from pyzbar.pyzbar import decode as zbar_decode
    _pyzbar_error = None
except Exception as exc:  # pragma: no cover - platform specific
    zbar_decode = None
    _pyzbar_error = exc


def get_supabase_client() -> Client:
    """
    Lazily instantiate the Supabase client so startup does not crash if env vars
    are missing. This keeps the /health endpoint alive and surfaces a clearer
    error once the database is actually needed.
    """
    url: str = os.getenv("SUPABASE_URL")
    key: str = os.getenv("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set in environment variables"
        )

    global _supabase_client
    if "_supabase_client" not in globals() or _supabase_client is None:
        _supabase_client = create_client(url, key)
    return _supabase_client


supabase: Optional[Client] = None

app = FastAPI()

class ImageRequest(BaseModel):
    image: str
    format: str = "image/jpeg"

class LanguageUpdateRequest(BaseModel):
    uid: str
    language: str

def read_barcode(image):
    if not zbar_decode:
        raise RuntimeError(
            "pyzbar/zbar is not available on the current platform, unable to decode barcodes"
        )

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
    except Exception:
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image, 'RGB')
        else:
            pil_image = Image.fromarray(image, 'L')
    
    try:
        decoded_objects = zbar_decode(pil_image)
        if decoded_objects:
            barcodes = [obj.data.decode("utf-8") for obj in decoded_objects]
            if barcodes:
                return barcodes[0]
    except Exception:
        if len(image.shape) == 3:
            gray = np.dot(image[...,:3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            pil_image = Image.fromarray(gray, 'L')
            try:
                decoded_objects = zbar_decode(pil_image)
                if decoded_objects:
                    barcodes = [obj.data.decode("utf-8") for obj in decoded_objects]
                    if barcodes:
                        return barcodes[0]
            except Exception:
                pass
    
    return None

def get_student_by_uid(uid: str):
    """
    Fetch student data by UID from Supabase.
    """
    supabase = get_supabase_client()
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
    supabase = get_supabase_client()
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
        logger.error(f"Error processing barcode: {e}", exc_info=True)
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
                "age": student.get("Age"),
                "allergy": student.get("Allergy"),
                "message": "Student found"
            }
        else:
            return {
                "success": False,
                "message": "Student not found"
            }
    except Exception as e:
        logger.error(f"Error fetching profile: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.post("/api/update-language")
async def update_language(request: LanguageUpdateRequest):
    try:
        data = update_student_language(request.uid, request.language)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error(f"Error updating language: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Service is running"}

@app.post("/update-profile")
async def update_profile(request: Request):
    """Update student profile information"""
    try:
        data = await request.json()
        uid = data.get("uid")
        age = data.get("age")
        allergy = data.get("allergy")
        number = data.get("number")
        
        if not uid:
            return JSONResponse(
                content={"success": False, "message": "Student ID is required"},
                status_code=400
            )
        
        # Build update data
        update_data = {}
        if age is not None:
            update_data["age"] = int(age)
        if allergy is not None:
            update_data["allergy"] = allergy if allergy.strip() else None
        if number is not None:
            update_data["number"] = number if number.strip() else None
        
        if not update_data:
            return JSONResponse(
                content={"success": False, "message": "No fields to update"},
                status_code=400
            )
        
        # Update in database
        response = supabase.table("students").update(update_data).eq("uid", uid).execute()
        
        if response.data:
            logging.info(f"Profile updated for UID: {uid}")
            return JSONResponse(content={"success": True, "message": "Profile updated successfully"})
        else:
            logging.error(f"Failed to update profile for UID: {uid}")
            return JSONResponse(
                content={"success": False, "message": "Failed to update profile"},
                status_code=500
            )
            
    except Exception as e:
        logging.error(f"Error updating profile: {str(e)}")
        return JSONResponse(
            content={"success": False, "message": "Internal server error"},
            status_code=500
        )

handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
