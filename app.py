from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from mangum import Mangum

import base64
import numpy as np
from PIL import Image
import io
import os
import logging
from typing import Optional

from supabase import create_client, Client
from dotenv import load_dotenv

# =====================
# ENV + LOGGING
# =====================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =====================
# OPTIONAL BARCODE DEP
# =====================
try:
    from pyzbar.pyzbar import decode as zbar_decode
except Exception as exc:
    zbar_decode = None
    logger.warning("pyzbar not available: %s", exc)

# =====================
# SUPABASE
# =====================
SUPABASE_TABLE = "Database"  # CHANGE HERE if needed

_supabase_client: Optional[Client] = None

def get_supabase_client() -> Client:
    global _supabase_client

    if _supabase_client:
        return _supabase_client

    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")

    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set")

    _supabase_client = create_client(url, key)
    return _supabase_client

# =====================
# FASTAPI APP
# =====================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================
# MODELS
# =====================
class ImageRequest(BaseModel):
    image: str
    format: str = "image/jpeg"

class LanguageUpdateRequest(BaseModel):
    uid: str
    language: str

# =====================
# HELPERS
# =====================
def read_barcode(image: np.ndarray) -> Optional[str]:
    if not zbar_decode:
        raise RuntimeError("Barcode decoding not supported on this platform")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    pil_image = Image.fromarray(image)
    decoded = zbar_decode(pil_image)

    if decoded:
        return decoded[0].data.decode("utf-8")

    return None

def get_student_by_uid(uid: str):
    supabase = get_supabase_client()
    return (
        supabase.table(SUPABASE_TABLE)
        .select("*")
        .eq("UID", uid)
        .execute()
        .data
    )

def update_student_language(uid: str, language: str):
    supabase = get_supabase_client()
    return (
        supabase.table(SUPABASE_TABLE)
        .update({"Language": language})
        .eq("UID", uid)
        .execute()
        .data
    )

# =====================
# ROUTES
# =====================
@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/read-barcode")
async def read_barcode_endpoint(request: ImageRequest):
    try:
        image_bytes = base64.b64decode(request.image)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        barcode = read_barcode(image_np)

        if not barcode:
            return {"success": False, "message": "No barcode detected"}

        student = get_student_by_uid(barcode)
        first_name = None

        if student:
            name = student[0].get("Name", "Student")
            first_name = name.split(" ")[0]

        return {
            "success": True,
            "barcode": barcode,
            "firstName": first_name,
            "message": "Barcode detected"
        }

    except Exception as e:
        logger.error("Barcode error", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/student-profile/{uid}")
async def get_student_profile(uid: str):
    try:
        data = get_student_by_uid(uid)

        if not data:
            return {"success": False, "message": "Student not found"}

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
        }

    except Exception as e:
        logger.error("Profile error", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/update-language")
async def update_language(request: LanguageUpdateRequest):
    try:
        data = update_student_language(request.uid, request.language)
        return {"success": True, "data": data}
    except Exception as e:
        logger.error("Language update error", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/update-profile")
async def update_profile(request: Request):
    try:
        body = await request.json()
        uid = body.get("uid")

        if not uid:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "UID is required"}
            )

        update_data = {
            k: v for k, v in {
                "Age": body.get("age"),
                "Allergy": body.get("allergy"),
                "Number": body.get("number"),
            }.items() if v is not None
        }

        if not update_data:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No fields to update"}
            )

        supabase = get_supabase_client()
        supabase.table(SUPABASE_TABLE).update(update_data).eq("UID", uid).execute()

        return {"success": True, "message": "Profile updated"}

    except Exception as e:
        logger.error("Update profile error", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# =====================
# AWS LAMBDA SUPPORT
# =====================
handler = Mangum(app)

# =====================
# LOCAL RUN
# =====================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
