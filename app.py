# ------------------ START: Cure Genie human-image -> gpt-oss-20 pipeline ------------------
# Merge/import lines (add near your other imports)
import os
import io
import json
import base64
import re
import asyncio
from datetime import datetime
from typing import Optional, Dict, Any

from PIL import Image
import numpy as np
from pyzbar.pyzbar import decode

# tensorflow is heavy; import only when model is loaded to avoid slow startup if not used
import httpx
from fastapi import APIRouter, Body, HTTPException, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# CONFIG (env)
# -------------------------
MODEL_API_URL = os.getenv("MODEL_API_URL", "https://openrouter.ai/api/v1")  # set real endpoint
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-oss-20")
# DB_PATH removed as we are using Supabase
MODEL_TIMEOUT = int(os.getenv("MODEL_TIMEOUT", "30"))
MODEL_RETRIES = int(os.getenv("MODEL_RETRIES", "3"))
RETRY_BACKOFF = float(os.getenv("MODEL_RETRY_BACKOFF", "1.5"))
CNN_MODEL_PATH = os.getenv("CNN_MODEL_PATH", "human_main.h5")  # relative path for Docker
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# -------------------------
# Supabase Client Init
# -------------------------
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY) if SUPABASE_URL and SUPABASE_KEY else None

async def save_analysis_full(uid: str, mode: str, language: str, keyword: str, description: str, age: Optional[int], allergies: Optional[str], image_present: bool, result: dict):
    if not supabase:
        print("Supabase not configured, skipping save.")
        return None

    def _save():
        data = {
            "created_at": datetime.utcnow().isoformat(),
            "uid": uid,
            "mode": mode,
            "language": language,
            "keyword": keyword,
            "description": description,
            "age": age,
            "allergies": allergies or "",
            "image_present": bool(image_present),
            "result_json": result  # Supabase handles JSON/JSONB automatically
        }
        try:
            response = supabase.table("analyses").insert(data).execute()
            # response.data is a list of inserted records
            if response.data and len(response.data) > 0:
                return response.data[0].get("id")
            return None
        except Exception as e:
            print(f"Error saving to Supabase: {e}")
            return None

    return await asyncio.to_thread(_save)

async def fetch_analysis_by_id(an_id: int):
    if not supabase:
        return None

    def _fetch():
        try:
            response = supabase.table("analyses").select("*").eq("id", an_id).limit(1).execute()
            if response.data and len(response.data) > 0:
                r = response.data[0]
                return {
                    "id": r.get("id"),
                    "created_at": r.get("created_at"),
                    "uid": r.get("uid"),
                    "mode": r.get("mode"),
                    "language": r.get("language"),
                    "keyword": r.get("keyword"),
                    "description": r.get("description"),
                    "age": r.get("age"),
                    "allergies": r.get("allergies"),
                    "image_present": r.get("image_present"),
                    "result": r.get("result_json")
                }
            return None
        except Exception as e:
            print(f"Error fetching from Supabase: {e}")
            return None

    return await asyncio.to_thread(_fetch)

# -------------------------
# CNN: lazy-load model and predict in thread
# -------------------------
_CNN_MODEL = None
_CNN_LABELS = None  # list of class names in same order as model outputs; replace with your labels

def _load_cnn_sync():
    global _CNN_MODEL, _CNN_LABELS
    if _CNN_MODEL is not None:
        return _CNN_MODEL
    try:
        # local import to avoid import cost on cold start if tensorflow isn't installed elsewhere
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        _CNN_MODEL = load_model(CNN_MODEL_PATH)
        # If you stored labels in a sidecar JSON, load it here; otherwise set manually
        labels_path = os.path.splitext(CNN_MODEL_PATH)[0] + "_labels.json"
        if os.path.exists(labels_path):
            with open(labels_path, "r", encoding="utf-8") as fh:
                _CNN_LABELS = json.load(fh)
        else:
            # TODO: replace with your real class names in correct order
            _CNN_LABELS = ["condition_a", "condition_b", "condition_c"]
        return _CNN_MODEL
    except Exception as e:
        raise RuntimeError(f"Failed to load CNN model: {e}")

def _preprocess_pil_for_model(pil_image: Image.Image, target_size=(224,224)):
    # convert to RGB, resize, scale to [0,1] float32
    img = pil_image.convert("RGB")
    img = img.resize(target_size, Image.BILINEAR)
    arr = np.asarray(img).astype("float32") / 255.0
    # expand dims to [1, H, W, C]
    return np.expand_dims(arr, axis=0)

async def predict_keyword_from_image(image_b64: str) -> str:
    """
    Decodes base64, preprocesses image, runs the CNN in a thread, returns best keyword label.
    """
    # decode
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        img_bytes = base64.b64decode(image_b64)
        pil = Image.open(io.BytesIO(img_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image/base64: {e}")

    def _predict_sync(pil_img):
        model = _load_cnn_sync()
        x = _preprocess_pil_for_model(pil_img, target_size=(224,224))  # adjust target if your model needs different
        preds = model.predict(x)  # shape (1, n_classes) or whatever your model yields
        # handle different output shapes robustly
        if hasattr(preds, "shape"):
            if preds.shape[-1] == 1:
                # single-output model (e.g. regression) -> treat threshold
                val = float(preds.ravel()[0])
                return str(val)
            else:
                idx = int(np.argmax(preds, axis=-1).ravel()[0])
                label = _CNN_LABELS[idx] if _CNN_LABELS and idx < len(_CNN_LABELS) else f"class_{idx}"
                return label
        else:
            # fallback to string
            return str(preds)
    # run heavy work in thread
    return await asyncio.to_thread(_predict_sync, pil)

# -------------------------
# Supabase helper (reuse global supabase client)
# -------------------------
async def get_user_profile(uid: str) -> Dict[str, Any]:
    """
    Try to retrieve age and allergies from Supabase (sync calls via asyncio.to_thread to avoid blocking).
    """
    if not supabase:
        return {"age": None, "allergies": ""}

    def _sync_query():
        try:
            resp = supabase.table("students").select("age,allergies").eq("uid", uid).limit(1).execute()
            data = resp.data if hasattr(resp, "data") else resp
            if data and isinstance(data, list) and len(data) > 0:
                rec = data[0]
                return {"age": rec.get("age"), "allergies": rec.get("allergies","")}
        except Exception:
            pass
        return {"age": None, "allergies": ""}

    return await asyncio.to_thread(_sync_query)

# -------------------------
# Prompt builder & model caller (reuse your call_gpt_oss style function)
# -------------------------
def build_combined_prompt(keyword: str, description: str, age: Optional[int], allergies: Optional[str]) -> str:
    # keep prompt explicit and instruction-focused
    allergies_txt = allergies or "No known allergies reported."
    age_txt = f"Age: {age} years." if age is not None else "Age: unknown."
    prompt = (
        "You are Cure Genie — a clinical-first-aid assistant. Use the information below to provide:\n"
        "1) Short diagnosis summary (1-2 sentences).\n"
        "2) Medication / first-aid steps (clear, actionable).\n"
        "3) Prevention tips.\n"
        "4) Items to buy from the campus vending machine (exact product names/brands if possible), and how to use them.\n\n"
        f"Context:\n- CNN keyword (possible condition): {keyword}\n- User description: {description}\n- {age_txt}\n- Allergies: {allergies_txt}\n\n"
        "Constraints: If the user is a minor or age unknown, avoid prescribing strong or prescription-only meds; prefer OTC and first-aid items. "
        "When suggesting items, keep them available at a typical campus vending machine (bandages, antiseptic wipes, topical creams, oral antacids, antihistamine tablets, hydrocortisone cream 1% etc.). "
        "Be concise and list items as bullet points with short usage instructions. Output must be in plain text sections labeled: Diagnosis:, Medication/First-aid:, Prevention:, VendingItems:."
    )
    return prompt

async def call_gpt_oss(prompt: str) -> str:
    if not MODEL_API_KEY:
        raise RuntimeError("MODEL_API_KEY not configured in env")

    # User requested OpenRouter URL
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {MODEL_API_KEY}",
        "Content-Type": "application/json",
        # Optional: Add site URL and title for OpenRouter rankings
        # "HTTP-Referer": "https://curegenie.app", 
        # "X-Title": "CureGenie",
    }
    
    # User requested model and reasoning parameter
    payload = {
        "model": "openai/gpt-oss-20b:free",
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "reasoning": {"enabled": True}
    }

    async with httpx.AsyncClient(timeout=MODEL_TIMEOUT) as client:
        last_exc = None
        for attempt in range(1, MODEL_RETRIES + 1):
            try:
                r = await client.post(url, json=payload, headers=headers)
                r.raise_for_status()
                j = r.json()
                
                # Extract the assistant message content
                if "choices" in j and j["choices"]:
                    message = j["choices"][0].get("message", {})
                    content = message.get("content", "")
                    # Note: reasoning_details are available in message.get("reasoning_details") if needed later
                    return content
                
                # Fallback if structure is unexpected
                return json.dumps(j)
                
            except Exception as e:
                last_exc = e
                backoff = RETRY_BACKOFF ** attempt
                await asyncio.sleep(min(5, backoff))
        raise RuntimeError(f"Model API failed after {MODEL_RETRIES} attempts: {last_exc}")

# -------------------------
# Use your exact parse function here — REPLACE placeholder with your real parse logic.
# -------------------------
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace('\\n', '\n')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def parse_analysis_result(result_text: str) -> dict:
    """
    Replace this stub with your original parse_analysis_result function.
    The function should return a dict with keys like:
    { "diagnosis": "...", "precautions": "...", "recommendations":[...], "treatment":[...], "confidence": XX }
    """
    cleaned = clean_text(result_text)
    # naive parse: split by sections if model followed labels
    out = {"diagnosis": "", "medication": "", "prevention": "", "vending_items": [], "raw": cleaned}
    # try to extract labeled sections
    for label in ["Diagnosis:", "Medication/First-aid:", "Prevention:", "VendingItems:"]:
        if label in cleaned:
            part = cleaned.split(label,1)[1]
            # take until next label or end
            next_labels = ["Diagnosis:", "Medication/First-aid:", "Prevention:", "VendingItems:"]
            next_part = part
            for nl in next_labels:
                if nl != label and nl in part:
                    next_part = part.split(nl,1)[0]
                    break
            out_label = label.replace(":", "").lower().replace("/", "_")
            if out_label == "vendingitems":
                # parse lines into list
                lines = [l.strip("-* \t") for l in next_part.splitlines() if l.strip()]
                out["vending_items"] = lines
            else:
                out[out_label] = next_part.strip()
    # fallback minimal fields
    if not out["diagnosis"]:
        out["diagnosis"] = cleaned.split(".")[0] + "." if cleaned else ""
    return out

# -------------------------
# Router & Endpoint
# -------------------------
router_ai = APIRouter()

@router_ai.post("/api/analyze")  # this is the main integrated endpoint
async def api_analyze_combined(body: dict = Body(...)):
    """
    Required fields in body:
      - image: base64 image (data URI or plain base64)
      - description: user text describing symptoms (required)
      - uid: user id to fetch age/allergies from Supabase (optional but recommended)
      - language: 'english' or 'hindi' (optional)
    """
    image_b64 = body.get("image")
    description = (body.get("description") or "").strip()
    uid = body.get("uid", "")
    language = body.get("language", "english")
    mode = "human"

    if not image_b64 or not description:
        raise HTTPException(status_code=400, detail="Both image and description are required")

    # 1) CNN -> keyword
    try:
        keyword = await predict_keyword_from_image(image_b64)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CNN prediction failed: {e}")

    # 2) fetch user profile
    try:
        profile = await get_user_profile(uid) if uid else {"age": None, "allergies": ""}
        age = profile.get("age")
        allergies = profile.get("allergies", "")
    except Exception:
        age = None
        allergies = ""

    # 3) build prompt and call gpt-oss-20
    prompt = build_combined_prompt(keyword, description, age, allergies)
    try:
        raw_model_out = await call_gpt_oss(prompt)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model call failed: {e}")

    # 4) parse using your parse function
    parsed = parse_analysis_result(raw_model_out)

    # 5) persist (best-effort)
    try:
        record_id = await save_analysis_full(uid=uid, mode=mode, language=language, keyword=keyword, description=description, age=age, allergies=allergies, image_present=True, result=parsed)
    except Exception:
        record_id = None

    return {"success": True, "keyword": keyword, "data": parsed, "raw": raw_model_out, "record_id": record_id}

@app.post("/api/read-barcode")
async def api_read_barcode(body: dict = Body(...)):
    """
    Decodes a barcode from a base64 image and looks up the student.
    """
    image_b64 = body.get("image")
    if not image_b64:
        raise HTTPException(status_code=400, detail="No image provided")

    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]

    try:
        # Decode base64 image
        img_bytes = base64.b64decode(image_b64)
        img = Image.open(io.BytesIO(img_bytes))
        
        # Decode barcode using pyzbar
        # This runs synchronously, so we might want to wrap in thread if heavy, 
        # but for single barcode it's usually fast enough.
        decoded_objects = decode(img)
        
        if not decoded_objects:
            return {"success": False, "message": "No barcode detected"}
            
        # Take the first detected barcode
        obj = decoded_objects[0]
        barcode_data = obj.data.decode("utf-8")
        barcode_type = obj.type
        
        print(f"Detected barcode: {barcode_data} ({barcode_type})")
        
        # Lookup student in Supabase
        first_name = None
        if supabase:
            try:
                # Assuming the barcode data corresponds to the 'uid' column
                resp = supabase.table("students").select("first_name").eq("uid", barcode_data).execute()
                if resp.data and len(resp.data) > 0:
                    first_name = resp.data[0].get("first_name")
            except Exception as e:
                print(f"Supabase lookup failed: {e}")
                
        return {
            "success": True,
            "barcode": barcode_data,
            "type": barcode_type,
            "firstName": first_name,
            "message": "Barcode verified"
        }

    except Exception as e:
        print(f"Barcode processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process barcode: {str(e)}")


@app.get("/api/student-profile/{uid}")
async def get_student_profile(uid: str):
    """Get student profile by UID"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        response = supabase.table("students").select("*").eq("uid", uid).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        student = response.data[0]
        return {
            "success": True,
            "uid": student.get("uid"),
            "firstName": student.get("first_name"),
            "fullName": student.get("full_name"),
            "number": student.get("number"),
            "language": student.get("language", "english"),
            "age": student.get("age"),
            "allergy": student.get("allergy"),
            "message": "Profile retrieved successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Profile fetch error: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.post("/api/update-language")
async def update_language(body: dict = Body(...)):
    """Update student language preference"""
    uid = body.get("uid")
    language = body.get("language")
    
    if not uid or not language:
        raise HTTPException(status_code=400, detail="Missing uid or language")
    
    if language not in ["english", "hindi"]:
        raise HTTPException(status_code=400, detail="Invalid language. Must be 'english' or 'hindi'")
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    try:
        response = supabase.table("students").update({
            "language": language
        }).eq("uid", uid).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return {
            "success": True,
            "message": "Language updated successfully",
            "language": language
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Language update error: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.post("/api/update-profile")
async def update_profile(body: dict = Body(...)):
    """Update student profile (age, allergy, number)"""
    uid = body.get("uid")
    
    if not uid:
        raise HTTPException(status_code=400, detail="Missing uid")
    
    if not supabase:
        raise HTTPException(status_code=500, detail="Database not configured")
    
    # Build update dict with only provided fields
    update_data = {}
    if "age" in body and body["age"] is not None:
        update_data["age"] = int(body["age"])
    if "allergy" in body:
        update_data["allergy"] = body["allergy"]
    if "number" in body:
        update_data["number"] = body["number"]
    
    if not update_data:
        raise HTTPException(status_code=400, detail="No fields to update")
    
    try:
        response = supabase.table("students").update(update_data).eq("uid", uid).execute()
        
        if not response.data or len(response.data) == 0:
            raise HTTPException(status_code=404, detail="Student not found")
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "updated_fields": list(update_data.keys())
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Profile update error: {e}")
        raise HTTPException(status_code=500, detail=f"Update failed: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "CureGenie Backend V2",
        "timestamp": datetime.utcnow().isoformat()
    }


# Mount router into your app: add this after app creation in your file
# app.include_router(router_ai)
# ------------------ END: Cure Genie pipeline ------------------
