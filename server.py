# app_flask.py
from flask import Flask, request, jsonify
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

app = Flask(__name__)


def read_barcode(image):
    """
    Accepts a numpy array or PIL-compatible array and returns first barcode string or None.
    """
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
    except Exception:
        # second attempt: convert to grayscale and retry
        try:
            if len(image.shape) == 3:
                gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
                pil_image = Image.fromarray(gray, 'L')
                decoded_objects = decode(pil_image)
                if decoded_objects:
                    barcodes = [obj.data.decode("utf-8") for obj in decoded_objects]
                    if barcodes:
                        return barcodes[0]
        except Exception:
            pass

    return None


def get_student_by_uid(uid: str):
    """Fetch student data by UID from Supabase."""
    response = (
        supabase.table("Database")
        .select("*")
        .eq("UID", uid)
        .execute()
    )
    return response.data


def update_student_language(uid: str, language: str):
    """Update student language preference in Supabase."""
    response = (
        supabase.table("Database")
        .update({"Language": language})
        .eq("UID", uid)
        .execute()
    )
    return response.data


@app.route("/api/read-barcode", methods=["POST"])
def read_barcode_endpoint():
    """
    Expected JSON:
    {
      "image": "<base64 string>",
      "format": "image/jpeg"   # optional
    }
    """
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"success": False, "message": "Missing 'image' in request"}), 400

        image_b64 = data["image"]
        image_bytes = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)

        barcode = read_barcode(image_np)

        if barcode:
            # Fetch student details using the barcode as UID
            student_data = get_student_by_uid(barcode)
            first_name = None
            if student_data and len(student_data) > 0:
                name = student_data[0].get("Name", "Student")
                first_name = name.split(" ")[0] if name else "Student"

            return jsonify({
                "success": True,
                "barcode": barcode,
                "firstName": first_name,
                "message": "Barcode detected"
            })
        else:
            return jsonify({
                "success": False,
                "barcode": None,
                "message": "No barcode detected"
            })
    except Exception as e:
        app.logger.exception("Error processing barcode")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/student-profile/<uid>", methods=["GET"])
def get_student_profile(uid):
    try:
        data = get_student_by_uid(uid)
        if data and len(data) > 0:
            student = data[0]
            return jsonify({
                "success": True,
                "uid": student.get("UID"),
                "firstName": (student.get("Name") or "").split(" ")[0],
                "fullName": student.get("Name"),
                "number": student.get("Number"),
                "language": student.get("Language", "English"),
                "message": "Student found"
            })
        else:
            return jsonify({"success": False, "message": "Student not found"}), 404
    except Exception as e:
        app.logger.exception("Error fetching profile")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/update-language", methods=["POST"])
def update_language():
    """
    Expected JSON:
    {
      "uid": "123",
      "language": "HI"
    }
    """
    try:
        data = request.get_json(force=True)
        if not data or "uid" not in data or "language" not in data:
            return jsonify({"success": False, "message": "Missing uid or language"}), 400

        uid = data["uid"]
        language = data["language"]
        res = update_student_language(uid, language)
        return jsonify({"success": True, "data": res})
    except Exception as e:
        app.logger.exception("Error updating language")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok", "message": "Service is running"})


if __name__ == "__main__":
    # Use port from env (for cloud run) or default to 8000
    port = int(os.environ.get("PORT", 8000))
    # Debug True only for local dev
    app.run(host="0.0.0.0", port=port, debug=True)
