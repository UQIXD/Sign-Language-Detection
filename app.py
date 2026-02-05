# ================= INSTALL DEPENDENCY =================
# pip install ultralytics flask flask-cors pillow numpy torch opencv-python

import torch
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import io, base64

# ================= CEK DEVICE =================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Menggunakan Device: {device.upper()}")

# ================= INIT FLASK =================
app = Flask(__name__)
CORS(app)  # supaya HTML lokal tidak kena CORS

# ================= LOAD MODEL =================
model_path = "best.pt"  # SESUAIKAN PATH
model = YOLO(model_path)
model.to(device)

# ================= ROUTES =================
@app.route("/")
def home():
    return jsonify({
        "status": "LOCAL YOLOv8 SERVER ACTIVE",
        "device": device
    })

@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.get_json()
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "No image"}), 400

        header, encoded = img_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_w, img_h = img.size

        results = model.predict(
            img,
            conf=0.40,
            imgsz=640,
            verbose=False,
            device=device
        )

        draw = ImageDraw.Draw(img)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = model.names[cls_id].upper()

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()

            detections.append({
                "label": label,
                "confidence": conf
            })

            # === DRAW BOX ===
            draw.rectangle([x1, y1, x2, y2], outline="lime", width=4)
            draw.text((x1, y1 - 10), f"{label} {int(conf*100)}%", fill="lime")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return jsonify({
            "image": "data:image/jpeg;base64," + img_base64,
            "detections": detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ================= RUN SERVER =================
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",  # bisa diakses HP 1 jaringan
        port=5000,
        debug=True
    )
