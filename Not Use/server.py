import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request, jsonify
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io, base64, json

app = Flask(__name__, static_folder="static", template_folder="templates")

# === LOAD MODEL ===
model = YOLO("./best.pt")  # ganti path kalau berbeda
model.to('cpu')  # ← PAKSA YOLO HANYA PAKAI CPU
torch.set_num_threads(4)  # batasi agar tidak overheat CPU

# === KELAS ===
CLASS_NAMES = [
    "A",
    "Ayah",
    "B",
    "D",
    "H",
    "Halo",
    "Kakak",
    "L",
    "Minum",
    "Makan"
]


@app.route("/")
def index():
    return render_template("index.html", class_names=json.dumps(CLASS_NAMES))


@app.route("/detect", methods=["POST"])
def detect():
    try:
        data = request.json
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "no image"}), 400

        # Decode base64 → image
        header, encoded = img_b64.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # === YOLOv8 DETECTION ===
        results = model.predict(img, conf=0.25)

        draw = ImageDraw.Draw(img)
        detections = []

        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            label = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else str(cls_id)
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            x1, y1, x2, y2 = xyxy

            # Draw bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            text = f"{label} {conf:.2f}"
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except:
                font = ImageFont.load_default()
            tw, th = draw.textsize(text, font=font)
            draw.rectangle([x1, y1 - th - 4, x1 + tw + 4, y1], fill="red")
            draw.text((x1 + 2, y1 - th - 2), text, fill="white", font=font)

            detections.append({"label": label, "confidence": round(conf, 3)})

        # Encode hasil gambar
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")
        img_str = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

        return jsonify({"detections": detections, "image": img_str})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
