"""YOLOv10 垃圾分类检测 — FastAPI 后端
启动: uvicorn backend.server:app --host 0.0.0.0 --port 8000
"""
import io
import time
import base64
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import cv2
import numpy as np
from ultralytics import YOLOv10

# --- 配置 ---
MODEL_PATH = Path(__file__).parent.parent / "runs" / "detect" / "train5" / "weights" / "best.pt"
CLASS_NAMES = {
    0: {"en": "recyclable waste", "zh": "可回收垃圾"},
    1: {"en": "hazardous waste", "zh": "有害垃圾"},
    2: {"en": "kitchen waste", "zh": "厨余垃圾"},
    3: {"en": "other waste", "zh": "其他垃圾"},
}
COLORS = {
    0: [56, 189, 248],   # 蓝 #38bdf8
    1: [245, 158, 11],   # 橙 #f59e0b
    2: [16, 185, 129],   # 绿 #10b981
    3: [148, 163, 184],  # 灰 #94a3b8
}

model = None


def get_model():
    global model
    if model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
        model = YOLOv10(str(MODEL_PATH))
    return model


def annotate_image(image: np.ndarray, results) -> np.ndarray:
    """在图片上绘制检测框和中文标签"""
    annotated = image.copy()
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            color = COLORS.get(cls_id, [255, 255, 255])
            name_zh = CLASS_NAMES.get(cls_id, {}).get("zh", "未知")

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            label = f"{name_zh} {conf:.0%}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return annotated


def image_to_base64(image: np.ndarray) -> str:
    """numpy 图片转 base64 data URI"""
    _, buffer = cv2.imencode(".jpg", image)
    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


# --- FastAPI 应用 ---
app = FastAPI(title="YOLOv10 垃圾分类检测 API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    try:
        m = get_model()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_path": str(MODEL_PATH),
            "classes": CLASS_NAMES,
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "model_loaded": False, "error": str(e)},
        )


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        m = get_model()
        t0 = time.time()
        results = m.predict(source=img_array, imgsz=imgsz, conf=conf)
        elapsed_ms = (time.time() - t0) * 1000

        annotated = annotate_image(img_array, results)

        detections = []
        if results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES[cls_id]["en"],
                    "class_name_zh": CLASS_NAMES[cls_id]["zh"],
                    "confidence": round(float(box.conf[0]), 4),
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                })

        return {
            "success": True,
            "image_base64": image_to_base64(annotated),
            "detections": detections,
            "count": len(detections),
            "inference_time_ms": round(elapsed_ms, 1),
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


@app.post("/api/batch-detect")
async def batch_detect(
    images: list[UploadFile] = File(...),
    conf: float = Form(0.25),
    imgsz: int = Form(640),
):
    try:
        m = get_model()
        all_results = []
        total_detections = 0
        summary = {CLASS_NAMES[i]["en"]: 0 for i in range(4)}

        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            for img_file in images:
                contents = await img_file.read()
                pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
                img_array = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

                t0 = time.time()
                results = m.predict(source=img_array, imgsz=imgsz, conf=conf)
                elapsed_ms = (time.time() - t0) * 1000

                annotated = annotate_image(img_array, results)

                _, buf = cv2.imencode(".jpg", annotated)
                zf.writestr(f"annotated_{img_file.filename}", buf.tobytes())

                detections = []
                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        cls_id = int(box.cls[0])
                        cls_en = CLASS_NAMES[cls_id]["en"]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        detections.append({
                            "class_id": cls_id,
                            "class_name": cls_en,
                            "class_name_zh": CLASS_NAMES[cls_id]["zh"],
                            "confidence": round(float(box.conf[0]), 4),
                            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                        })
                        summary[cls_en] += 1

                all_results.append({
                    "filename": img_file.filename,
                    "image_base64": image_to_base64(annotated),
                    "detections": detections,
                    "count": len(detections),
                    "inference_time_ms": round(elapsed_ms, 1),
                })
                total_detections += len(detections)

        zip_buffer.seek(0)
        zip_b64 = base64.b64encode(zip_buffer.read()).decode("utf-8")

        return {
            "success": True,
            "total_images": len(images),
            "total_detections": total_detections,
            "results": all_results,
            "summary": summary,
            "zip_base64": f"data:application/zip;base64,{zip_b64}",
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
