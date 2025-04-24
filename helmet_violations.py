from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np
from paddleocr import PaddleOCR
import os

# Load YOLO model
trained_model = YOLO("models/best.pt")

# Load PaddleOCR
ocr = PaddleOCR(use_angle_cls=True, lang='en', det_db_box_thresh=0.3, rec_algorithm='CRNN')

# Class names from your model
class_names = ["number plate", "rider", "with helmet", "without helmet"]

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

def correct_common_ocr_errors(text):
    replacements = {'O': '0'}
    return ''.join(replacements.get(c, c) for c in text)

def is_inside(inner, outer):
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    return ix1 >= ox1 and iy1 >= oy1 and ix2 <= ox2 and iy2 <= oy2

def get_center(xyxy):
    x1, y1, x2, y2 = map(int, xyxy)
    return ((x1 + x2) // 2, (y1 + y2) // 2)

def number_plate_det(res, plate_box, plate_index=0):
    x1, y1, x2, y2 = map(int, plate_box.xyxy[0])
    padding = 15
    x1 = max(x1 - padding, 0)
    y1 = max(y1 - padding, 0)
    x2 = min(x2 + padding, res.orig_img.shape[1])
    y2 = min(y2 + padding, res.orig_img.shape[0])
    crop = res.orig_img[y1:y2, x1:x2]

    if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
        return ""  # Skip if crop is too small

    ocr_result = ocr.ocr(crop, cls=True)

    if ocr_result and len(ocr_result) > 0 and len(ocr_result[0]) > 0:
        full_text = ""
        for line in ocr_result[0]:
            part = line[1][0].strip().replace(" ", "")
            full_text += part
        full_text = full_text.upper()
        corrected_text = correct_common_ocr_errors(full_text)
        return corrected_text  # ✅ Only return corrected plate
    else:
        print(f"❌ Plate {plate_index + 1} - OCR failed or returned no usable text.")
        return ""  # OCR failed


def detect_helmet_violations(image_path):
    result = trained_model(image_path)[0]
    helmetless_boxes = []
    number_plate_boxes = []

    for box in result.boxes:
        class_id = int(box.cls)
        label = class_names[class_id]

        if label == "without helmet":
            helmetless_boxes.append(box)
        elif label == "number plate":
            number_plate_boxes.append(box)

    processed_rider_boxes = []
    used_plate_boxes = []
    messages = []

    if helmetless_boxes:
        for i, helmetless_box in enumerate(helmetless_boxes):
            matched_rider_box = None
            for rider_box in result.boxes:
                if int(rider_box.cls) == class_names.index("rider"):
                    if is_inside(helmetless_box.xyxy[0], rider_box.xyxy[0]):
                        if rider_box not in processed_rider_boxes:
                            matched_rider_box = rider_box
                            processed_rider_boxes.append(rider_box)
                            break

            if not matched_rider_box:
                continue

            rider_xyxy = matched_rider_box.xyxy[0]
            inside_plates = [
                plate_box for plate_box in number_plate_boxes
                if is_inside(plate_box.xyxy[0], rider_xyxy) and plate_box not in used_plate_boxes
            ]

            if inside_plates:
                # Calculate the center of the helmetless rider once
                rider_cx, rider_cy = get_center(helmetless_box.xyxy[0])

                # Define the distance function here
                def distance_to_rider_center(plate_box):
                    px, py = get_center(plate_box.xyxy[0])
                    return ((rider_cx - px) ** 2 + (rider_cy - py) ** 2) ** 0.5

                # Choose the closest plate
                matched_plate = min(inside_plates, key=distance_to_rider_center)
                used_plate_boxes.append(matched_plate)

                corrected_text = number_plate_det(result, matched_plate, i)
                if corrected_text:
                    messages.append(f"✅ Helmetless Rider {i + 1}: Detected number plate - {corrected_text}")
                else:
                    messages.append(f"❌ Helmetless Rider {i + 1}: OCR failed on the detected number plate")
            else:
                messages.append(f"❌ Helmetless Rider {i + 1}: No number plate detected inside the rider's bounding box")

    else:
        messages.append("✅ All riders appear to be wearing helmets.")

    # Annotate image
    annotator = Annotator(result.orig_img)
    for box in result.boxes:
        label = class_names[int(box.cls)]
        annotator.box_label(box.xyxy[0], label, color=(255, 0, 0))
    annotated_img = annotator.result()
    annotated_path = os.path.join("uploads", "annotated_result.jpg")
    cv2.imwrite(annotated_path, annotated_img)

    return "\n".join(messages), f"uploads/annotated_result.jpg"
