import cv2
import numpy as np
import os
import pickle
from tkinter import messagebox

def Prediction(area, perimeter):
    filename = 'model/knn_shrimp_model.sav'  # ที่อยู่ของไฟล์โมเดล
    model = pickle.load(open(filename, 'rb'))
    Pred = model.predict([[area, perimeter]])
    return Pred

def Shrimp_Detection(image_path):
    original = cv2.imread(image_path)
    if original is None:
        print(f"ไม่สามารถโหลดภาพ {image_path} ได้")
        return None, None, None
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"ไม่พบกุ้งใน {image_path}")
        return None, None, None

    best_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_contour)
    perimeter = cv2.arcLength(best_contour, True)

    image_with_box = original.copy()
    x, y, w, h = cv2.boundingRect(best_contour)
    cv2.rectangle(image_with_box, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return area, perimeter, image_with_box, (x, y, w, h)

def Capture():
    name = 'Shrimp'  
    directory = f'prediction/{name}/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("press space to take a photo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("press space to take a photo", 500, 300)
    img_counter = 1

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("press space to take a photo", frame)

        k = cv2.waitKey(1)
        if k % 256 == 27:
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            img_name = f"{directory}image_{img_counter}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1
            area, perimeter, image_with_box, bbox = Shrimp_Detection(img_name)
            if area is not None and perimeter is not None:
                Pred = Prediction(area, perimeter)
                label = f"Size: {Pred} Units/kg"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                x, y, w, h = bbox

                text_x = x + (w - text_size[0]) // 2
                text_y = y - 10 if y - 10 > 10 else y + h + 20
                
                cv2.rectangle(
                    image_with_box,
                    (text_x - 5, text_y - text_size[1] - 5),
                    (text_x + text_size[0] + 5, text_y + 5),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(image_with_box, label, (text_x, text_y), font, font_scale, (255, 255, 255), thickness)
                
                cv2.namedWindow("Prediction Result", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Prediction Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                cv2.imshow("Prediction Result", image_with_box)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                messagebox.showerror("ข้อผิดพลาด", "ไม่สามารถทำนายได้")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    Capture()
