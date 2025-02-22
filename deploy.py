import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import pickle

def Prediction(area, perimeter):
    filename = 'model/knn_shrimp_model.sav' # แก้ไขเป็นที่อยู่ของไฟล์โมเดล
    model = pickle.load(open(filename, 'rb'))
    Pred = model.predict([[area, perimeter]])
    return Pred

def Shrimp_Detection(image_path):

    original = cv2.imread(image_path)
    if original is None:
        print(f"ไม่สามารถโหลดภาพ {image_path} ได้")
        return None, None
    
    gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"ไม่พบกุ้งใน {image_path}")
        return None, None

    best_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(best_contour)
    perimeter = cv2.arcLength(best_contour, True)

    return area, perimeter

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
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = f"{directory}image_{img_counter}.jpg"
            cv2.imwrite(img_name, frame)
            print(f"{img_name} written!")
            img_counter += 1
            area, perimeter = Shrimp_Detection(img_name)
            if area is not None and perimeter is not None:
                print(f"พื้นที่: {area:.2f} พิกเซล")
                print(f"เส้นรอบวง: {perimeter:.2f} พิกเซล")
                Pred = Prediction(area, perimeter)
                print(f"ผลการทำนาย: {Pred[0]}")
            else:
                print("ไม่สามารถทำนายได้")

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    Capture()
