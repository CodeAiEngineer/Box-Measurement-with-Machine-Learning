import cv2
from object_detector import *
import numpy as np
import pandas as pd
wlist=[]
hlist=[]
#  Aruco detektor yükle
parameters = cv2.aruco.DetectorParameters_create()
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_50)


# Nesne detektor yükle
detector = HomogeneousBgDetector()

# Resim yükle
# img = cv2.imread("phone_aruco_marker.jpg")
img = cv2.imread("phone_aruco_marker(4).jpg")

#  Aruco işaretleyici yükle
corners, _, _ = cv2.aruco.detectMarkers(img, aruco_dict, parameters=parameters)

# İşaretçinin etrafına çokgen çiz
int_corners = np.int0(corners)
cv2.polylines(img, int_corners, True, (0, 255, 0), 5)

# Aruco Çevre uzunluğu
aruco_perimeter = cv2.arcLength(corners[0], True)

# Piksel cm oranı
pixel_cm_ratio = aruco_perimeter / 20
    
contours = detector.detect_objects(img)

# Nesne sınırlarını çiz
for cnt in contours:
    # Get rect
    rect = cv2.minAreaRect(cnt)
    (x, y), (w, h), angle = rect

    # Oran pikselini cm'ye uygulayarak Nesnelerin Genişliğini ve Yüksekliğini Al
    object_width = w / pixel_cm_ratio
    object_height = h / pixel_cm_ratio
    wlist.append(object_width)
    hlist.append(object_height)

    # dörtgeni göster
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.circle(img, (int(x), int(y)), 5, (0, 0, 255), -1)
    cv2.polylines(img, [box], True, (255, 0, 0), 2)
    cv2.putText(img, "Width {} cm".format(round(object_width, 1)), (int(x - 100), int(y - 20)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)
    cv2.putText(img, "Height {} cm".format(round(object_height, 1)), (int(x - 100), int(y + 15)), cv2.FONT_HERSHEY_PLAIN, 2, (100, 200, 0), 2)

# w ve h değerlerini tek tek alıp, birleştirip excel'e koy.
c=zip(wlist,hlist)
# print(list(c))
d=list(c)
print(d)
print(type(d))
pd.DataFrame(d).to_excel('output2.xlsx', header=False, index=False)
cv2.imshow("Image", img)
cv2.waitKey(0)
