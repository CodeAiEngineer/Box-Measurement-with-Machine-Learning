import cv2


class HomogeneousBgDetector():
    def __init__(self):
        pass

    def detect_objects(self, frame):
        # Görüntüyü gri tonlamaya dönüştür
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Uyarlanabilir eşiğe sahip bir Maske oluştur.
        mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 5)


        # edged = cv2.Canny(mask, -1000, 1)


        # Kontür bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #cv2.imshow("mask", mask)
        objects_contours = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 2000:
                #cnt = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)
                objects_contours.append(cnt)

        return objects_contours

    # def get_objects_rect(self):
    #     box = cv2.boxPoints(rect)  # cv2.boxPoints(rect) for OpenCV 3.x
    #     box = np.int0(box)
    
    
    # sadece kutuları almamızı saglayacak kodlar.
    
    
    # edged = cv2.Canny(gray, 30, 200)

    # # find contours in the edged image, keep only the largest
    # # ones, and initialize our screen contour
    
    
    # cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    # cnts = imutils.grab_contours(cnts)
    # cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
    # screenCnt = None
