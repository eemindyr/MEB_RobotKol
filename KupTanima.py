import cv2
import numpy as np
import time

def kup_tani():
    #Kamerayı Tanı
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    if not cap.isOpened():
        print("HATA: Kamera kanalı yanlış")
        return
    
    #Kamera Açıldıktan Sonra Geri Sayım
    baslangic = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: print("Kamera Var Ama Okumuyor") #Çokta Önemli Değil
        gecen_sure = time.time() - baslangic
        cv2.waitKey(1)
        if gecen_sure >= 3:
            print("3 saniye doldu")
            break          

    #Son Kareyi Yakala
    analiz_frame = frame.copy()
    cap.release()

    #Analiz
    blur = cv2.GaussianBlur(frame, (5, 5), 0) # Eğer Işıklar Güçlüyse 9-9 Yap
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask_mavi = cv2.inRange(hsv, np.array([90, 100, 50]), np.array([115, 255, 255])) + \
                cv2.inRange(hsv, np.array([115, 100, 50]), np.array([130, 255, 255]))
    mask_kirmizi = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([160, 100, 70]), np.array([180, 255, 255]))

    kupKordinatlari = []
    def bul(mask, renk_adi):
        mask = cv2.medianBlur(mask, 7)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            alan = cv2.contourArea(cnt)
            if alan < 400: continue
            
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Yarıçap yerine ALAN bazlı boyut ayrımı
            boyut = "b" if alan > 1300 else "k"
            kupKordinatlari.append({'x': cx, 'y': cy, 'renk': renk_adi, 'boyut': boyut})

    bul(mask_mavi, "m")
    bul(mask_kirmizi, "k")

    h, w, _ = analiz_frame.shape
    matris = [["BOS", "BOS", "BOS"], ["BOS", "BOS", "BOS"], ["BOS", "BOS", "BOS"]]

    for t in kupKordinatlari:
        col = t['x'] // (w // 3)
        row = t['y'] // (h // 3)
        
        col = min(2, max(0, col))
        row = min(2, max(0, row))
        
        matris[row][col] = f"{t['renk']}{t['boyut']}"

    cv2.waitKey(0)
    cv2.destroyAllWindows()

