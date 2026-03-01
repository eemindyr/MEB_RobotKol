import cv2
import numpy as np
import time

def analiz_baslat():
    # CAP_DSHOW Windows'ta kamerayı anında açar
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) 
    
    if not cap.isOpened():
        print("HATA: Kamera açılmadı!")
        return

    print("Kamera açıldı, 5 saniye geri sayım başlıyor...")
    
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret: break

        gecen_sure = time.time() - start_time
        kalan_sure = 5 - int(gecen_sure)

        # --- EKRANA GERİ SAYIM YAZMA ---
        # Görüntü üzerine büyük ve net bir yazı ekliyoruz
        cv2.rectangle(frame, (10, 10), (250, 70), (0,0,0), -1) # Arka plan paneli
        cv2.putText(frame, f"KALAN: {max(0, kalan_sure)}s", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Bu satır görüntünün ekranda tazelenmesini sağlar
        cv2.imshow("Robot Gozu - Hazirlik", frame)
        
        # 1 milisaniye bekle (ekranın çizilmesi için şarttır)
        if cv2.waitKey(1) & 0xFF == ord('q') or gecen_sure >= 5:
            break

    # 5 saniye dolunca son temiz kareyi analiz için sakla
    analiz_frame = frame.copy()
    cap.release()
    cv2.destroyWindow("Robot Gozu - Hazirlik")

    # --- GELİŞMİŞ ANALİZ (BLOB & GRID SİSTEMİ) ---
    hsv = cv2.cvtColor(analiz_frame, cv2.COLOR_BGR2HSV)
    
    # Renk Maskeleri
    mask_mavi = cv2.inRange(hsv, np.array([90, 80, 50]), np.array([130, 255, 255]))
    mask_kirmizi = cv2.inRange(hsv, np.array([0, 100, 70]), np.array([10, 255, 255])) + \
                   cv2.inRange(hsv, np.array([160, 100, 70]), np.array([180, 255, 255]))

    tespitler = []

    def bul(mask, renk_adi):
        mask = cv2.medianBlur(mask, 7) # Gürültü ve bulanıklık giderme
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            alan = cv2.contourArea(cnt)
            if alan < 400: continue
            
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Yarıçap yerine ALAN bazlı boyut ayrımı
            boyut = "30" if alan > 1300 else "15"
            tespitler.append({'x': cx, 'y': cy, 'renk': renk_adi, 'boyut': boyut})
            
            # Dağınık olmayan, temiz merkez noktası çizimi
            cv2.circle(analiz_frame, (cx, cy), 10, (0, 255, 0), -1)
            cv2.putText(analiz_frame, f"{renk_adi}-{boyut}", (cx-25, cy-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    bul(mask_mavi, "Mavi")
    bul(mask_kirmizi, "Kirmizi")

    # --- 3x3 IZGARA (GRID) MANTIĞI ---
    # Ekranı 3 sütun ve 3 satıra hayali olarak bölüyoruz
    h, w, _ = analiz_frame.shape
    matris = [["BOS", "BOS", "BOS"], ["BOS", "BOS", "BOS"], ["BOS", "BOS", "BOS"]]

    for t in tespitler:
        # Hangi sütun? (0, 1, 2)
        col = t['x'] // (w // 3)
        # Hangi satır? (0, 1, 2)
        row = t['y'] // (h // 3)
        
        # Sınır aşımını engelle
        col = min(2, max(0, col))
        row = min(2, max(0, row))
        
        matris[row][col] = f"{t['renk']}-{t['boyut']}"

    print("\n--- KESİN 3x3 MATRİS ---")
    for r_idx, r in enumerate(matris):
        print(f"Satir {r_idx+1}: {' | '.join(r)}")

    cv2.imshow("Final Analiz Sonucu", analiz_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

analiz_baslat()