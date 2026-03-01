import cv2
import numpy as np
import time

def kutu_analiz():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened(): return

    # Işık ayarı için 5 saniye bekleme
    start = time.time()
    while time.time() - start < 5:
        ret, frame = cap.read()
        cv2.putText(frame, f"KUTU ANALIZINE: {5-int(time.time()-start)}s", (20, 40), 1, 2, (255,255,0), 2)
        cv2.imshow("Kamera", frame)
        cv2.waitKey(1)
    
    cap.release()
    cv2.destroyAllWindows()

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 1. KUTU DIŞ ÇERÇEVELERİNİ BUL (60x60)
    # Kırmızı ve Mavi kutuların dış gövde maskeleri
    mask_k_kutu = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255])) + \
                  cv2.inRange(hsv, np.array([160, 100, 50]), np.array([180, 255, 255]))
    mask_m_kutu = cv2.inRange(hsv, np.array([100, 100, 50]), np.array([130, 255, 255]))

    kutu_tespitleri = []

    def kutulari_isle(mask, renk_adi):
        mask = cv2.medianBlur(mask, 5)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            alan = cv2.contourArea(cnt)
            # 60x60 kutunun alanı uzaklığa göre değişir, yaklaşık 2500+ pikseldir
            if alan < 1500: continue 

            M = cv2.moments(cnt)
            if M["m00"] == 0: continue
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

            # 2. KUTUNUN İÇİNDEKİ NESNEYİ ANALİZ ET (15/30)
            # Kutu merkezindeki küçük bir alanda (ROI) nesne var mı bakıyoruz
            roi = mask[cy-20:cy+20, cx-20:cx+20]
            nesne_alani = np.sum(roi == 255)
            
            # Eğer kutu içinde nesne varsa boyutunu belirle
            boyut = "30" if nesne_alani > 1000 else "15"
            
            kutu_tespitleri.append({'x': cx, 'y': cy, 'renk': renk_adi, 'boyut': boyut})
            
            # Görselleştirme (Kutu dış hatlarını çiz)
            x_b, y_b, w_b, h_b = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x_b, y_b), (x_b+w_b, y_b+h_b), (0, 255, 0), 2)
            cv2.putText(frame, f"{renk_adi} Kutu-{boyut}", (cx-40, cy-10), 1, 1, (255,255,255), 2)

    kutulari_isle(mask_k_kutu, "Kirmizi")
    kutulari_isle(mask_m_kutu, "Mavi")

    # --- 2x2 MATRİS SIRALAMA ---
    if len(kutu_tespitleri) < 1: 
        print("Kutu bulunamadı!")
        return

    # Y koordinatına göre 2 satıra böl
    kutu_tespitleri.sort(key=lambda b: b['y'])
    satirlar = [kutu_tespitleri[:2], kutu_tespitleri[2:]]
    
    print("\n--- 2x2 KUTU MATRİSİ ---")
    for i, satir in enumerate(satirlar):
        satir.sort(key=lambda b: b['x']) # Soldan sağa sırala
        row_str = f"Sıra {i+1}: "
        for k in satir:
            row_str += f"| {k['renk']} ({k['boyut']}) | "
        print(row_str)

    cv2.imshow("Kutu Analiz Sonucu", frame)
    cv2.waitKey(0)

kutu_analiz()