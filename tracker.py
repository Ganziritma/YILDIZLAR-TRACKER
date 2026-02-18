import cv2
import sys

# 1. Tracker Oluştur (CSRT Seçimi)
# OpenCV sürümüne göre fonksiyon değişebilir ama genelde budur:
tracker = cv2.TrackerCSRT_create()

# 2. Videoyu veya Kamerayı Aç (0 webcam demektir)
video = cv2.VideoCapture(0)

# Kamera açıldı mı kontrol et
if not video.isOpened():
    print("Kamera açılamadı!")
    sys.exit()

# 3. İlk Kareyi Oku
ok, frame = video.read()
if not ok:
    print("Video okunamadı")
    sys.exit()

# 4. Takip Edilecek Hedefi Seç (Açılan pencerede mouse ile kutu çizip ENTER'a bas)
print("Lutfen takip edilecek nesneyi secin ve ENTER'a basin.")
bbox = cv2.selectROI("Takip Ekrani", frame, False)
cv2.destroyWindow("Takip Ekrani") # Seçim ekranını kapat

# 5. Tracker'ı Başlat (Init)
ok = tracker.init(frame, bbox)

while True:
    # Yeni kareyi oku
    ok, frame = video.read()
    if not ok:
        break
    
    # 6. Tracker'ı Güncelle (Update) - Sihir burada oluyor
    basarili, kutu = tracker.update(frame)
    
    if basarili:
        # Kutuyu çiz (x, y, genişlik, yükseklik)
        (x, y, w, h) = [int(v) for v in kutu]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, 1)
        cv2.putText(frame, "Hedef Kilitlendi (CSRT)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        # Takip koptuysa kırmızı yaz
        cv2.putText(frame, "HEDEF KAYIP!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Görüntüyü göster
    cv2.imshow("Savasan IHA Tracker Test", frame)
    
    # 'q' tuşuna basınca çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()