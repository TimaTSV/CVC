import cv2
import numpy as np

fps = 20.0
seconds = 30
total_frames = int(fps * seconds)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, fps, (640, 480))
x, y = 20.0, 220.0

for i in range(total_frames):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    x += 1.0 # рух
    cv2.rectangle(frame, (int(x), int(y)), (int(x) + 60, int(y) + 40), (0, 255, 0), -1)
    cv2.putText(frame, f"Frame: {i} | Time: {i/fps:.1f}s", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out.write(frame)

out.release()
print(f"Відео 'test.avi' на {seconds} секунд створено!")