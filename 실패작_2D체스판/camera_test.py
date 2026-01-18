"""
카메라 전체 스캔 (여러 백엔드 시도)
"""

import cv2
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

print("카메라 스캔 중...")
print("="*50)

working_cameras = []

# 먼저 기본 방식으로 시도
for i in range(11):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"Camera {i}: O ({frame.shape[1]}x{frame.shape[0]})")
            working_cameras.append(i)
        cap.release()

print("="*50)
print(f"발견된 카메라: {working_cameras}")

if len(working_cameras) == 0:
    print("\n카메라가 없습니다!")
    input("엔터를 눌러 종료...")
    exit()

print(f"\n총 {len(working_cameras)}개 카메라 창을 엽니다.")
print("'q' 키로 종료\n")

# 모든 카메라 열기
caps = {}
for idx in working_cameras:
    cap = cv2.VideoCapture(idx)
    caps[idx] = cap

while True:
    for idx, cap in caps.items():
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, f"CAM {idx}", (30, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow(f"Camera {idx}", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

for cap in caps.values():
    cap.release()
cv2.destroyAllWindows()

print("\n스캔 완료!")
print("HD Webcam Pro 2개가 몇 번인지 알려주세요.")


