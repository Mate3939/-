"""
Rectification 확인 - 좌우 이미지가 제대로 정렬되었는지 체크
"""
import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 2
RIGHT_CAM = 1
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz"

# 캘리브레이션 데이터 로드
calib_data = np.load(CALIB_FILE)
map1_l = calib_data['map1_l']
map2_l = calib_data['map2_l']
map1_r = calib_data['map1_r']
map2_r = calib_data['map2_r']

# 카메라 열기
cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("="*60)
print("Rectification 정렬 확인")
print("="*60)
print("좌우 이미지가 수평으로 정렬되어야 합니다.")
print("같은 물체가 같은 Y 좌표(높이)에 있어야 정상입니다.")
print()
print("'q' - 종료")
print("="*60)

while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        continue

    # Rectification 적용
    rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # 좌우 이미지 합치기
    combined = np.hstack([rect_left, rect_right])

    # 수평선 그리기 (정렬 확인용)
    h = combined.shape[0]
    for y in range(0, h, 50):
        cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)

    # 중앙 세로선 (좌우 구분)
    cv2.line(combined, (1280, 0), (1280, h), (0, 0, 255), 2)

    cv2.putText(combined, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(combined, "RIGHT", (1330, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    cv2.putText(combined, "Green lines: Check if objects align horizontally",
                (50, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 화면 크기 조정
    combined_resized = cv2.resize(combined, (1280, 360))
    cv2.imshow("Rectification Check", combined_resized)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
