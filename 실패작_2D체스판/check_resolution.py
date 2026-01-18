"""
카메라 실제 해상도 및 설정 확인
"""

import cv2
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 0
RIGHT_CAM = 2

print("="*50)
print("카메라 해상도 및 설정 확인")
print("="*50)

cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

# 요청한 해상도
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 실제 해상도 확인
import time
time.sleep(1)

ret1, frame_left = cap_left.read()
ret2, frame_right = cap_right.read()

if ret1 and ret2:
    print(f"\n요청 해상도: 640x480")
    print(f"실제 왼쪽: {frame_left.shape[1]}x{frame_left.shape[0]}")
    print(f"실제 오른쪽: {frame_right.shape[1]}x{frame_right.shape[0]}")

    # 카메라 속성 확인
    print(f"\n왼쪽 카메라 설정:")
    print(f"  WIDTH: {cap_left.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  HEIGHT: {cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  ZOOM: {cap_left.get(cv2.CAP_PROP_ZOOM)}")
    print(f"  FOCUS: {cap_left.get(cv2.CAP_PROP_FOCUS)}")

    print(f"\n오른쪽 카메라 설정:")
    print(f"  WIDTH: {cap_right.get(cv2.CAP_PROP_FRAME_WIDTH)}")
    print(f"  HEIGHT: {cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
    print(f"  ZOOM: {cap_right.get(cv2.CAP_PROP_ZOOM)}")
    print(f"  FOCUS: {cap_right.get(cv2.CAP_PROP_FOCUS)}")

    # 화면에 표시
    cv2.putText(frame_left, f"LEFT: {frame_left.shape[1]}x{frame_left.shape[0]}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame_right, f"RIGHT: {frame_right.shape[1]}x{frame_right.shape[0]}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    combined = cv2.hstack([frame_left, frame_right])
    cv2.imshow("Camera Check", combined)

    print("\n'q' 키로 종료")

    while True:
        ret1, frame_left = cap_left.read()
        ret2, frame_right = cap_right.read()

        if ret1 and ret2:
            cv2.putText(frame_left, f"LEFT: {frame_left.shape[1]}x{frame_left.shape[0]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame_right, f"RIGHT: {frame_right.shape[1]}x{frame_right.shape[0]}",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            combined = cv2.hstack([frame_left, frame_right])
            cv2.imshow("Camera Check", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
