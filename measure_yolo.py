"""
YOLOv8 + 스테레오 거리 측정
- 객체 감지 (YOLO)
- 감지된 객체까지의 거리 자동 측정
"""
import cv2
import numpy as np
import os
from ultralytics import YOLO

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 2
RIGHT_CAM = 1
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz"

print("="*60)
print("YOLOv8 + 스테레오 거리 측정")
print("="*60)
print("YOLOv8 모델 로드 중...")

# YOLOv8 모델 로드 (첫 실행 시 자동 다운로드)
model = YOLO('yolov8n.pt')  # nano 버전 (빠름)

print("✓ YOLO 모델 로드 완료")
print()

# 캘리브레이션 데이터 로드
if not os.path.exists(CALIB_FILE):
    print(f"❌ 캘리브레이션 파일이 없습니다")
    exit(1)

calib_data = np.load(CALIB_FILE)
map1_l = calib_data['map1_l']
map2_l = calib_data['map2_l']
map1_r = calib_data['map1_r']
map2_r = calib_data['map2_r']
Q = calib_data['Q']

# 카메라 열기
cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("[ 화면 구성 ]")
print("  Main View: 오른쪽 카메라 + YOLO 감지 + 거리 표시")
print("  Depth Map: 흑백 거리 맵 (밝음=가까움)")
print()
print("[ 조작법 ]")
print("  'q' - 종료")
print()
print("  슬라이더:")
print("    Confidence - YOLO 신뢰도 임계값")
print("    NumDisp(x16) - 디스패리티 범위")
print("    BlockSize - 매칭 블록 크기")
print("="*60)

# 윈도우 생성
cv2.namedWindow("Main View (YOLO + Distance)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Main View (YOLO + Distance)", 1280, 720)
cv2.resizeWindow("Depth Map", 640, 360)

def trackbar_callback(x):
    pass

cv2.createTrackbar("Confidence %", "Main View (YOLO + Distance)", 50, 100, trackbar_callback)
cv2.createTrackbar("NumDisp(x16)", "Main View (YOLO + Distance)", 8, 16, trackbar_callback)
cv2.createTrackbar("BlockSize", "Main View (YOLO + Distance)", 11, 25, trackbar_callback)

print("\n프로그램 실행 중...\n")

frame_count = 0

while True:
    frame_count += 1

    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        continue

    # Rectification
    rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # 오른쪽 카메라를 메인으로
    display = rect_right.copy()

    # 그레이스케일 (depth 계산용)
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.equalizeHist(gray_left)
    gray_right = cv2.equalizeHist(gray_right)

    # 슬라이더
    confidence = cv2.getTrackbarPos("Confidence %", "Main View (YOLO + Distance)") / 100.0
    num_disp_mult = max(1, cv2.getTrackbarPos("NumDisp(x16)", "Main View (YOLO + Distance)"))
    num_disp = 16 * num_disp_mult
    block_size = cv2.getTrackbarPos("BlockSize", "Main View (YOLO + Distance)")
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 5:
        block_size = 5

    # 스테레오 매처
    stereo = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=15,
        speckleWindowSize=100,
        speckleRange=32,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )

    # Disparity 계산
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # 3D 포인트 계산
    points_3d = cv2.reprojectImageTo3D(disparity, Q)

    # Depth map 시각화
    disp_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_visual = np.uint8(disp_visual)

    # YOLO 객체 감지 (매 프레임마다는 부하가 크므로 3프레임마다 실행)
    if frame_count % 3 == 0:
        results = model(rect_right, conf=confidence, verbose=False)
        detections = results[0].boxes
    else:
        # 이전 결과 재사용
        pass

    # 감지된 객체 처리
    if frame_count % 3 == 0 and len(detections) > 0:
        for box in detections:
            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Bounding box 중심점
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # 중심점의 거리 측정
            if 0 <= cx < disparity.shape[1] and 0 <= cy < disparity.shape[0]:
                depth = points_3d[cy, cx][2]

                if 0 < depth < 10000:  # 유효한 거리
                    distance_text = f"{depth:.0f}mm ({depth/10:.1f}cm)"

                    # Bounding box 그리기
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # 라벨 + 거리
                    text = f"{label} {conf:.2f}"
                    cv2.putText(display, text, (x1, y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, distance_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    # 중심점 표시
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    # 거리 측정 불가
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    text = f"{label} {conf:.2f} [NO DEPTH]"
                    cv2.putText(display, text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # 정보 표시
    cv2.putText(display, f"YOLO Conf: {confidence:.2f} | NumDisp: {num_disp} | Block: {block_size}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Main View (YOLO + Distance)", display)
    cv2.imshow("Depth Map", disp_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print("\n프로그램 종료")
