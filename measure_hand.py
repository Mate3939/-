"""
YOLO + 손 감지 + 스테레오 거리 측정 (단일 화면)
- 1개 창에 원본 + Depth 분할 표시
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
print("YOLO + 거리 측정 (단일 화면)")
print("="*60)
print("모델 로드 중...")

# YOLO 모델
model = YOLO('yolov8n.pt')
print("✓ YOLO 로드 완료")

# 캘리브레이션 데이터
print("\n캘리브레이션 데이터 로드 중...")
if not os.path.exists(CALIB_FILE):
    print(f"❌ 캘리브레이션 파일이 없습니다: {CALIB_FILE}")
    exit(1)

calib_data = np.load(CALIB_FILE)
map1_l = calib_data['map1_l']
map2_l = calib_data['map2_l']
map1_r = calib_data['map1_r']
map2_r = calib_data['map2_r']
Q = calib_data['Q']
print("✓ 캘리브레이션 데이터 로드 완료")

# 카메라
print(f"\n카메라 열기 중...")
print(f"  LEFT (보조): 카메라 {LEFT_CAM}")
print(f"  RIGHT (메인): 카메라 {RIGHT_CAM}")

cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

if not cap_left.isOpened():
    print(f"❌ 왼쪽 카메라 {LEFT_CAM}를 열 수 없습니다!")
    exit(1)

if not cap_right.isOpened():
    print(f"❌ 오른쪽 카메라 {RIGHT_CAM}를 열 수 없습니다!")
    exit(1)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"✓ 카메라 열기 완료")
print(f"  LEFT: {int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"  RIGHT: {int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print()

print("[ 화면 레이아웃 - 단일 창 ]")
print("  좌측: 원본 컬러 (오른쪽 카메라 + YOLO)")
print("  우측: Depth Map (흑백)")
print()
print("[ 카메라 역할 ]")
print("  오른쪽 카메라: 메인 (화면 출력)")
print("  왼쪽 카메라: 보조 (거리 측정용)")
print()
print("[ 조작법 ]")
print("  'q' - 종료")
print("="*60)

# 단일 윈도우
cv2.namedWindow("Stereo Vision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Stereo Vision", 1600, 720)

def trackbar_callback(x):
    pass

cv2.createTrackbar("YOLO Conf %", "Stereo Vision", 50, 100, trackbar_callback)
cv2.createTrackbar("NumDisp(x16)", "Stereo Vision", 8, 16, trackbar_callback)
cv2.createTrackbar("BlockSize", "Stereo Vision", 11, 25, trackbar_callback)

print("\n프로그램 실행 중...\n")

frame_count = 0
yolo_results = None

while True:
    frame_count += 1

    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        continue

    # Rectification
    rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # Main View = 오른쪽 카메라 원본 (컬러)
    display = rect_right.copy()

    # 그레이스케일 (depth용)
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.equalizeHist(gray_left)
    gray_right = cv2.equalizeHist(gray_right)

    # 슬라이더
    yolo_conf = cv2.getTrackbarPos("YOLO Conf %", "Stereo Vision") / 100.0
    num_disp_mult = max(1, cv2.getTrackbarPos("NumDisp(x16)", "Stereo Vision"))
    num_disp = 16 * num_disp_mult
    block_size = cv2.getTrackbarPos("BlockSize", "Stereo Vision")
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 5:
        block_size = 5

    # 스테레오 매처 (오른쪽 카메라 기준)
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

    # Disparity 계산 - 오른쪽 기준
    disparity = stereo.compute(gray_right, gray_left).astype(np.float32) / 16.0

    # Q 매트릭스 오른쪽 기준 수정
    Q_right = Q.copy()
    Q_right[3, 2] = -Q_right[3, 2]
    points_3d = cv2.reprojectImageTo3D(disparity, Q_right)

    # Depth map 시각화
    disp_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_visual = np.uint8(disp_visual)
    depth_colored = cv2.cvtColor(disp_visual, cv2.COLOR_GRAY2BGR)

    # YOLO 객체 감지 (3프레임마다)
    if frame_count % 3 == 0:
        yolo_results = model(rect_right, conf=yolo_conf, verbose=False)

    # YOLO 결과 그리기
    if yolo_results is not None and len(yolo_results[0].boxes) > 0:
        for box in yolo_results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # 중심점 거리
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            if 0 <= cx < disparity.shape[1] and 0 <= cy < disparity.shape[0]:
                depth = points_3d[cy, cx][2]

                if 0 < depth < 10000:
                    # 원본에 표시
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"{label} {conf:.2f}"
                    dist_text = f"{depth:.0f}mm ({depth/10:.1f}cm)"
                    cv2.putText(display, text, (x1, y1 - 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display, dist_text, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)

                    # Depth map에도 표시
                    cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.circle(depth_colored, (cx, cy), 5, (0, 0, 255), -1)

    # 정보 표시
    cv2.putText(display, f"Original (Right Camera + YOLO)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display, f"Conf: {yolo_conf:.2f} | Disp: {num_disp} | Block: {block_size}",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(depth_colored, "Depth Map (Brightness = Distance)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(depth_colored, "Bright = Close, Dark = Far", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 좌우 합치기
    combined = np.hstack([display, depth_colored])

    # 중앙 구분선
    h = combined.shape[0]
    cv2.line(combined, (1280, 0), (1280, h), (0, 255, 0), 3)

    cv2.imshow("Stereo Vision", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print("\n프로그램 종료")
