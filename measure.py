"""
거리 측정 v2
- 흑백 명암비 (가까움=밝게, 멀리=어둡게)
- HD 해상도 (1280x720)
- alpha=1 (전체 FOV)
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 2
RIGHT_CAM = 1
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz"

# 캘리브레이션 데이터 로드
if not os.path.exists(CALIB_FILE):
    print(f"❌ 캘리브레이션 파일이 없습니다: {CALIB_FILE}")
    print("먼저 calibration_v2.py를 실행하세요!")
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

# HD 해상도
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("="*60)
print("거리 측정 v2 - 흑백 명암비 (왼쪽 카메라 기준)")
print("="*60)
print("[ 카메라 설정 ]")
print("  왼쪽 카메라 (카메라 2) = 레퍼런스 (OpenCV 표준)")
print("  오른쪽 카메라 (카메라 1) = 보조 (거리 계산용)")
print()
print("[ 흑백 명암 ]")
print("  밝은 부분 (흰색) = 가까운 거리")
print("  어두운 부분 (검정) = 먼 거리")
print()
print("[ 조작법 ]")
print("  마우스 클릭 - 거리 측정")
print("  's' - 측정값 기록")
print("  'q' - 종료")
print()
print("  슬라이더:")
print("    NumDisp(x16) - 디스패리티 범위")
print("    BlockSize - 매칭 블록 크기")
print("="*60)

measurements = []

cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Map", 1280, 720)

# 슬라이더
def trackbar_callback(x):
    pass

cv2.createTrackbar("NumDisp(x16)", "Depth Map", 8, 16, trackbar_callback)
cv2.createTrackbar("BlockSize", "Depth Map", 11, 25, trackbar_callback)

click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"✓ 클릭: ({x}, {y})")

cv2.setMouseCallback("Depth Map", mouse_callback)

last_measurement = None
frame_count = 0

print("\n프로그램 실행 중...\n")

while True:
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"프레임 {frame_count} 처리 중...")

    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        continue

    # 렉티피케이션
    rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # 그레이스케일 + 히스토그램 균등화
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.equalizeHist(gray_left)
    gray_right = cv2.equalizeHist(gray_right)

    # 슬라이더 값
    num_disp_mult = max(1, cv2.getTrackbarPos("NumDisp(x16)", "Depth Map"))
    num_disp = 16 * num_disp_mult

    block_size = cv2.getTrackbarPos("BlockSize", "Depth Map")
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

    # 디스패리티 계산 (왼쪽 카메라 기준 - OpenCV 표준)
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # 흑백 명암비로 시각화 (가까움=밝게)
    disp_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_visual = np.uint8(disp_visual)

    # 3채널로 변환 (그래픽 오버레이를 위해)
    disp_display = cv2.cvtColor(disp_visual, cv2.COLOR_GRAY2BGR)

    # 클릭 거리 계산
    if click_point is not None:
        x, y = click_point

        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            disp_value = disparity[y, x]

            if disp_value > 0:
                points_3d = cv2.reprojectImageTo3D(disparity, Q)
                depth = points_3d[y, x][2]

                if 0 < depth < 10000:
                    last_measurement = depth
                    # 녹색 원 표시
                    cv2.circle(disp_display, (x, y), 8, (0, 255, 0), 2)
                    # 십자선
                    cv2.line(disp_display, (x-15, y), (x+15, y), (0, 255, 0), 1)
                    cv2.line(disp_display, (x, y-15), (x, y+15), (0, 255, 0), 1)
                    # 거리 표시
                    text = f"{depth:.1f}mm ({depth/10:.1f}cm)"
                    cv2.putText(disp_display, text, (x+15, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.circle(disp_display, (x, y), 8, (0, 0, 255), 2)
                    cv2.putText(disp_display, "OUT OF RANGE", (x+15, y-15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.circle(disp_display, (x, y), 8, (0, 0, 255), 2)
                cv2.putText(disp_display, "NO DISPARITY", (x+15, y-15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        click_point = None

    # 정보 오버레이
    cv2.putText(disp_display, f"NumDisp: {num_disp}", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(disp_display, f"BlockSize: {block_size}", (10, 60),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    if last_measurement is not None:
        cv2.putText(disp_display, f"Last: {last_measurement:.1f}mm ({last_measurement/10:.1f}cm)",
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Depth Map", disp_display)

    # 왼쪽 카메라 원본 (레퍼런스) - 별도 창
    cv2.imshow("Original (Left Camera - Reference)", rect_left)

    key = cv2.waitKey(30) & 0xFF

    if key == ord('s'):
        if last_measurement is not None:
            actual_dist = input("\n실제 거리(mm): ")
            try:
                actual_dist = float(actual_dist)
                error = abs(last_measurement - actual_dist)
                error_pct = (error / actual_dist) * 100
                measurements.append({
                    'measured': last_measurement,
                    'actual': actual_dist,
                    'error': error,
                    'error_pct': error_pct
                })
                print(f"✓ 기록됨 - 측정: {last_measurement:.1f}mm, 실제: {actual_dist:.1f}mm, 오차: {error:.1f}mm ({error_pct:.1f}%)")

                if len(measurements) > 0:
                    print(f"\n현재까지 {len(measurements)}개 측정:")
                    avg_error = np.mean([m['error_pct'] for m in measurements])
                    print(f"평균 오차: {avg_error:.1f}%")
            except:
                print("❌ 숫자를 입력하세요")
        else:
            print("❌ 먼저 거리를 측정하세요!")

    elif key == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

if len(measurements) > 0:
    print("\n" + "="*60)
    print("측정 결과 요약")
    print("="*60)
    for i, m in enumerate(measurements, 1):
        print(f"{i}. 측정={m['measured']:.1f}mm, 실제={m['actual']:.1f}mm, 오차={m['error']:.1f}mm ({m['error_pct']:.1f}%)")
    print(f"\n평균 오차: {np.mean([m['error_pct'] for m in measurements]):.1f}%")
