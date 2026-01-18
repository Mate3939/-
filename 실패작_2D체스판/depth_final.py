"""
스테레오 거리 측정 - 최종 간단 버전
클릭하면 거리 표시
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 0
RIGHT_CAM = 2
CALIB_FILE = r"C:\Users\mts20\OneDrive\바탕 화면\스테레오 카메라\calibration_data.npz"

print("="*60)
print("스테레오 카메라 거리 측정")
print("="*60)
print("로딩 중... 카메라 켜지는데 3~5분 걸릴 수 있습니다")

# 캘리브레이션 데이터 로드
calib_data = np.load(CALIB_FILE)
map1_l = calib_data['map1_l']
map2_l = calib_data['map2_l']
map1_r = calib_data['map1_r']
map2_r = calib_data['map2_r']
Q = calib_data['Q']

print("캘리브레이션 로드 완료")

# 카메라 열기
print("카메라 열기 중...")
cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("카메라 준비 완료!")
print("\n" + "="*60)
print("사용법:")
print("  1. Depth Map 창의 슬라이더 2개로 조정")
print("  2. 물체를 Depth Map에서 클릭하면 거리 표시")
print("  3. 'q' 키로 종료")
print("="*60 + "\n")

# 윈도우 생성 (크기 지정)
cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Map", 800, 600)

# 슬라이더 생성 (Depth Map 창에 직접!)
def nothing(x):
    pass

cv2.createTrackbar("NumDisp(x16)", "Depth Map", 8, 16, nothing)
cv2.createTrackbar("BlockSize", "Depth Map", 11, 25, nothing)

print("슬라이더 확인: Depth Map 창 위쪽에 2개 슬라이더 있어야 함")

click_point = None
last_depth = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

cv2.setMouseCallback("Depth Map", mouse_callback)

frame_count = 0

while True:
    ret1, frame_left = cap_left.read()
    ret2, frame_right = cap_right.read()

    if not ret1 or not ret2:
        continue

    frame_count += 1

    # 렉티피케이션
    rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
    rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

    # 그레이스케일 + 히스토그램 균등화
    gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)
    gray_left = cv2.equalizeHist(gray_left)
    gray_right = cv2.equalizeHist(gray_right)

    # 슬라이더 값 읽기
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

    # 디스패리티 계산
    disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0

    # 시각화
    disp_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
    disp_visual = np.uint8(disp_visual)
    disp_color = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)

    # 상단 정보 표시
    cv2.putText(disp_color, f"NumDisp:{num_disp} BlockSize:{block_size}",
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    # 마지막 측정 거리 표시
    if last_depth is not None:
        cv2.putText(disp_color, f"Distance: {last_depth:.0f}mm = {last_depth/10:.1f}cm",
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # 클릭 거리 계산
    if click_point is not None:
        x, y = click_point
        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            if disparity[y, x] > 0:
                points_3d = cv2.reprojectImageTo3D(disparity, Q)
                depth = points_3d[y, x][2]

                if 0 < depth < 10000:
                    last_depth = depth

                    # 화면에 표시
                    cv2.circle(disp_color, (x, y), 8, (0, 255, 0), 2)
                    cv2.putText(disp_color, f"{depth:.0f}mm",
                               (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                    # 터미널에 출력
                    print(f"[{frame_count}] 클릭 위치({x},{y}): {depth:.1f}mm = {depth/10:.1f}cm")
                else:
                    print(f"[{frame_count}] 범위 초과: {depth:.1f}mm")
            else:
                print(f"[{frame_count}] 디스패리티 없음 (흰 벽?)")
        click_point = None

    # 안내문
    cv2.putText(disp_color, "Click to measure distance | Press 'q' to quit",
               (10, disp_color.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    cv2.imshow("Depth Map", disp_color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

print("\n종료!")
