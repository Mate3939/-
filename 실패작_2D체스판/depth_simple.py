"""
스테레오 깊이 측정 - 간단 버전
Depth Map 창에 트랙바 직접 부착
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 0
RIGHT_CAM = 2
CALIB_FILE = r"C:\Users\mts20\OneDrive\바탕 화면\스테레오 카메라\calibration_data.npz"

def nothing(x):
    pass

# 캘리브레이션 데이터 로드
print("캘리브레이션 데이터 로드 중...")
calib_data = np.load(CALIB_FILE)
map1_l = calib_data['map1_l']
map2_l = calib_data['map2_l']
map1_r = calib_data['map1_r']
map2_r = calib_data['map2_r']
Q = calib_data['Q']

# 카메라 열기
print("카메라 열기 중...")
cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("준비 완료!")
print("NumDisp, BlockSize 슬라이더로 조정하세요")
print("'q' 키로 종료\n")

# 윈도우 생성
cv2.namedWindow("Depth Map")

# Depth Map 창에 트랙바 직접 부착!
cv2.createTrackbar("NumDisp", "Depth Map", 5, 16, nothing)  # x16
cv2.createTrackbar("BlockSize", "Depth Map", 9, 21, nothing)

click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

cv2.setMouseCallback("Depth Map", mouse_callback)

while True:
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

    # 트랙바 값 읽기
    num_disp_mult = max(1, cv2.getTrackbarPos("NumDisp", "Depth Map"))
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
        uniquenessRatio=10,
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

    # 파라미터 표시
    cv2.putText(disp_color, f"NumDisp:{num_disp} BlockSize:{block_size}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # 클릭 거리 계산
    if click_point is not None:
        x, y = click_point
        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            if disparity[y, x] > 0:
                points_3d = cv2.reprojectImageTo3D(disparity, Q)
                depth = points_3d[y, x][2]
                if 0 < depth < 10000:
                    cv2.putText(disp_color, f"{depth:.0f}mm ({depth/10:.1f}cm)",
                               (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.circle(disp_color, (x, y), 5, (0, 255, 0), -1)
                    print(f"거리: {depth:.1f}mm ({depth/10:.1f}cm)")

    cv2.imshow("Depth Map", disp_color)
    cv2.imshow("Left", gray_left)
    cv2.imshow("Right", gray_right)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()
