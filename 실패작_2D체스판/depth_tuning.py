"""
스테레오 깊이 측정 파라미터 튜닝
- 트랙바로 실시간 파라미터 조정
- 최적 파라미터 찾기
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# 카메라 설정
LEFT_CAM = 0
RIGHT_CAM = 2

# 캘리브레이션 데이터 경로
CALIB_FILE = r"C:\Users\mts20\OneDrive\바탕 화면\스테레오 카메라\calibration_data.npz"

# 전역 변수
min_disp = 0
num_disp = 16 * 5  # 80
block_size = 5

def nothing(x):
    pass

def main():
    global min_disp, num_disp, block_size

    print("="*50)
    print("스테레오 깊이 측정 파라미터 튜닝")
    print("="*50)

    # 캘리브레이션 데이터 로드
    if not os.path.exists(CALIB_FILE):
        print(f"캘리브레이션 파일이 없습니다: {CALIB_FILE}")
        return

    print("캘리브레이션 데이터 로드 중...")
    calib_data = np.load(CALIB_FILE)

    map1_l = calib_data['map1_l']
    map2_l = calib_data['map2_l']
    map1_r = calib_data['map1_r']
    map2_r = calib_data['map2_r']
    Q = calib_data['Q']

    print("캘리브레이션 데이터 로드 완료!")

    # 카메라 열기
    print("\n카메라 열기 중...")
    cap_left = cv2.VideoCapture(LEFT_CAM)
    cap_right = cv2.VideoCapture(RIGHT_CAM)

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("카메라를 열 수 없습니다!")
        return

    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("카메라 준비 완료!")
    print("\n조작법:")
    print("  트랙바로 파라미터 조정")
    print("  's' - 현재 파라미터 출력")
    print("  'q' - 종료")
    print("="*50)

    # 트랙바 윈도우 생성
    cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Controls", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Controls", 400, 600)

    # 빈 이미지로 Controls 창 초기화
    controls_img = np.zeros((600, 400, 3), dtype=np.uint8)
    cv2.imshow("Controls", controls_img)

    print("\nControls 창이 보이나요? (트랙바가 있는 창)")

    # 트랙바 생성
    cv2.createTrackbar("Min Disp", "Controls", 0, 50, nothing)
    cv2.createTrackbar("Num Disp x16", "Controls", 5, 16, nothing)  # 16~256
    cv2.createTrackbar("Block Size", "Controls", 5, 21, nothing)
    cv2.createTrackbar("P1 Mult", "Controls", 8, 50, nothing)
    cv2.createTrackbar("P2 Mult", "Controls", 32, 200, nothing)
    cv2.createTrackbar("Uniqueness", "Controls", 10, 50, nothing)
    cv2.createTrackbar("Speckle Win", "Controls", 100, 300, nothing)
    cv2.createTrackbar("Speckle Range", "Controls", 32, 100, nothing)
    cv2.createTrackbar("PreFilter", "Controls", 63, 255, nothing)

    click_point = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal click_point
        if event == cv2.EVENT_LBUTTONDOWN:
            click_point = (x, y)

    cv2.setMouseCallback("Depth Map", mouse_callback)

    while True:
        try:
            ret1, frame_left = cap_left.read()
            ret2, frame_right = cap_right.read()

            if not ret1 or not ret2:
                continue

            # 렉티피케이션
            rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

            # 히스토그램 균등화 (대비 향상)
            rect_left_gray = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            rect_right_gray = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

            rect_left_gray = cv2.equalizeHist(rect_left_gray)
            rect_right_gray = cv2.equalizeHist(rect_right_gray)

            # 트랙바 값 읽기
            min_disp = cv2.getTrackbarPos("Min Disp", "Controls")
            num_disp_mult = max(1, cv2.getTrackbarPos("Num Disp x16", "Controls"))
            num_disp = 16 * num_disp_mult

            block_size = cv2.getTrackbarPos("Block Size", "Controls")
            if block_size % 2 == 0:
                block_size += 1
            if block_size < 5:
                block_size = 5

            p1_mult = cv2.getTrackbarPos("P1 Mult", "Controls")
            p2_mult = cv2.getTrackbarPos("P2 Mult", "Controls")
            uniqueness = cv2.getTrackbarPos("Uniqueness", "Controls")
            speckle_win = cv2.getTrackbarPos("Speckle Win", "Controls")
            speckle_range = cv2.getTrackbarPos("Speckle Range", "Controls")
            prefilter = cv2.getTrackbarPos("PreFilter", "Controls")

            # 스테레오 매처 생성
            stereo = cv2.StereoSGBM_create(
                minDisparity=min_disp,
                numDisparities=num_disp,
                blockSize=block_size,
                P1=p1_mult * block_size ** 2,
                P2=p2_mult * block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=uniqueness,
                speckleWindowSize=speckle_win,
                speckleRange=speckle_range,
                preFilterCap=prefilter,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )

            # 디스패리티 계산
            disparity = stereo.compute(rect_left_gray, rect_right_gray).astype(np.float32) / 16.0

            # WLS 필터 (옵션 - 품질 향상)
            # right_matcher = cv2.ximgproc.createRightMatcher(stereo)
            # disparity_right = right_matcher.compute(rect_right_gray, rect_left_gray)
            # wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo)
            # disparity = wls_filter.filter(disparity, rect_left_gray, disparity_right=disparity_right)

            # 시각화
            disp_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
            disp_visual = np.uint8(disp_visual)
            disp_color = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)

            # 클릭 위치 거리 계산
            if click_point is not None:
                x, y = click_point
                if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
                    if disparity[y, x] > 0:
                        points_3d = cv2.reprojectImageTo3D(disparity, Q)
                        point_3d = points_3d[y, x]
                        depth = point_3d[2]

                        if 0 < depth < 10000:
                            cv2.putText(disp_color, f"{depth:.1f}mm ({depth/10:.1f}cm)",
                                       (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.circle(disp_color, (x, y), 5, (0, 255, 0), -1)
                            print(f"({x}, {y}): {depth:.1f}mm ({depth/10:.1f}cm)")

            # 화면 표시
            cv2.imshow("Depth Map", disp_color)
            cv2.imshow("Left (Equalized)", rect_left_gray)
            cv2.imshow("Right (Equalized)", rect_right_gray)

            # 빈 이미지 (컨트롤 윈도우용)
            controls_img = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(controls_img, "Adjust trackbars", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            cv2.putText(controls_img, f"Min Disp: {min_disp}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(controls_img, f"Num Disp: {num_disp}", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(controls_img, f"Block Size: {block_size}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Controls", controls_img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                print("\n현재 파라미터:")
                print(f"  minDisparity={min_disp}")
                print(f"  numDisparities={num_disp}")
                print(f"  blockSize={block_size}")
                print(f"  P1={p1_mult * block_size ** 2}")
                print(f"  P2={p2_mult * block_size ** 2}")
                print(f"  uniquenessRatio={uniqueness}")
                print(f"  speckleWindowSize={speckle_win}")
                print(f"  speckleRange={speckle_range}")
                print(f"  preFilterCap={prefilter}")

            if key == ord('q'):
                break

        except Exception as e:
            print(f"에러: {e}")
            continue

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"치명적 에러: {e}")
        import traceback
        traceback.print_exc()
