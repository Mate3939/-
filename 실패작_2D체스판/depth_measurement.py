"""
스테레오 카메라 실시간 거리 측정
- 캘리브레이션 데이터를 사용하여 깊이 맵 생성
- 마우스 클릭으로 특정 지점의 거리 측정
- 'q' 키: 종료
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

# 클릭한 지점 저장
click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)

def main():
    print("="*50)
    print("스테레오 카메라 거리 측정")
    print("="*50)

    # 캘리브레이션 데이터 로드
    if not os.path.exists(CALIB_FILE):
        print(f"캘리브레이션 파일이 없습니다: {CALIB_FILE}")
        print("먼저 calibration.py를 실행하세요.")
        return

    print("캘리브레이션 데이터 로드 중...")
    calib_data = np.load(CALIB_FILE)

    map1_l = calib_data['map1_l']
    map2_l = calib_data['map2_l']
    map1_r = calib_data['map1_r']
    map2_r = calib_data['map2_r']
    Q = calib_data['Q']

    print("캘리브레이션 데이터 로드 완료!")
    print(f"Q 매트릭스:\n{Q}")

    # 카메라 열기
    print("\n카메라 열기 중...")
    try:
        cap_left = cv2.VideoCapture(LEFT_CAM)
        cap_right = cv2.VideoCapture(RIGHT_CAM)

        print(f"왼쪽 카메라 열림: {cap_left.isOpened()}")
        print(f"오른쪽 카메라 열림: {cap_right.isOpened()}")

        if not cap_left.isOpened() or not cap_right.isOpened():
            print("카메라를 열 수 없습니다!")
            return
    except Exception as e:
        print(f"카메라 열기 에러: {e}")
        return

    # 해상도 설정
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("카메라 준비 완료!")
    print("\n조작법:")
    print("  마우스 클릭 - 해당 지점의 거리 측정")
    print("  'q' - 종료")
    print("="*50)

    # 스테레오 매칭 설정 (SGBM)
    min_disp = 0
    num_disp = 64  # 16의 배수
    block_size = 9

    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
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

    # 마우스 콜백 설정
    cv2.namedWindow("Depth Map")
    cv2.setMouseCallback("Depth Map", mouse_callback)

    global click_point
    measured_distance = None

    while True:
        try:
            ret1, frame_left = cap_left.read()
            ret2, frame_right = cap_right.read()

            if not ret1 or not ret2:
                print("프레임 읽기 실패")
                continue

            # 렉티피케이션 적용
            rect_left = cv2.remap(frame_left, map1_l, map2_l, cv2.INTER_LINEAR)
            rect_right = cv2.remap(frame_right, map1_r, map2_r, cv2.INTER_LINEAR)

            # 그레이스케일 변환
            gray_left = cv2.cvtColor(rect_left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(rect_right, cv2.COLOR_BGR2GRAY)

            # 디스패리티 맵 계산
            disparity = stereo.compute(gray_left, gray_right).astype(np.float32) / 16.0
        except Exception as e:
            print(f"프레임 처리 에러: {e}")
            continue

        # 디스패리티 맵 시각화
        disp_visual = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_visual = np.uint8(disp_visual)
        disp_color = cv2.applyColorMap(disp_visual, cv2.COLORMAP_JET)

        # 3D 좌표 계산
        points_3d = cv2.reprojectImageTo3D(disparity, Q)

        # 클릭한 지점의 거리 계산
        if click_point is not None:
            x, y = click_point
            if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
                if disparity[y, x] > 0:
                    # 3D 좌표에서 거리 계산
                    point_3d = points_3d[y, x]
                    distance = np.sqrt(point_3d[0]**2 + point_3d[1]**2 + point_3d[2]**2)

                    # Z 좌표 (깊이)만 사용할 수도 있음
                    depth = point_3d[2]

                    if 0 < depth < 10000:  # 유효한 범위
                        measured_distance = depth
                        print(f"클릭 위치: ({x}, {y}), 거리: {depth:.1f}mm ({depth/10:.1f}cm)")
                    else:
                        print(f"클릭 위치: ({x}, {y}), 거리 측정 불가 (범위 초과)")
                else:
                    print(f"클릭 위치: ({x}, {y}), 디스패리티 없음")
            click_point = None

        # 측정된 거리 표시
        if measured_distance is not None:
            cv2.putText(disp_color, f"Distance: {measured_distance:.1f}mm ({measured_distance/10:.1f}cm)",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # 렉티피케이션된 이미지 합치기 (수평선 확인용)
        rect_combined = np.hstack([rect_left, rect_right])
        # 수평선 그리기
        for i in range(0, rect_combined.shape[0], 30):
            cv2.line(rect_combined, (0, i), (rect_combined.shape[1], i), (0, 255, 0), 1)

        # 화면 표시
        cv2.imshow("Rectified (Left | Right)", rect_combined)
        cv2.imshow("Depth Map", disp_color)
        cv2.imshow("Left Camera", frame_left)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()
    print("\n종료!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n치명적 에러: {e}")
        import traceback
        traceback.print_exc()
        input("\n엔터를 눌러 종료...")
