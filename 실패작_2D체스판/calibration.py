"""
스테레오 카메라 캘리브레이션
- 체스보드 패턴을 사용하여 카메라 파라미터 추출
- 's' 키: 현재 프레임 캡처
- 'c' 키: 캘리브레이션 실행
- 'q' 키: 종료
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

# 카메라 설정
LEFT_CAM = 0
RIGHT_CAM = 2

# 체스보드 설정 (내부 코너 개수)
CHESSBOARD_SIZE = (7, 5)  # 가로 7, 세로 5 내부 코너 (더 작아서 인식 쉬움)
SQUARE_SIZE = 23.5  # 실제 사각형 크기 (mm) - 태블릿에서 측정함

# 저장 폴더
SAVE_DIR = r"C:\Users\mts20\OneDrive\바탕 화면\스테레오 카메라\calibration_images"
os.makedirs(SAVE_DIR, exist_ok=True)

def main():
    print("="*50)
    print("스테레오 카메라 캘리브레이션")
    print("="*50)
    print(f"왼쪽 카메라: {LEFT_CAM}, 오른쪽 카메라: {RIGHT_CAM}")
    print(f"체스보드 내부 코너: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]}")
    print()
    print("조작법:")
    print("  's' - 현재 프레임 캡처 (최소 10장 필요)")
    print("  'c' - 캘리브레이션 실행")
    print("  'q' - 종료")
    print("="*50)

    # 카메라 열기
    cap_left = cv2.VideoCapture(LEFT_CAM)
    cap_right = cv2.VideoCapture(RIGHT_CAM)

    print(f"왼쪽 카메라 열림: {cap_left.isOpened()}")
    print(f"오른쪽 카메라 열림: {cap_right.isOpened()}")

    if not cap_left.isOpened() or not cap_right.isOpened():
        print("카메라를 열 수 없습니다!")
        return

    print("카메라 창을 여는 중...")

    # 해상도 설정 (HD급)
    cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # 실제 해상도 확인
    import time
    time.sleep(1)  # 카메라 초기화 대기
    ret1, test_frame1 = cap_left.read()
    ret2, test_frame2 = cap_right.read()
    if ret1 and ret2:
        print(f"실제 왼쪽 해상도: {test_frame1.shape[1]}x{test_frame1.shape[0]}")
        print(f"실제 오른쪽 해상도: {test_frame2.shape[1]}x{test_frame2.shape[0]}")

    # 캡처된 이미지 저장
    img_counter = 0
    captured_left = []
    captured_right = []

    # 체스보드 코너 검출 기준
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n체스보드를 카메라에 비춰주세요...")

    while True:
        ret1, frame_left = cap_left.read()
        ret2, frame_right = cap_right.read()

        if not ret1 or not ret2:
            continue

        # 그레이스케일 변환
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 찾기 (빠른 검색 모드)
        flags = cv2.CALIB_CB_FAST_CHECK
        found_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, flags)
        found_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, flags)

        # 디스플레이용 복사본
        display_left = frame_left.copy()
        display_right = frame_right.copy()

        # 코너 그리기
        if found_left:
            cv2.drawChessboardCorners(display_left, CHESSBOARD_SIZE, corners_left, found_left)
            cv2.putText(display_left, "DETECTED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_left, "No chessboard", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if found_right:
            cv2.drawChessboardCorners(display_right, CHESSBOARD_SIZE, corners_right, found_right)
            cv2.putText(display_right, "DETECTED!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_right, "No chessboard", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 캡처 카운트 표시
        cv2.putText(display_left, f"LEFT - Captured: {img_counter}", (20, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_right, f"RIGHT - Captured: {img_counter}", (20, 460),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 양쪽 모두 감지되면 초록색 테두리
        if found_left and found_right:
            cv2.rectangle(display_left, (0, 0), (639, 479), (0, 255, 0), 5)
            cv2.rectangle(display_right, (0, 0), (639, 479), (0, 255, 0), 5)

        # 이미지 합쳐서 표시
        combined = np.hstack([display_left, display_right])
        cv2.imshow("Stereo Calibration (Press 's' to capture)", combined)

        key = cv2.waitKey(1) & 0xFF

        # 's' - 캡처
        if key == ord('s'):
            if found_left and found_right:
                # 이미지 저장
                left_path = os.path.join(SAVE_DIR, f"left_{img_counter:02d}.png")
                right_path = os.path.join(SAVE_DIR, f"right_{img_counter:02d}.png")

                cv2.imwrite(left_path, frame_left)
                cv2.imwrite(right_path, frame_right)

                captured_left.append((gray_left.copy(), corners_left))
                captured_right.append((gray_right.copy(), corners_right))

                img_counter += 1
                print(f"캡처 {img_counter}장 완료! 저장: {left_path}")
            else:
                print(f"양쪽 카메라 모두에서 체스보드가 감지되어야 합니다! (왼쪽:{found_left}, 오른쪽:{found_right})")

        # 'c' - 캘리브레이션 실행
        if key == ord('c'):
            if img_counter < 10:
                print(f"최소 10장 필요합니다. 현재 {img_counter}장")
            else:
                print("\n캘리브레이션 진행 중...")
                run_calibration(captured_left, captured_right, gray_left.shape[::-1])

        # 'q' - 종료
        if key == ord('q'):
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()


def run_calibration(captured_left, captured_right, img_size):
    """스테레오 캘리브레이션 실행"""

    # 3D 점 준비 (체스보드의 실제 좌표)
    objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    obj_points = []  # 3D 점
    img_points_left = []  # 왼쪽 2D 점
    img_points_right = []  # 오른쪽 2D 점

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for (gray_l, corners_l), (gray_r, corners_r) in zip(captured_left, captured_right):
        # 서브픽셀 정밀도로 코너 개선
        corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
        corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

        obj_points.append(objp)
        img_points_left.append(corners_l)
        img_points_right.append(corners_r)

    print("개별 카메라 캘리브레이션 중...")

    # 왼쪽 카메라 캘리브레이션
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        obj_points, img_points_left, img_size, None, None
    )
    print(f"왼쪽 카메라 RMS 오차: {ret_l:.4f}")

    # 오른쪽 카메라 캘리브레이션
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        obj_points, img_points_right, img_size, None, None
    )
    print(f"오른쪽 카메라 RMS 오차: {ret_r:.4f}")

    print("\n스테레오 캘리브레이션 중...")

    # 스테레오 캘리브레이션
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        obj_points, img_points_left, img_points_right,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, criteria=criteria_stereo, flags=flags
    )
    print(f"스테레오 RMS 오차: {ret_stereo:.4f}")

    print("\n스테레오 정합(Rectification) 계산 중...")

    # 스테레오 정합
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T,
        alpha=1, flags=cv2.CALIB_ZERO_DISPARITY
    )

    # Rectification 맵 생성
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

    # 결과 저장
    save_path = r"C:\Users\mts20\OneDrive\바탕 화면\스테레오 카메라\calibration_data.npz"
    np.savez(save_path,
             mtx_l=mtx_l, dist_l=dist_l,
             mtx_r=mtx_r, dist_r=dist_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1_l=map1_l, map2_l=map2_l,
             map1_r=map1_r, map2_r=map2_r,
             roi1=roi1, roi2=roi2)

    print(f"\n캘리브레이션 완료!")
    print(f"저장 위치: {save_path}")
    print(f"\n카메라 간 거리 (Baseline): {abs(T[0][0]):.2f}mm")
    print("\n이제 depth_measurement.py를 실행하여 거리 측정을 할 수 있습니다!")


if __name__ == "__main__":
    main()
