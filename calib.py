"""
스테레오 카메라 캘리브레이션 v2
- HD 해상도 (1280x720)
- alpha=1 (전체 FOV 유지)
- 10~20장 권장
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 2
RIGHT_CAM = 1  # USB 허브 사용, 좌우 확인됨
CHESSBOARD_SIZE = (7, 5)  # 내부 코너 개수 (가로, 세로)
SQUARE_SIZE = 23.5  # 실제 사각형 크기 (mm) - 측정 확인됨

# 캘리브레이션 데이터 저장 경로
CALIB_FILE = r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz"

print("="*60)
print("스테레오 카메라 캘리브레이션 v2")
print("="*60)
print(f"체스보드 크기: {CHESSBOARD_SIZE[0]}x{CHESSBOARD_SIZE[1]} (내부 코너)")
print(f"사각형 크기: {SQUARE_SIZE}mm")
print(f"해상도: 1280x720 (HD)")
print(f"권장 촬영 개수: 10~20장 (고품질)")
print("="*60)

# 카메라 열기
cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

if not cap_left.isOpened() or not cap_right.isOpened():
    print("❌ 카메라를 열 수 없습니다!")
    exit(1)

# HD 해상도 설정
cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print(f"\n왼쪽 카메라: {int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"오른쪽 카메라: {int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

# 3D 포인트 준비
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

# 저장할 포인트들
objpoints = []  # 3D 포인트
imgpoints_l = []  # 왼쪽 이미지 포인트
imgpoints_r = []  # 오른쪽 이미지 포인트

img_size = None
capture_count = 0

def main():
    global img_size, capture_count

    # 서브픽셀 정확도를 위한 criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("\n체스보드를 카메라에 비춰주세요...")
    print("'s' - 사진 촬영 (10~20장 권장)")
    print("'c' - 캘리브레이션 시작")
    print("'q' - 종료")

    while True:
        ret1, frame_left = cap_left.read()
        ret2, frame_right = cap_right.read()

        if not ret1 or not ret2:
            continue

        if img_size is None:
            img_size = (frame_left.shape[1], frame_left.shape[0])

        # 그레이스케일 변환
        gray_left = cv2.cvtColor(frame_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(frame_right, cv2.COLOR_BGR2GRAY)

        # 체스보드 코너 찾기
        flags = cv2.CALIB_CB_FAST_CHECK
        found_left, corners_left = cv2.findChessboardCorners(gray_left, CHESSBOARD_SIZE, flags)
        found_right, corners_right = cv2.findChessboardCorners(gray_right, CHESSBOARD_SIZE, flags)

        # 화면에 표시
        display_left = frame_left.copy()
        display_right = frame_right.copy()

        if found_left:
            cv2.drawChessboardCorners(display_left, CHESSBOARD_SIZE, corners_left, found_left)
            cv2.putText(display_left, "LEFT: OK", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_left, "LEFT: NOT FOUND", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if found_right:
            cv2.drawChessboardCorners(display_right, CHESSBOARD_SIZE, corners_right, found_right)
            cv2.putText(display_right, "RIGHT: OK", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(display_right, "RIGHT: NOT FOUND", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.putText(display_left, f"Count: {capture_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        cv2.putText(display_right, f"Count: {capture_count}", (20, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # 화면 크기 조정 (HD는 크니까 50% 크기로 표시)
        display_left = cv2.resize(display_left, (640, 360))
        display_right = cv2.resize(display_right, (640, 360))

        combined = np.hstack([display_left, display_right])
        cv2.imshow("Calibration", combined)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('s'):
            if found_left and found_right:
                # 서브픽셀 정확도 향상
                corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11),
                                                        (-1, -1), criteria)
                corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11),
                                                         (-1, -1), criteria)

                objpoints.append(objp)
                imgpoints_l.append(corners_left_refined)
                imgpoints_r.append(corners_right_refined)
                capture_count += 1
                print(f"✓ 캡처 {capture_count}장 완료!")
            else:
                print("❌ 양쪽 카메라에서 모두 체스보드를 찾아야 합니다!")

        elif key == ord('c'):
            if capture_count >= 10:
                print(f"\n캘리브레이션 시작... ({capture_count}장)")
                calibrate()
                break
            else:
                print(f"❌ 최소 10장 이상 필요합니다! (현재: {capture_count}장)")

        elif key == ord('q'):
            print("종료합니다.")
            break

    cap_left.release()
    cap_right.release()
    cv2.destroyAllWindows()

def calibrate():
    print("\n" + "="*60)
    print("개별 카메라 캘리브레이션 중...")
    print("="*60)

    # 왼쪽 카메라
    ret_l, mtx_l, dist_l, rvecs_l, tvecs_l = cv2.calibrateCamera(
        objpoints, imgpoints_l, img_size, None, None
    )
    print(f"왼쪽 카메라 RMS 오차: {ret_l:.4f}")

    # 오른쪽 카메라
    ret_r, mtx_r, dist_r, rvecs_r, tvecs_r = cv2.calibrateCamera(
        objpoints, imgpoints_r, img_size, None, None
    )
    print(f"오른쪽 카메라 RMS 오차: {ret_r:.4f}")

    print("\n스테레오 캘리브레이션 중...")

    # 스테레오 캘리브레이션
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria_stereo = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)

    ret_stereo, mtx_l, dist_l, mtx_r, dist_r, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r,
        mtx_l, dist_l, mtx_r, dist_r,
        img_size, criteria=criteria_stereo, flags=flags
    )

    print(f"스테레오 RMS 오차: {ret_stereo:.4f}")

    print("\n스테레오 정합(Rectification) 계산 중...")

    # 스테레오 정합 - alpha=1로 전체 FOV 유지
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T,
        alpha=1, flags=cv2.CALIB_ZERO_DISPARITY
    )

    # Rectification 맵 생성
    map1_l, map2_l = cv2.initUndistortRectifyMap(mtx_l, dist_l, R1, P1, img_size, cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(mtx_r, dist_r, R2, P2, img_size, cv2.CV_32FC1)

    # 결과 저장
    np.savez(CALIB_FILE,
             mtx_l=mtx_l, dist_l=dist_l,
             mtx_r=mtx_r, dist_r=dist_r,
             R=R, T=T, E=E, F=F,
             R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
             map1_l=map1_l, map2_l=map2_l,
             map1_r=map1_r, map2_r=map2_r,
             roi1=roi1, roi2=roi2)

    print(f"\n✅ 캘리브레이션 완료!")
    print(f"저장 위치: {CALIB_FILE}")

    # 주요 정보 출력
    baseline = np.linalg.norm(T)
    focal_length = Q[2, 3]

    print("\n주요 파라미터:")
    print(f"  Baseline: {baseline:.2f}mm")
    print(f"  Focal Length: {focal_length:.2f}")
    print(f"  alpha: 1 (전체 FOV 유지)")

    print("\nQ 매트릭스:")
    print(Q)

if __name__ == "__main__":
    main()
