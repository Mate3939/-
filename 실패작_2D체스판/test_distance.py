"""
거리 측정 정확도 테스트
알려진 거리의 물체로 테스트
"""

import cv2
import numpy as np
import os

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

LEFT_CAM = 0
RIGHT_CAM = 2
CALIB_FILE = r"C:\Users\mts20\OneDrive\바탕 화면\스테레오 카메라\calibration_data.npz"

# 캘리브레이션 데이터 로드
calib_data = np.load(CALIB_FILE)
map1_l = calib_data['map1_l']
map2_l = calib_data['map2_l']
map1_r = calib_data['map1_r']
map2_r = calib_data['map2_r']
Q = calib_data['Q']

# 카메라 열기
cap_left = cv2.VideoCapture(LEFT_CAM)
cap_right = cv2.VideoCapture(RIGHT_CAM)

cap_left.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_left.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap_right.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap_right.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 줌 리셋
cap_left.set(cv2.CAP_PROP_ZOOM, 100)
cap_right.set(cv2.CAP_PROP_ZOOM, 100)

print("="*50)
print("거리 측정 정확도 테스트")
print("="*50)
print("1. 텍스처 있는 물체(책, 키보드 등)를 카메라 앞에 두세요")
print("2. 자로 실제 거리를 재세요 (카메라~물체)")
print("3. Depth Map에서 물체를 클릭하세요")
print("4. 실제 거리와 측정 거리를 비교하세요")
print("\n's' - 측정값 기록")
print("'q' - 종료")
print("="*50)

measurements = []

cv2.namedWindow("Depth Map", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Depth Map", 800, 600)

# Depth Map 창에 슬라이더 추가
def trackbar_callback(x):
    pass

cv2.createTrackbar("NumDisp(x16)", "Depth Map", 8, 16, trackbar_callback)
cv2.createTrackbar("BlockSize", "Depth Map", 11, 25, trackbar_callback)

print("슬라이더 생성됨: Depth Map 창 위쪽 확인")

click_point = None

def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)
        print(f"✓ 마우스 클릭 감지: ({x}, {y})")

cv2.setMouseCallback("Depth Map", mouse_callback)

# 스테레오 매처는 루프 안에서 슬라이더 값으로 생성

last_measurement = None
frame_count = 0

print("\n프로그램 실행 중...")
print("Depth Map 창을 클릭해서 포커스를 맞춘 후 's' 키를 눌러보세요!\n")

while True:
    frame_count += 1
    if frame_count % 100 == 0:  # 100프레임마다 살아있다는 표시
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

    # 슬라이더 값 읽기
    num_disp_mult = max(1, cv2.getTrackbarPos("NumDisp(x16)", "Depth Map"))
    num_disp = 16 * num_disp_mult

    block_size = cv2.getTrackbarPos("BlockSize", "Depth Map")
    if block_size % 2 == 0:
        block_size += 1
    if block_size < 5:
        block_size = 5

    # 스테레오 매처 생성
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

    # 클릭 거리 계산
    if click_point is not None:
        x, y = click_point
        print(f"  처리 중: 좌표({x}, {y})")

        if 0 <= x < disparity.shape[1] and 0 <= y < disparity.shape[0]:
            disp_value = disparity[y, x]
            print(f"  디스패리티 값: {disp_value:.2f}")

            if disp_value > 0:
                points_3d = cv2.reprojectImageTo3D(disparity, Q)
                depth = points_3d[y, x][2]
                print(f"  깊이 값: {depth:.1f}mm")

                if 0 < depth < 10000:
                    last_measurement = depth
                    cv2.circle(disp_color, (x, y), 8, (0, 255, 0), 2)
                    cv2.putText(disp_color, f"{depth:.0f}mm",
                               (x+15, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    print(f"  ✓ 측정 성공: {depth:.1f}mm ({depth/10:.1f}cm)")
                else:
                    print(f"  ✗ 범위 초과: {depth:.1f}mm (유효 범위: 0~10000mm)")
            else:
                print(f"  ✗ 디스패리티 없음 (흰 벽이거나 텍스처 부족)")
        else:
            print(f"  ✗ 좌표 범위 벗어남")

        click_point = None

    # 파라미터 표시
    cv2.putText(disp_color, f"NumDisp:{num_disp} BlockSize:{block_size}",
               (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 측정값 표시
    if last_measurement is not None:
        cv2.putText(disp_color, f"Measured: {last_measurement:.0f}mm ({last_measurement/10:.1f}cm)",
                   (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(disp_color, "Press 's' to record (Focus this window!)",
                   (10, disp_color.shape[0] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # 기록된 측정 표시
    y_pos = 90
    for i, m in enumerate(measurements[-3:]):  # 최근 3개만
        cv2.putText(disp_color, f"{i+1}: Real={m[0]}cm Measured={m[1]:.1f}cm",
                   (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_pos += 25

    cv2.imshow("Depth Map", disp_color)
    cv2.imshow("Left", rect_left)

    key = cv2.waitKey(30) & 0xFF  # 30ms로 늘림 (키 입력 더 잘 받음)

    # 키 입력 디버깅
    if key != 255:  # 키가 눌렸을 때
        print(f"[프레임 {frame_count}] 키 눌림: '{chr(key)}' (코드: {key})")

    # 's' - 측정값 기록 (화면 정지)
    if key == ord('s'):
        print(f"s 키 눌림! last_measurement={last_measurement}")
        if last_measurement is None:
            print("⚠️ 먼저 물체를 클릭해서 거리를 측정하세요!")
            continue

        # 측정값이 있을 때만 진행
        # 현재 화면 저장 (freeze)
        frozen_frame = disp_color.copy()
        cv2.putText(frozen_frame, "FROZEN - Enter real distance in CMD",
                   (10, frozen_frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Depth Map", frozen_frame)
        cv2.waitKey(1)  # 화면 업데이트

        print("\n" + "="*50)
        print("측정값 기록하기 (화면 정지됨)")
        print("="*50)
        print(f"측정 거리: {last_measurement/10:.1f}cm")
        print("물체를 내려놓으셔도 됩니다.")

        # 입력 받기 (화면은 frozen 상태 유지)
        real_dist = input("실제 거리(cm)를 입력하세요: ")

        try:
            real_cm = float(real_dist)
            measured_cm = last_measurement / 10
            error = abs(real_cm - measured_cm)
            error_pct = (error / real_cm) * 100

            measurements.append((real_cm, measured_cm))

            print(f"\n✓ 기록됨!")
            print(f"  실제 거리: {real_cm:.1f}cm")
            print(f"  측정 거리: {measured_cm:.1f}cm")
            print(f"  오차: {error:.1f}cm ({error_pct:.1f}%)")
        except:
            print("✗ 잘못된 입력")

        print("\n계속 진행합니다...")

    if key == ord('q'):
        break

cap_left.release()
cap_right.release()
cv2.destroyAllWindows()

# 결과 요약
if measurements:
    print("\n" + "="*50)
    print("측정 결과 요약")
    print("="*50)
    for i, (real, measured) in enumerate(measurements, 1):
        error = abs(real - measured)
        error_pct = (error / real) * 100
        print(f"{i}. 실제={real:.1f}cm, 측정={measured:.1f}cm, 오차={error:.1f}cm ({error_pct:.1f}%)")

    avg_error_pct = np.mean([abs(r-m)/r*100 for r, m in measurements])
    print(f"\n평균 오차율: {avg_error_pct:.1f}%")

    if avg_error_pct > 10:
        print("\n⚠️ 오차가 10% 이상입니다!")
        print("해결 방법:")
        print("1. 체스보드 SQUARE_SIZE 확인 (태블릿에서 자로 측정)")
        print("2. 캘리브레이션 다시 하기 (더 다양한 각도로)")
