"""
Alpha 값 비교 테스트
"""
import cv2
import numpy as np

# 캘리브레이션 데이터 로드
calib_data = np.load(r"C:\Users\mts20\Desktop\자율주행\calibration_data.npz")

mtx_l = calib_data['mtx_l']
dist_l = calib_data['dist_l']
mtx_r = calib_data['mtx_r']
dist_r = calib_data['dist_r']
R = calib_data['R']
T = calib_data['T']

img_size = (1280, 720)

print("="*60)
print("Alpha 값별 비교")
print("="*60)

for alpha_val in [0, 0.5, 1.0]:
    print(f"\nalpha = {alpha_val}")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, img_size, R, T,
        alpha=alpha_val, flags=cv2.CALIB_ZERO_DISPARITY
    )

    print(f"  ROI Left: {roi1}")
    print(f"  ROI Right: {roi2}")
    print(f"  Q[2,3] (focal): {Q[2,3]:.2f}")

print("\n" + "="*60)
print("설명:")
print("  alpha = 0.0: FOV 좁음, 검은 영역 없음")
print("  alpha = 0.5: 중간")
print("  alpha = 1.0: FOV 넓음, 검은 영역 있을 수 있음 (현재 설정)")
print("="*60)
