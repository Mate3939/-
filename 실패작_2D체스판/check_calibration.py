"""
캘리브레이션 데이터 검증
"""

import numpy as np
import cv2

CALIB_FILE = r"C:\Users\mts20\OneDrive\바탕 화면\스테레오 카메라\calibration_data.npz"

print("="*50)
print("캘리브레이션 데이터 검증")
print("="*50)

# 데이터 로드
data = np.load(CALIB_FILE)

print("\n저장된 데이터 키:")
for key in data.keys():
    print(f"  - {key}")

print("\n카메라 매트릭스 (왼쪽):")
print(data['mtx_l'])

print("\n카메라 매트릭스 (오른쪽):")
print(data['mtx_r'])

print("\n왜곡 계수 (왼쪽):")
print(data['dist_l'])

print("\n왜곡 계수 (오른쪽):")
print(data['dist_r'])

print("\n회전 행렬 R:")
print(data['R'])

print("\n변환 벡터 T (카메라 간 거리):")
print(data['T'])
baseline = abs(data['T'][0][0])
print(f"Baseline: {baseline:.2f}mm")

print("\nQ 매트릭스 (3D 재투영용):")
print(data['Q'])

# Q 매트릭스에서 focal length 확인
Q = data['Q']
focal_length = Q[2, 3]
print(f"\nFocal Length (from Q): {abs(focal_length):.2f}")

# Rectification 체크
print("\nP1 (왼쪽 투영 매트릭스):")
print(data['P1'])

print("\nP2 (오른쪽 투영 매트릭스):")
print(data['P2'])

print("\n" + "="*50)
print("검증 결과:")
print("="*50)

# 검증
issues = []

if baseline < 50 or baseline > 300:
    issues.append(f"⚠️ Baseline이 이상함: {baseline:.2f}mm (정상 범위: 50-300mm)")

if abs(focal_length) < 100 or abs(focal_length) > 2000:
    issues.append(f"⚠️ Focal length가 이상함: {abs(focal_length):.2f} (정상 범위: 100-2000)")

# 왜곡 계수 확인
dist_l_max = np.max(np.abs(data['dist_l']))
dist_r_max = np.max(np.abs(data['dist_r']))

if dist_l_max > 2 or dist_r_max > 2:
    issues.append(f"⚠️ 왜곡 계수가 너무 큼: L={dist_l_max:.2f}, R={dist_r_max:.2f}")

if len(issues) == 0:
    print("✅ 캘리브레이션 데이터가 정상 범위 내에 있습니다.")
else:
    print("문제 발견:")
    for issue in issues:
        print(issue)

print("\n실제 카메라 간 거리를 자로 재서 확인해주세요!")
print(f"현재 저장된 Baseline: {baseline:.2f}mm ({baseline/10:.2f}cm)")
print("\n만약 실제 거리와 많이 다르면 캘리브레이션을 다시 해야 합니다.")

input("\n엔터를 눌러 종료...")
