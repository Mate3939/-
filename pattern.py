"""
체스보드 패턴 생성 (인쇄용)
A4 용지에 맞는 크기
"""

import cv2
import numpy as np

# 체스보드 설정
SQUARES_X = 8  # 가로 사각형 개수
SQUARES_Y = 6  # 세로 사각형 개수
SQUARE_SIZE = 300  # 픽셀 단위 사각형 크기 (인쇄 후 측정 필요)

# 이미지 크기 계산
width = SQUARES_X * SQUARE_SIZE
height = SQUARES_Y * SQUARE_SIZE

# 체스보드 생성
chessboard = np.zeros((height, width), dtype=np.uint8)

for i in range(SQUARES_Y):
    for j in range(SQUARES_X):
        if (i + j) % 2 == 0:
            y1 = i * SQUARE_SIZE
            y2 = (i + 1) * SQUARE_SIZE
            x1 = j * SQUARE_SIZE
            x2 = (j + 1) * SQUARE_SIZE
            chessboard[y1:y2, x1:x2] = 255

# 저장
cv2.imwrite("chessboard_print.png", chessboard)

print("✅ 체스보드 생성 완료: chessboard_print.png")
print(f"크기: {width}x{height} 픽셀")
print(f"사각형: {SQUARES_X}x{SQUARES_Y}")
print(f"내부 코너: {SQUARES_X-1}x{SQUARES_Y-1} = {(SQUARES_X-1)}x{(SQUARES_Y-1)}")
print()
print("[ 인쇄 방법 ]")
print("1. 이 이미지를 A4 용지에 인쇄하세요")
print("2. 자로 사각형 한 칸의 크기를 측정하세요 (mm)")
print("3. calibration_v2.py의 SQUARE_SIZE를 수정하세요")
print()
print("예: 인쇄 후 한 칸이 25mm라면 → SQUARE_SIZE = 25")
