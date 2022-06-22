# 예제 12.2.3 번호판 영상 생성

from plate_preprocess import *        # 전처리 및 후보 영역 검출 함수
from plate_candidate import *         # 후보 영역 개선 및 후보 영상 생성 함수
import cv2

car_no = 0
image, morph = preprocessing(car_no)  # 전처리 - 이진화
candidates = find_candidates(morph)  # 번호판 후보 영역 검색

fills = [color_candidate_img(image, size) for size, _, _ in candidates]     # 후보 영역 재생성
new_candis = [find_candidates(fill) for fill in fills]                      # 재생성 영역 검사
new_candis = [cand[0] for cand in new_candis if cand]                       # 재후보 있으면 저장
candidate_imgs = [rotate_plate(image, cand) for cand in new_candis]         # 후보 영역 번호판 영상

for i, img in enumerate(candidate_imgs):        # 후보 영상 표시
    cv2.imshow("candidate_img - " + str(i), img)
    cv2.polylines(image, [np.int32(cv2.boxPoints(new_candis[i]))], True, (0, 255, 0), 2)

cv2.imshow("image", image)
cv2.waitKey()