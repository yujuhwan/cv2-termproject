# 예제 12.2.4 번호판 인식 프로그램 완성

from plate_preprocess import *        # 전처리 및 후보 영역 검출 함수
from plate_candidate import *         # 후보 영역 개선 및 후보 영상 생성 함수
import cv2

car_no = int(input("자동차 영상 번호 (0~15): "))
image, morph = preprocessing(car_no)                               # 전처리
candidates = find_candidates(morph)                                # 번호판 후보 영역 검색

fills = [color_candidate_img(image, size) for size, _, _ in candidates]
new_candis = [find_candidates(fill) for fill in fills]
new_candis = [cand[0] for cand in new_candis if cand]
candidate_imgs = [rotate_plate(image, cand) for cand in new_candis]

svm = cv2.ml.SVM_load("svmtrain3.xml")                          # 학습된 데이터 로드
rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))    # 1행 데이터들로 변환
_, results = svm.predict(rows.astype("float32"))                # 분류 수행
correct = np.where(results == 1)[0]                             # 1인 값의 위치 찾기

print('분류 결과:\n', results)
print('번호판 영상 인덱스:', correct )

for i, idx in enumerate(correct):                               # 후보 영역 중 번호판으로 분류된 영역의 인덱스(idx) 후보 영역 영상
    cv2.imshow("plate_" +str(i), candidate_imgs[idx])
    # cv2.resizeWindow("plate image_" + str(i), (250,28))

for i, candi in enumerate(new_candis):                          # 모든 후보 영상을 순회
    color = (0, 255, 0) if i in correct else (0, 0, 255)            # 번호판은 녹색(0,255,0), 아닌 것 빨간색(0,0,255)
    cv2.polylines(image, [np.int32(cv2.boxPoints(candi))], True, color, 2)      # 입력영상의 사각형으로 표시

print("번호판 검출완료") if len(correct)>0 else print("번호판 미검출")

cv2.imshow("image", image)
cv2.waitKey(0)