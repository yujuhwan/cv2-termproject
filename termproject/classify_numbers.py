# 예제 12.3.1 번호판 숫자 인식 프로그램 완성

from plate_preprocess import *        # 전처리 및 후보 영역 검출 함수
from plate_candidate import *         # 후보 영역 개선 및 후보 영상 생성 함수
from plate_classify import *  # k-NN 학습 및 분류

car_no = int(input("자동차 영상 번호 (0~20): "))
image, morph = preprocessing(car_no)                                    # 전처리
candidates = find_candidates(morph)                            # 후보 영역 검색

fills = [color_candidate_img(image, size) for size, _, _ in candidates]  # 후보 영역 재생성
new_candis = [find_candidates(fill) for fill in fills]                   # 재생성 영역 검사
new_candis = [cand[0] for cand in new_candis if cand]                    # 재후보 있으면 저장
candidate_imgs = [rotate_plate(image, cand) for cand in new_candis]      # 후보 영역 번호판 영상

svm = cv2.ml.SVM_load("SVMTrain.xml")                  # 학습된 데이터 로드
rows = np.reshape(candidate_imgs, (len(candidate_imgs), -1))    # 1행 데이터들로 변환
_, results = svm.predict(rows.astype("float32"))                # 분류 수행
result = np.where(results == 1)[0]        # 1인 값의 위치 찾기

plate_no = result[0] if len(result)>0 else -1            # 번호판 판정

# k-NN 학습
K1, K2 = 10, 10
nknn = kNN_train("images/train_numbers.png", K1, 10, 20) # 숫자 학습
tknn = kNN_train("images/train_texts.png", K2, 40, 20)   # 문자 학습


if plate_no >= 0:
    plate_img = preprocessing_plate(candidate_imgs[plate_no])   # 번호판 영상 전처리
    cells_roi = find_objects(cv2.bitwise_not(plate_img))        # 번호판 영상에서 숫자(문자) 영역 검출
    cells = [plate_img[y:y+h, x:x+w] for x,y,w,h in cells_roi]  # 셀(숫자/문자) 영상 생성

    classify_numbers(cells, nknn, tknn, K1, K2, cells_roi)      # 숫자 객체 분류

    pts = np.int32(cv2.boxPoints(new_candis[plate_no]))         # 번호판 영역 좌표(정수형)으로 가져와 좌표점
    cv2.polylines(image, [pts], True,  (0, 255, 0), 2)          # 직선을 입력 영상에 그림

    color_plate = cv2.cvtColor(plate_img, cv2.COLOR_GRAY2BGR)  # 컬러 번호판 영상
    for x,y, w, h in cells_roi:
        cv2.rectangle(color_plate, (x,y), (x+w,y+h), (0, 0, 255), 1)        # 번호판에 사각형 그리기

    h,w  = color_plate.shape[:2]
    image[0:h, 0:w] = color_plate        # 번호판 원본 영상에 복사
else:
    print("번호판 미검출")

cv2.imshow("image", image)
cv2.waitKey(0)