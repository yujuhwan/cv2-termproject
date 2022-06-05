# 예제 12.2.2 번호판 후보 영역 검색

from plate_preprocess import *                             # 전처리 및 후보 영역 검출 함수 호출

car_no = int(input("자동차 영상 번호 (0~15): "))
image, morph = preprocessing(car_no)                       # 전처리 - 이진화, 소벨&열림 연산
if image is None: Exception("영상 읽기 에러")

candidates = find_candidates(morph)                        # find_candidates() 함수에서 번호판 후보 영역 검색
for candidate in candidates:                               # 후보 영역 표시
    pts = np.int32(cv2.boxPoints(candidate))               # 회전 사각형의 4개 꼭지점 좌표 가져오기(좌표를 그리기 위해 정수형 행렬)
    cv2.polylines(image, [pts], True, (0, 225, 255), 2)    # 다중 좌표 잇기
    print(candidate)

if not candidates:  # 리스트 원소가 없으면
    print("번호판 후보 영역 미검출")
cv2.imshow("image", image)
cv2.waitKey(0)
