import numpy as np, cv2
from Common.knn import find_number, place_middle            # k-NN 관련 함수

# 숫자 및 문자 영상 학습
def kNN_train(train_fname, K, nclass, nsample):
    size = (40, 40)  # 숫자 영상 크기
    train_img = cv2.imread(train_fname, cv2.IMREAD_GRAYSCALE)  # 학습 영상 적재
    h, w = train_img.shape[:2]
    dy = h % size[1]// 2
    dx = w % size[0]// 2
    train_img = train_img[dy:h-dy-1, dx:w-dx-1]             # 학습 영상 여백 제거
    cv2.threshold(train_img, 32, 255, cv2.THRESH_BINARY, train_img)

    cells = [np.hsplit(row, nsample) for row in np.vsplit(train_img, nclass)]
    nums = [find_number(c) for c in np.reshape(cells, (-1, 40,40))]
    trainData = np.array([place_middle(n, size) for n in nums])
    labels = np.array([i for i in range(nclass) for j in range(nsample)], np.float32)

    knn = cv2.ml.KNearest_create()
    knn.train(trainData, cv2.ml.ROW_SAMPLE, labels)  # k-NN 학습 수행
    return knn

# 번호판 영상 전처리
def preprocessing_plate(plate_img):
    plate_img = cv2.resize(plate_img, (180, 35))            # 번호판 영상 크기 정규화
    flag = cv2.THRESH_BINARY | cv2.THRESH_OTSU              # 이진화 방법
    cv2.threshold(plate_img, 32, 255, flag, plate_img)      # 이진화

    h, w = plate_img.shape[:2]
    dx, dy = (6, 3)
    ret_img= plate_img[dy:h-dy, dx:w-dx]                    # 여백 제거
    return ret_img

# 숫자 및 문자 객체 검색
def find_objects(sub_mat):
    results = cv2.findContours(sub_mat, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rois = [cv2.boundingRect(contour) for contour in contours]
    rois = [(x, y, w, h, w*h) for x,y,w,h in rois if w / h < 2.5]

    text_rois = [(x, y, x+w, y+h) for x, y, w, h, a in rois if 45 < x < 80 and a > 60]
    num_rois  = [(x, y, w, h) for x, y, w, h, a in rois  if not(45 < x < 80) and a > 150]

    if text_rois:                         # 분리된 문자 영역 누적
        # pts= cv2.sort(np.array(text_rois), cv2.SORT_EVERY_COLUMN)  # 열단위 오름차순
        pts= np.sort(text_rois, axis=0)             # y 방향 정렬
        x0, y0 = pts[ 0, 0:2]                  # 시작좌표 중 최소인 x, y 좌표
        x1, y1 = pts[-1, 2:]                         # 종료좌표 중 최대인 x, y 좌표
        w, h = x1-x0, y1-y0                             # 너비, 높이 계산
        num_rois.append((x0, y0, w, h))             # 문자 영역 구성 및 저장

    return num_rois

# 검출 객체 영상의 숫자 및 문자 인식
def classify_numbers(cells, nknn, tknn, K1, K2, object_rois):
    if len(cells) != 7:
        print("검출된 숫자(문자)가 7개가 아닙니다.")
        return

    texts  = "가나다라마거너더러머버서어저고노도로모보" \
             "소오조구누두루무부수우주아바사자바하허호"

    numbers = [find_number(cell) for cell in cells]
    datas = [place_middle(num, (40,40)) for num in numbers]
    datas = np.reshape(datas, (len(datas), -1))

    idx = np.argsort(object_rois, axis=0).T[0]
    text = datas[idx[2]].reshape(1,-1)

    _, resp1, _, _ = nknn.findNearest(datas, K1)  # 숫자 k-NN 분류 수행
    _, [[resp2]], _, _ = tknn.findNearest(text, K2)  # 문자 k-NN 분류 수행

    resp1 = resp1.flatten().astype('int')
    results = resp1[idx].astype(str)
    results[2] = texts[int(resp2)]

    print("정렬 인덱스:", idx)
    print("숫자 분류 결과:", resp1)
    print("문자 분류 결과:", int(resp2))
    print("분류 결과: ", ' '.join(results))



















