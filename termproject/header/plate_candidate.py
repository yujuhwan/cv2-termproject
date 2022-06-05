import numpy as np, cv2

def color_candidate_img(image, center):
    h, w = image.shape[:2]
    fill = np.zeros((h + 2, w + 2), np.uint8)           # 채움 영역
    dif1, dif2 = (25, 25, 25), (25, 25, 25)             # 채움 색상 범위
    flags = 4 + 0xff00 + cv2.FLOODFILL_FIXED_RANGE                                # 채움 방향
    flags += cv2.FLOODFILL_MASK_ONLY

    # 후보 영역을 유사 컬러로 채우기
    pts = np.random.randint( -15, 15, (20,2) )          # 20개 좌표 생성
    pts = pts + center
    for x, y in pts:                                 # 랜덤 좌표 평행 이동
        if 0 <= x < w and 0 <= y < h:
            _, _, fill, _ = cv2.floodFill(image, fill, (x,y), 255, dif1, dif2, flags)

    # 이진화 및 외곽영역 추출후 회전 사각형 검출
    return cv2.threshold(fill, 120, 255, cv2.THRESH_BINARY)[1]

def rotate_plate(image, rect):
    center, (w, h), angle = rect       # 중심점, 크기, 회전 각도
    if w < h :                         # 세로가 긴 영역이면
        w, h = h, w                    # 가로와 세로 맞바꿈
        angle += 90                    # 회전 각도 조정

    size = image.shape[1::-1]            # 행태와 크기는 역순
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1)  # 회전 행렬 계산
    rot_img= cv2.warpAffine(image, rot_mat, size, cv2.INTER_CUBIC)  # 회전 변환

    crop_img = cv2.getRectSubPix(rot_img, (w, h), center)  # 후보영역 가져오기
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    return cv2.resize(crop_img, (144, 28))

