import numpy as np, cv2

def preprocessing(car_no):
    image = cv2.imread("images/car/%02d.jpg" %car_no, cv2.IMREAD_COLOR)
    if image is None: return None, None

    kernel = np.ones((5, 13), np.uint8)                          # 닫힘 연산 마스크
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)               # 명암도 영상 변환
    gray = cv2.blur(gray, (5, 5))                                # 블러링
    gray = cv2.Sobel(gray, cv2.CV_8U, 1, 0, 3)                   # 소벨 에지 검출
        
    th_img = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)[1] # 이진화 수행
    morph = cv2.morphologyEx(th_img, cv2.MORPH_CLOSE,  kernel, iterations=3)

    # cv2.imshow("th_img", th_img); cv2.imshow("morph", morph)
    return image, morph

def verify_aspect_size(size):
    w, h = size
    if h == 0 or w == 0: return False

    aspect = h/ w if h > w else w/ h       # 종횡비 계산

    chk1 = 3000 < (h * w) < 12000          # 번호판 넓이 조건
    chk2 = 2.0 < aspect < 6.5       # 번호판 종횡비 조건
    return (chk1 and chk2)

def find_candidates(image):
    results = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = results[0] if int(cv2.__version__[0]) >= 4 else results[1]

    rects = [cv2.minAreaRect(c) for c in contours]         # 외곽 최소 영역
    candidates = [(tuple(map(int, center)), tuple(map(int, size)), angle)
             for center, size, angle in rects if verify_aspect_size(size)]

    return candidates

# def draw_rotatedRect(image, rot_rect, color, thickness=2 , mode=False):
#     box = cv2.boxPoints(rot_rect)
#     pts = [pt for pt in np.int32(box)]           # box 원소의 자료형을 정수형 튜플로 변환
#     cv2.polylines(image, [np.array(pts)], True, color, thickness)      # pts는 numpy 배열
#
#     if mode:                                 # true면 회전 사각형 중심점 출력
#         center , _, _ = rot_rect
#         cv2.circle(image, center, 2, (255, 0, 0), 2)
#         cv2.imshow("rotated_image", image)