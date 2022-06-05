import numpy as np, cv2

def SVM_create(type, max_iter, epsilon):
    svm = cv2.ml.SVM_create()                           # SVM 객체 선언
    # SVM 파라미터 지정
    svm.setType(cv2.ml.SVM_C_SVC)             # C-Support Vector Classification
    svm.setKernel(cv2.ml.SVM_LINEAR)                    # 선형 SVM
    svm.setGamma(1)                                     # 커널 함수의 감마 값
    svm.setC(1)                                         # 최적화를 위한 C 파라미터
    svm.setTermCriteria((type, max_iter, epsilon))      # 학습 반복 조건 지정
    return svm

nsample = 140
trainData = [cv2.imread("images/plate/%03d.png" %i, 0) for i in range(nsample)]
trainData = np.reshape( trainData, (nsample, -1)).astype("float32")
labels = np.zeros((nsample,1), np.int32)
labels[:70] = 1

print("SVM 객체 생성")
svm = SVM_create(cv2.TERM_CRITERIA_MAX_ITER, 1000, 1e-6)   # SVM 객체 생성
svm.train(trainData, cv2.ml.ROW_SAMPLE, labels)             # 학습 수행
svm.save("SVMtrain.xml")                             # 학습된 데이터 저장
print("SVM 객체 저장 완료")