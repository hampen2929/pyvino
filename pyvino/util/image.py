
import cv2
import numpy as np

def pil2cv(image):
    ''' PIL -> OpenCV '''
    new_image = np.array(image)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
    return new_image


def imshow(image):
    image = pil2cv(image)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cam_test(model):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    while True:
        ret, frame = cap.read()
        frame = np.array(frame)

        ############################
        frame = model.compute(frame)
        ############################

        cv2.imshow('demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


