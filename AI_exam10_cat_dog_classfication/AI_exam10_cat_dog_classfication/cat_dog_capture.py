import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PIL import Image
import numpy as np
from keras.models import load_model
import cv2

form_class = uic.loadUiType('./cat_and_dog.ui')[0]

class ExampleApp(QWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('./imgs/cat_dog/train/cat.18.jpg', '')
        self.btn_open.clicked.connect(self.btn_clicked_slot) # 버튼을 누르면 btn_clicked_slot 함수를 실행한다
        self.model = load_model('./models/cat_dog_model 0.817.h5')

    def btn_clicked_slot(self):
        capture = cv2.VideoCapture(0)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            v, frame = capture.read()
            print(type(frame))

            if(v):
                cv2.imshow('capture', frame)
                cv2.imwrite('./capture.png', frame)
            key = cv2.waitKey(15)
            if key == 27: # ESC 키를 누르면
                capture.release()
                cv2.destroyAllWindows()
                break
            try:
                pixmap = QPixmap('./capture.png')
                self.lbl_img.setPixmap(pixmap)
                img = Image.open('./capture.png')
                img = img.convert('RGB')
                img = img.resize((64, 64))
                img = np.array(img)
                img = img / 255
                img = img.reshape(1, 64, 64, 3)
                pred = self.model.predict(img)
                print(pred)
                if pred[0][0] > 0.5:
                    self.lbl_result.setText('강아지!')
                    print('강아지!')
                else:
                    self.lbl_result.setText('고양이!')
                    print('고양이!')
            except:
                print('error')


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ExampleApp()
    main_window.show()
    sys.exit(app.exec())