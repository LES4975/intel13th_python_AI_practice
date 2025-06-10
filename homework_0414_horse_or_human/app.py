import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PIL import Image
import numpy as np
from keras.models import load_model

form_class = uic.loadUiType('./horse_human.ui')[0]

class ExampleApp(QWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.path = ('./start.jpg', '')
        self.btn_open.clicked.connect(self.btn_clicked_slot) # 버튼을 누르면 btn_clicked_slot 함수를 실행한다
        self.model = load_model('./models/horse_or_human_model 1.0.h5')

    def btn_clicked_slot(self):
        old_path = self.path  # 파일 경로 저장 임시 변수
        self.path = QFileDialog.getOpenFileName(self, 'Open File',
                                                './dataset/',
                                                'Image Files (*.jpg *.jpeg *.png);;All Files (*)')  # File Chooser를 실행한다, 파일 포맷도 설정
        print(self.path)
        if self.path[0] == '':
            self.path = old_path  # 파일 경로가 빈 문자열이라면 기존에 저장해놓았던 파일 경로로
        else:
            try:
                pixmap = QPixmap(self.path[0])
                self.lbl_img.setPixmap(pixmap)
                img = Image.open(self.path[0])
                img = img.convert('RGB')
                img = img.resize((150, 150))
                img = np.array(img)
                img = img / 255
                img = img.reshape(1, 150, 150, 3)
                pred = self.model.predict(img)
                print(pred)
                if pred[0][0] > 0.5:
                    self.lbl_result.setText('사람!')
                    print('사람!')
                else:
                    self.lbl_result.setText('말!')
                    print('말!')
            except:
                print('error')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ExampleApp()
    main_window.show()
    sys.exit(app.exec())