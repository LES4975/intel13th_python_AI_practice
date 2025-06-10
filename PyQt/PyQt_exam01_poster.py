import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

form_class = uic.loadUiType('./poster.ui')[0]

class ExampleApp(QWidget, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.label_list = [self.lbl_dragon, self.lbl_LiloStitch,
                           self.lbl_miim, self.lbl_soju]

        self.btn_dragon.clicked.connect(self.btn_slot)
        self.btn_LiloStitch.clicked.connect(self.btn_slot)
        self.btn_miim.clicked.connect(self.btn_slot)
        self.btn_soju.clicked.connect(self.btn_slot)

    def btn_slot(self):
        btn = self.sender()
        # 일단 모든 라벨을 숨기기
        for label in self.label_list:
            label.hide()

        # objectName의 마지막 3글자만 슬라이싱해서 if문으로 구분하기
        if btn.objectName()[-3:] == 'gon' : self.lbl_dragon.show()
        elif btn.objectName()[-3:] == 'tch' : self.lbl_LiloStitch.show()
        elif btn.objectName()[-3:] == 'iim': self.lbl_miim.show()
        elif btn.objectName()[-3:] == 'oju': self.lbl_soju.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ExampleApp()
    main_window.show()
    sys.exit(app.exec())