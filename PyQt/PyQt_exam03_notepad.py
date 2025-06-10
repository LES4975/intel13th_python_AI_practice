import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic

form_class = uic.loadUiType('./notepad.ui')[0]

class ExampleApp(QMainWindow, form_class): # 매개변수로 그동안은 QWidget을 썼었지만, 이번에는 QMainWindow
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = ("제목 없음", '') # 초기 파일 이름 (걍 빈 문서)
        self.setWindowTitle(self.path[0] + " - QT Notepad") # 제목표시줄에 표시할 이름

        self.edited_flag = False # 파일이 열린 뒤에 내용이 수정되었을 경우에 올리는 플래그
        self.plain_te.textChanged.connect(self.text_change_slot) # text가 변경될 때마다 함수를 호출

        # File 메뉴 기능
        self.actionSave_as.triggered.connect(self.save_as_slot)
        self.actionSave.triggered.connect(self.save_slot)
        self.actionOpen.triggered.connect(self.open_slot)
        self.actionNew.triggered.connect(self.new_slot)
        self.actionExit.triggered.connect(self.exit_slot)

        # Edit 메뉴 기능
        self.actionUndo.triggered.connect(self.plain_te.undo)
        self.actionRedo.triggered.connect(self.plain_te.redo)
        self.actionCopy.triggered.connect(self.plain_te.copy)
        self.actionCut.triggered.connect(self.plain_te.cut)
        self.actionPaste.triggered.connect(self.plain_te.paste)
        self.actionDelete.triggered.connect(self.plain_te.cut) # Plain Text에는 delete 기능이 없다.
        self.actionSelect_All.triggered.connect(self.plain_te.selectAll)

        # Font 기능 구현
        self.actionFont.triggered.connect(self.font_slot)

        # Help 기능 구현
        self.actionAbout.triggered.connect(self.about_slot)

    # 파일 수정 여부 확인
    def text_change_slot(self):
        self.edited_flag = True # 파일이 수정되면 플래그를 True로 설정
        self.setWindowTitle(self.path[0].split('/')[-1] + '*' + " - QT Notepad")  # 새로 저장했으면 저장 경로 갱신
        self.plain_te.textChanged.disconnect(self.text_change_slot) # 내용 변경 신호를 disconnect(더 이상 함수가 호출되지 않도록)
        print("change")

    # 수정 후에 저장하려고 할 때 물어보는 함수
    def save_edited(self):
        ans = QMessageBox.question(self, '저장하기', '저장할까요?',
                                   QMessageBox.No | QMessageBox.Cancel | QMessageBox.Yes,
                                   QMessageBox.Yes) # No, Cancel, Yes. Default: Yes로 포커스
        if ans == QMessageBox.Yes :
            return self.save_slot()
        elif ans == QMessageBox.No :
            return 0
        elif ans == QMessageBox.Cancel :
            return 1

    # 작업 초기화
    def init_edit(self):
        self.setWindowTitle(self.path[0].split('/')[-1] + " - QT Notepad")
        self.plain_te.textChanged.connect(self.text_change_slot)
        self.edited_flag = False


    # File 메뉴 기능 구현 -----------------------------------
    # Save as
    def save_as_slot(self):
        old_path = self.path
        self.path = QFileDialog.getSaveFileName(self,
                                                'save file', '',
                                                'Text Files(*.txt);;Python Files(*.py);;All Files(*.*)')
        # 파일 선택창 이름, 파일 선택창으로 띄울 디렉토리 경로(공백으로 설정하면 실행 파일이 들어 있는 폴더를 엶), 파일 형식
        print(self.path)
        if self.path[0]: # path 값이 있으면?
            with open(self.path[0], 'w') as f:
                f.write(self.plain_te.toPlainText())  # 텍스트 창에 입력된 문자열들을 파일로서 저장
            self.init_edit()
            return 0
        else :
            self.path = old_path
            return 1

    # Save
    def save_slot(self):
        if self.path[0] == "제목 없음":
            return self.save_as_slot()
        else :
            if self.edited_flag :
                self.plain_te.textChanged.connect(self.text_change_slot)
                with open(self.path[0], 'w') as f:
                    f.write(self.plain_te.toPlainText())
                self.init_edit()
            return 0
    # Open
    def open_slot(self):
        if self.edited_flag :# 저장할 거면 save_slot 호출
            if self.save_edited() : # Cancel을 선택해서 아무 것도 안 해야 할 때
                return
        old_path = self.path # 기존 파일 경로 저장
        self.path = QFileDialog.getOpenFileName(self,
                                                'Open file', '',
                                                'Text Files(*.txt);;Python Files(*.py);;All Files(*.*)') # 파일 경로 저장
        print(self.path)
        if self.path[0] : # 파일 경로가 유효하면
            with open(self.path[0], 'r') as f:
                str_read = f.read()
            self.plain_te.setPlainText(str_read) # 파일 내용을 Plain Text 칸에 띄우기
            self.init_edit()
        else :
            self.path = old_path # 기존 파일 경로로 덮어쓰기

    # New
    def new_slot(self):
        if self.edited_flag: # 저장할 거면 save_slot 호출
            if self.save_edited():  # Cancel을 선택해서 아무 것도 안 해야 할 때
                return

        self.path = ('제목 없음', '')
        self.plain_te.setPlainText('')  # Plain Text 안의 텍스트를 리셋
        self.init_edit()

    # Exit
    def exit_slot(self):
        if self.edited_flag: # 저장할 거면 save_slot 호출
            if self.save_edited():  # Cancel을 선택해서 아무 것도 안 해야 할 때
                return
        self.close() # 프로그램 close


    # Form 메뉴 기능 구현--------------------------
    def font_slot(self):
        font = QFontDialog.getFont()
        print(font)
        if font[1]: # font 튜플의 1번째 값이 True라면
            self.plain_te.setFont(font[0])

    # Help 메뉴 기능 구현 ---------------------
    def about_slot(self):
        QMessageBox.about(self, 'Qt Notepad',
                          '만든이: 나\n\r버전 정보: 1.0.0')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = ExampleApp()
    main_window.show()
    sys.exit(app.exec())