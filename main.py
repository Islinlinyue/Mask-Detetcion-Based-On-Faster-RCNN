import sys
from UI.Seg_UI import Ui_MainWindow, XX
from PyQt5.QtWidgets import QApplication, QMainWindow


if __name__ == '__main__':
    app = QApplication(sys.argv)

    mainWnd = XX()
    gui = Ui_MainWindow(mainWnd)

    mainWnd.show()
    sys.exit(app.exec_())