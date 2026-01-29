# main.py
from PySide6.QtWidgets import QApplication
from read_comtrade_gui import ComtradeViewer

def main():
    app = QApplication([])
    w = ComtradeViewer()
    w.resize(1600, 900)
    w.show()
    app.exec()

if __name__ == "__main__":
    main()
