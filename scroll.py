import os
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QScrollBar, QWidget, QGridLayout, QLabel, QScrollArea, QTabWidget, QPushButton, QHBoxLayout, QComboBox


class ImageTab(QWidget):
    def __init__(self):
        super().__init__()

        self.width = 1000
        # Create a layout for the tab
        layout = QGridLayout(self)

        # Create a top layout for the buttons
        top_layout = QHBoxLayout()
        layout.addLayout(top_layout, 0, 0, 1, -1)

        # Create two buttons
        # Create two combo boxes
        combo_box_1 = QComboBox()
        combo_box_1.addItems(["Option 1", "Option 2", "Option 3"])
        combo_box_2 = QComboBox()
        combo_box_2.addItems(["Option A", "Option B", "Option C"])
        prev_button = QPushButton("Previous")

        # Add the buttons to the top layout
        top_layout.addWidget(combo_box_1)
        top_layout.addWidget(prev_button)
        top_layout.addWidget(combo_box_2)

        # Create a scroll area
        scroll_area = QScrollArea(self)
        scroll_area.setWidgetResizable(True)

        # Add a stylesheet to the scrollbar
        scroll_area.verticalScrollBar().setStyleSheet("QScrollBar:vertical {"
                                                       "    border: none;"
                                                       "    background: #f6f6f6;"
                                                       "    width: 15px;"
                                                       "    margin: 0px 0 0px 0;"
                                                       "}"
                                                       "QScrollBar::handle:vertical {"
                                                       "    background: #a9a9a9;"
                                                       "    border-radius: 7px;"
                                                       "    min-height: 20px;"
                                                       "}"
                                                       "QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {"
                                                       "    background: none;"
                                                       "}"
                                                       "QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {"
                                                       "    background: #dcdcdc;"
                                                       "}")

        # Create a widget to hold the images
        images_widget = QWidget(scroll_area)
        images_layout = QGridLayout(images_widget)
        row, col = 0, 0

        # Load all images in the folder
        images_folder = "/Users/cadenli/Documents/FractureX-Dataset/"
        for filename in os.listdir(images_folder):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(images_folder, filename)
                pixmap = QPixmap(image_path)
                label = QLabel()
                img_width = 180 #(self.width * .88) / 3
                label.setPixmap(pixmap.scaled(QSize(int(img_width), 3000), Qt.AspectRatioMode.KeepAspectRatio))
                images_layout.addWidget(label, row, col)
                col += 1
                if col == 5:
                    col = 0
                    row += 1

        # Set the widget to the scroll area
        scroll_area.setWidget(images_widget)

        # Add the scroll area to the layout
        layout.addWidget(scroll_area, 1, 0, -1, -1)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set the window properties
        self.setWindowTitle("Image Viewer")
        self.setGeometry(100, 100, 1000, 1000)

        # Create a tab widget
        tab_widget = QTabWidget(self)

        # Create the image tab and add it to the tab widget
        image_tab = ImageTab()
        tab_widget.addTab(image_tab, "Images")

        # Add the tab widget to the main window
        self.setCentralWidget(tab_widget)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
