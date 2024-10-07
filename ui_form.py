# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QComboBox, QLabel, QPushButton,
    QSizePolicy, QTextEdit, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1169, 640)
        self.label = QLabel(Widget)
        self.label.setObjectName(u"label")
        self.label.setGeometry(QRect(500, 20, 191, 51))
        font = QFont()
        font.setPointSize(25)
        self.label.setFont(font)
        self.label_2 = QLabel(Widget)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setGeometry(QRect(480, 120, 481, 21))
        font1 = QFont()
        font1.setPointSize(10)
        self.label_2.setFont(font1)
        self.pushButton = QPushButton(Widget)
        self.pushButton.setObjectName(u"pushButton")
        self.pushButton.setGeometry(QRect(220, 110, 191, 41))
        font2 = QFont()
        font2.setPointSize(15)
        self.pushButton.setFont(font2)
        self.pushButton_2 = QPushButton(Widget)
        self.pushButton_2.setObjectName(u"pushButton_2")
        self.pushButton_2.setGeometry(QRect(490, 520, 211, 61))
        self.pushButton_2.setFont(font2)
        self.comboBox = QComboBox(Widget)
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setObjectName(u"comboBox")
        self.comboBox.setGeometry(QRect(220, 160, 191, 31))
        self.label_3 = QLabel(Widget)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setGeometry(QRect(248, 290, 61, 20))
        font3 = QFont()
        font3.setPointSize(12)
        self.label_3.setFont(font3)
        self.ApiInput = QTextEdit(Widget)
        self.ApiInput.setObjectName(u"ApiInput")
        self.ApiInput.setGeometry(QRect(320, 280, 581, 31))
        self.label_4 = QLabel(Widget)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setGeometry(QRect(320, 470, 51, 20))
        self.label_4.setFont(font3)
        self.label_5 = QLabel(Widget)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setGeometry(QRect(600, 350, 141, 20))
        self.label_5.setFont(font3)
        self.label_6 = QLabel(Widget)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setGeometry(QRect(660, 410, 81, 20))
        self.label_6.setFont(font3)
        self.EpochsInput = QTextEdit(Widget)
        self.EpochsInput.setObjectName(u"EpochsInput")
        self.EpochsInput.setGeometry(QRect(380, 460, 81, 31))
        self.TrainSplitInput = QTextEdit(Widget)
        self.TrainSplitInput.setObjectName(u"TrainSplitInput")
        self.TrainSplitInput.setGeometry(QRect(750, 340, 81, 31))
        self.BatchInput = QTextEdit(Widget)
        self.BatchInput.setObjectName(u"BatchInput")
        self.BatchInput.setGeometry(QRect(750, 400, 81, 31))
        self.label_7 = QLabel(Widget)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setGeometry(QRect(550, 470, 181, 20))
        self.label_7.setFont(font3)
        self.ExportNameInput = QTextEdit(Widget)
        self.ExportNameInput.setObjectName(u"ExportNameInput")
        self.ExportNameInput.setGeometry(QRect(750, 460, 151, 31))
        self.label_8 = QLabel(Widget)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setGeometry(QRect(240, 350, 131, 20))
        self.label_8.setFont(font3)
        self.NumOQubInput = QTextEdit(Widget)
        self.NumOQubInput.setObjectName(u"NumOQubInput")
        self.NumOQubInput.setGeometry(QRect(380, 340, 81, 31))
        self.RepsInput = QTextEdit(Widget)
        self.RepsInput.setObjectName(u"RepsInput")
        self.RepsInput.setGeometry(QRect(380, 400, 81, 31))
        self.RepsInput.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        self.label_9 = QLabel(Widget)
        self.label_9.setObjectName(u"label_9")
        self.label_9.setGeometry(QRect(330, 410, 41, 20))
        self.label_9.setFont(font3)

        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"Widget", None))
        self.label.setText(QCoreApplication.translate("Widget", u"QNN Trainer", None))
        self.label_2.setText(QCoreApplication.translate("Widget", u"No file selected", None))
        self.pushButton.setText(QCoreApplication.translate("Widget", u"Select file", None))
        self.pushButton_2.setText(QCoreApplication.translate("Widget", u"Start Training", None))
        self.comboBox.setItemText(0, QCoreApplication.translate("Widget", u"Select a backend", None))
        self.comboBox.setItemText(1, QCoreApplication.translate("Widget", u"default.qubit", None))
        self.comboBox.setItemText(2, QCoreApplication.translate("Widget", u"Real Quantum Machine", None))
        self.comboBox.setItemText(3, QCoreApplication.translate("Widget", u"lightning.qubit", None))
        self.comboBox.setItemText(4, QCoreApplication.translate("Widget", u"qiskit.aer", None))

        self.label_3.setText(QCoreApplication.translate("Widget", u"Api Key:", None))
        self.label_4.setText(QCoreApplication.translate("Widget", u"Epochs:", None))
        self.label_5.setText(QCoreApplication.translate("Widget", u"Train/Test Split (%):", None))
        self.label_6.setText(QCoreApplication.translate("Widget", u"Batch Size:", None))
        self.label_7.setText(QCoreApplication.translate("Widget", u"Exported model file name:", None))
        self.label_8.setText(QCoreApplication.translate("Widget", u"Number of qubits:", None))
        self.RepsInput.setHtml(QCoreApplication.translate("Widget", u"<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:'Segoe UI'; font-size:9pt; font-weight:400; font-style:normal;\">\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>", None))
        self.label_9.setText(QCoreApplication.translate("Widget", u"Reps:", None))
    # retranslateUi

