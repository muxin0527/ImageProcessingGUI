# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'MainWin.ui'
#
# Created by: PyQt5 UI code generator 5.15.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1128, 962)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.binaryFunction = QtWidgets.QPushButton(self.centralwidget)
        self.binaryFunction.setGeometry(QtCore.QRect(10, 140, 75, 23))
        self.binaryFunction.setMouseTracking(False)
        self.binaryFunction.setObjectName("binaryFunction")
        self.OSTUFunction = QtWidgets.QPushButton(self.centralwidget)
        self.OSTUFunction.setGeometry(QtCore.QRect(90, 140, 75, 23))
        self.OSTUFunction.setObjectName("OSTUFunction")
        self.resultLabel_1 = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel_1.setGeometry(QtCore.QRect(190, 20, 441, 401))
        self.resultLabel_1.setText("")
        self.resultLabel_1.setObjectName("resultLabel_1")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(170, 10, 16, 861))
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.resultLabel_2 = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel_2.setGeometry(QtCore.QRect(660, 20, 441, 401))
        self.resultLabel_2.setText("")
        self.resultLabel_2.setObjectName("resultLabel_2")
        self.resultLabel_3 = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel_3.setGeometry(QtCore.QRect(190, 450, 441, 401))
        self.resultLabel_3.setText("")
        self.resultLabel_3.setObjectName("resultLabel_3")
        self.resultLabel_4 = QtWidgets.QLabel(self.centralwidget)
        self.resultLabel_4.setGeometry(QtCore.QRect(660, 450, 441, 401))
        self.resultLabel_4.setText("")
        self.resultLabel_4.setObjectName("resultLabel_4")
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setGeometry(QtCore.QRect(630, 10, 31, 861))
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setGeometry(QtCore.QRect(0, 860, 1121, 16))
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setGeometry(QtCore.QRect(180, 420, 941, 20))
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setGeometry(QtCore.QRect(1110, 10, 20, 861))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setGeometry(QtCore.QRect(0, 0, 1121, 16))
        self.line_6.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.thresholdSlider = QtWidgets.QSlider(self.centralwidget)
        self.thresholdSlider.setGeometry(QtCore.QRect(10, 190, 160, 16))
        self.thresholdSlider.setOrientation(QtCore.Qt.Horizontal)
        self.thresholdSlider.setObjectName("thresholdSlider")
        self.clearFunction = QtWidgets.QPushButton(self.centralwidget)
        self.clearFunction.setGeometry(QtCore.QRect(590, 875, 110, 35))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(12)
        self.clearFunction.setFont(font)
        self.clearFunction.setMouseTracking(False)
        self.clearFunction.setObjectName("clearFunction")
        self.checkBox_1 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_1.setGeometry(QtCore.QRect(10, 20, 80, 16))
        self.checkBox_1.setObjectName("checkBox_1")
        self.checkBox_2 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_2.setGeometry(QtCore.QRect(90, 20, 80, 16))
        self.checkBox_2.setObjectName("checkBox_2")
        self.checkBox_3 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_3.setGeometry(QtCore.QRect(10, 50, 80, 16))
        self.checkBox_3.setObjectName("checkBox_3")
        self.checkBox_4 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_4.setGeometry(QtCore.QRect(90, 50, 80, 16))
        self.checkBox_4.setObjectName("checkBox_4")
        self.checkBox_5 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_5.setGeometry(QtCore.QRect(10, 80, 80, 16))
        self.checkBox_5.setObjectName("checkBox_5")
        self.checkBox_6 = QtWidgets.QCheckBox(self.centralwidget)
        self.checkBox_6.setGeometry(QtCore.QRect(90, 80, 80, 16))
        self.checkBox_6.setObjectName("checkBox_6")
        self.RobertsOperator = QtWidgets.QPushButton(self.centralwidget)
        self.RobertsOperator.setGeometry(QtCore.QRect(30, 460, 100, 30))
        self.RobertsOperator.setMouseTracking(False)
        self.RobertsOperator.setObjectName("RobertsOperator")
        self.PrewittOperator = QtWidgets.QPushButton(self.centralwidget)
        self.PrewittOperator.setGeometry(QtCore.QRect(30, 510, 100, 30))
        self.PrewittOperator.setMouseTracking(False)
        self.PrewittOperator.setObjectName("PrewittOperator")
        self.SobelOperator = QtWidgets.QPushButton(self.centralwidget)
        self.SobelOperator.setGeometry(QtCore.QRect(30, 560, 100, 30))
        self.SobelOperator.setMouseTracking(False)
        self.SobelOperator.setObjectName("SobelOperator")
        self.GaussianFilter = QtWidgets.QPushButton(self.centralwidget)
        self.GaussianFilter.setGeometry(QtCore.QRect(30, 200, 100, 30))
        self.GaussianFilter.setMouseTracking(False)
        self.GaussianFilter.setObjectName("GaussianFilter")
        self.MedianFilter = QtWidgets.QPushButton(self.centralwidget)
        self.MedianFilter.setGeometry(QtCore.QRect(30, 310, 100, 30))
        self.MedianFilter.setMouseTracking(False)
        self.MedianFilter.setObjectName("MedianFilter")
        self.Edgelabel = QtWidgets.QLabel(self.centralwidget)
        self.Edgelabel.setGeometry(QtCore.QRect(40, 410, 100, 25))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)
        self.Edgelabel.setFont(font)
        self.Edgelabel.setObjectName("Edgelabel")
        self.Noiselabel = QtWidgets.QLabel(self.centralwidget)
        self.Noiselabel.setGeometry(QtCore.QRect(40, 150, 100, 25))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)
        self.Noiselabel.setFont(font)
        self.Noiselabel.setObjectName("Noiselabel")
        self.GassianspinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.GassianspinBox.setGeometry(QtCore.QRect(50, 260, 61, 30))
        self.GassianspinBox.setObjectName("GassianspinBox")
        self.dilationFunction = QtWidgets.QPushButton(self.centralwidget)
        self.dilationFunction.setGeometry(QtCore.QRect(40, 160, 90, 30))
        self.dilationFunction.setObjectName("dilationFunction")
        self.erosionFunction = QtWidgets.QPushButton(self.centralwidget)
        self.erosionFunction.setGeometry(QtCore.QRect(40, 210, 90, 30))
        self.erosionFunction.setObjectName("erosionFunction")
        self.openingFunction = QtWidgets.QPushButton(self.centralwidget)
        self.openingFunction.setGeometry(QtCore.QRect(40, 260, 90, 30))
        self.openingFunction.setObjectName("openingFunction")
        self.closingFunction = QtWidgets.QPushButton(self.centralwidget)
        self.closingFunction.setGeometry(QtCore.QRect(40, 310, 90, 30))
        self.closingFunction.setObjectName("closingFunction")
        self.GraydilationFunction = QtWidgets.QPushButton(self.centralwidget)
        self.GraydilationFunction.setGeometry(QtCore.QRect(40, 160, 90, 30))
        self.GraydilationFunction.setObjectName("GraydilationFunction")
        self.GrayerosionFunction = QtWidgets.QPushButton(self.centralwidget)
        self.GrayerosionFunction.setGeometry(QtCore.QRect(40, 260, 90, 30))
        self.GrayerosionFunction.setObjectName("GrayerosionFunction")
        self.GrayopeningFunction = QtWidgets.QPushButton(self.centralwidget)
        self.GrayopeningFunction.setGeometry(QtCore.QRect(40, 370, 90, 30))
        self.GrayopeningFunction.setObjectName("GrayopeningFunction")
        self.GrayclosingFunction = QtWidgets.QPushButton(self.centralwidget)
        self.GrayclosingFunction.setGeometry(QtCore.QRect(40, 470, 90, 30))
        self.GrayclosingFunction.setObjectName("GrayclosingFunction")
        self.TopleftLabel = QtWidgets.QLabel(self.centralwidget)
        self.TopleftLabel.setGeometry(QtCore.QRect(200, 30, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.TopleftLabel.setFont(font)
        self.TopleftLabel.setObjectName("TopleftLabel")
        self.TopRightLabel = QtWidgets.QLabel(self.centralwidget)
        self.TopRightLabel.setGeometry(QtCore.QRect(670, 30, 91, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.TopRightLabel.setFont(font)
        self.TopRightLabel.setObjectName("TopRightLabel")
        self.BottomLeftLabel = QtWidgets.QLabel(self.centralwidget)
        self.BottomLeftLabel.setGeometry(QtCore.QRect(200, 460, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.BottomLeftLabel.setFont(font)
        self.BottomLeftLabel.setObjectName("BottomLeftLabel")
        self.BottomRightLabel = QtWidgets.QLabel(self.centralwidget)
        self.BottomRightLabel.setGeometry(QtCore.QRect(670, 460, 121, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.BottomRightLabel.setFont(font)
        self.BottomRightLabel.setObjectName("BottomRightLabel")
        self.SkeletonFunction = QtWidgets.QPushButton(self.centralwidget)
        self.SkeletonFunction.setGeometry(QtCore.QRect(40, 280, 91, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.SkeletonFunction.setFont(font)
        self.SkeletonFunction.setObjectName("SkeletonFunction")
        self.SkeletonRestoration = QtWidgets.QPushButton(self.centralwidget)
        self.SkeletonRestoration.setGeometry(QtCore.QRect(40, 330, 91, 30))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.SkeletonRestoration.setFont(font)
        self.SkeletonRestoration.setObjectName("SkeletonRestoration")
        self.binaryImage = QtWidgets.QRadioButton(self.centralwidget)
        self.binaryImage.setGeometry(QtCore.QRect(40, 210, 100, 16))
        self.binaryImage.setObjectName("binaryImage")
        self.otherImage = QtWidgets.QRadioButton(self.centralwidget)
        self.otherImage.setGeometry(QtCore.QRect(40, 240, 100, 16))
        self.otherImage.setObjectName("otherImage")
        self.DTLabel = QtWidgets.QLabel(self.centralwidget)
        self.DTLabel.setGeometry(QtCore.QRect(70, 180, 54, 20))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.DTLabel.setFont(font)
        self.DTLabel.setObjectName("DTLabel")
        self.EdgeDetection = QtWidgets.QPushButton(self.centralwidget)
        self.EdgeDetection.setGeometry(QtCore.QRect(10, 180, 141, 30))
        self.EdgeDetection.setObjectName("EdgeDetection")
        self.Gradient = QtWidgets.QPushButton(self.centralwidget)
        self.Gradient.setGeometry(QtCore.QRect(10, 230, 141, 30))
        self.Gradient.setObjectName("Gradient")
        self.ConditionalDilation = QtWidgets.QPushButton(self.centralwidget)
        self.ConditionalDilation.setGeometry(QtCore.QRect(10, 280, 141, 30))
        self.ConditionalDilation.setObjectName("ConditionalDilation")
        self.ConditionalDilationSlider = QtWidgets.QSlider(self.centralwidget)
        self.ConditionalDilationSlider.setGeometry(QtCore.QRect(10, 330, 160, 16))
        self.ConditionalDilationSlider.setOrientation(QtCore.Qt.Horizontal)
        self.ConditionalDilationSlider.setObjectName("ConditionalDilationSlider")
        self.GrayReconstructionLabel = QtWidgets.QLabel(self.centralwidget)
        self.GrayReconstructionLabel.setGeometry(QtCore.QRect(40, 380, 100, 25))
        font = QtGui.QFont()
        font.setFamily("Microsoft YaHei UI")
        font.setPointSize(11)
        self.GrayReconstructionLabel.setFont(font)
        self.GrayReconstructionLabel.setObjectName("GrayReconstructionLabel")
        self.OBRFunction = QtWidgets.QPushButton(self.centralwidget)
        self.OBRFunction.setGeometry(QtCore.QRect(30, 420, 100, 30))
        self.OBRFunction.setMouseTracking(False)
        self.OBRFunction.setObjectName("OBRFunction")
        self.CBRFunction = QtWidgets.QPushButton(self.centralwidget)
        self.CBRFunction.setGeometry(QtCore.QRect(30, 540, 100, 30))
        self.CBRFunction.setMouseTracking(False)
        self.CBRFunction.setObjectName("CBRFunction")
        self.OBRSpinBox = QtWidgets.QSpinBox(self.centralwidget)
        self.OBRSpinBox.setGeometry(QtCore.QRect(50, 470, 61, 22))
        self.OBRSpinBox.setObjectName("OBRSpinBox")
        self.OBRSE = QtWidgets.QLabel(self.centralwidget)
        self.OBRSE.setGeometry(QtCore.QRect(20, 470, 41, 16))
        font = QtGui.QFont()
        font.setPointSize(11)
        self.OBRSE.setFont(font)
        self.OBRSE.setObjectName("OBRSE")
        self.GraydilationFunctionSlider = QtWidgets.QSlider(self.centralwidget)
        self.GraydilationFunctionSlider.setGeometry(QtCore.QRect(10, 220, 160, 16))
        self.GraydilationFunctionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.GraydilationFunctionSlider.setObjectName("GraydilationFunctionSlider")
        self.GrayerosionFunctionSlider = QtWidgets.QSlider(self.centralwidget)
        self.GrayerosionFunctionSlider.setGeometry(QtCore.QRect(10, 330, 160, 16))
        self.GrayerosionFunctionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.GrayerosionFunctionSlider.setObjectName("GrayerosionFunctionSlider")
        self.GrayopeningFunctionSlider = QtWidgets.QSlider(self.centralwidget)
        self.GrayopeningFunctionSlider.setGeometry(QtCore.QRect(10, 420, 160, 16))
        self.GrayopeningFunctionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.GrayopeningFunctionSlider.setObjectName("GrayopeningFunctionSlider")
        self.GrayclosingFunctionSlider = QtWidgets.QSlider(self.centralwidget)
        self.GrayclosingFunctionSlider.setGeometry(QtCore.QRect(10, 540, 160, 16))
        self.GrayclosingFunctionSlider.setOrientation(QtCore.Qt.Horizontal)
        self.GrayclosingFunctionSlider.setObjectName("GrayclosingFunctionSlider")
        self.EntropyThreshold = QtWidgets.QLabel(self.centralwidget)
        self.EntropyThreshold.setGeometry(QtCore.QRect(10, 280, 101, 41))
        self.EntropyThreshold.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.EntropyThreshold.setObjectName("EntropyThreshold")
        self.OSTUThreshold = QtWidgets.QLabel(self.centralwidget)
        self.OSTUThreshold.setGeometry(QtCore.QRect(10, 340, 101, 41))
        self.OSTUThreshold.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.OSTUThreshold.setObjectName("OSTUThreshold")
        self.EntropyThresholdShow = QtWidgets.QLabel(self.centralwidget)
        self.EntropyThresholdShow.setGeometry(QtCore.QRect(110, 295, 54, 12))
        self.EntropyThresholdShow.setObjectName("EntropyThresholdShow")
        self.OSTUThresholdShow = QtWidgets.QLabel(self.centralwidget)
        self.OSTUThresholdShow.setGeometry(QtCore.QRect(110, 355, 54, 12))
        self.OSTUThresholdShow.setObjectName("OSTUThresholdShow")
        self.ThresholdSliderLabel = QtWidgets.QLabel(self.centralwidget)
        self.ThresholdSliderLabel.setGeometry(QtCore.QRect(10, 400, 101, 41))
        self.ThresholdSliderLabel.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.ThresholdSliderLabel.setObjectName("ThresholdSliderLabel")
        self.ThresholdSliderShow = QtWidgets.QLabel(self.centralwidget)
        self.ThresholdSliderShow.setGeometry(QtCore.QRect(110, 415, 54, 12))
        self.ThresholdSliderShow.setObjectName("ThresholdSliderShow")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1128, 22))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.fileOpenAction = QtWidgets.QAction(MainWindow)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        self.fileOpenAction.setFont(font)
        self.fileOpenAction.setObjectName("fileOpenAction")
        self.menuFile.addAction(self.fileOpenAction)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.binaryFunction.setText(_translate("MainWindow", "最大熵"))
        self.OSTUFunction.setText(_translate("MainWindow", "OSTU"))
        self.clearFunction.setText(_translate("MainWindow", "清除"))
        self.checkBox_1.setText(_translate("MainWindow", "作业1"))
        self.checkBox_2.setText(_translate("MainWindow", "作业2"))
        self.checkBox_3.setText(_translate("MainWindow", "作业3"))
        self.checkBox_4.setText(_translate("MainWindow", "作业4"))
        self.checkBox_5.setText(_translate("MainWindow", "作业5"))
        self.checkBox_6.setText(_translate("MainWindow", "作业6"))
        self.RobertsOperator.setText(_translate("MainWindow", "Roberts算子"))
        self.PrewittOperator.setText(_translate("MainWindow", "Prewitt算子"))
        self.SobelOperator.setText(_translate("MainWindow", "Sobel算子"))
        self.GaussianFilter.setText(_translate("MainWindow", "高斯滤波"))
        self.MedianFilter.setText(_translate("MainWindow", "中值滤波"))
        self.Edgelabel.setText(_translate("MainWindow", "边缘检测"))
        self.Noiselabel.setText(_translate("MainWindow", "图像去噪"))
        self.dilationFunction.setText(_translate("MainWindow", "膨胀"))
        self.erosionFunction.setText(_translate("MainWindow", "腐蚀"))
        self.openingFunction.setText(_translate("MainWindow", "开运算"))
        self.closingFunction.setText(_translate("MainWindow", "闭运算"))
        self.GraydilationFunction.setText(_translate("MainWindow", "灰度膨胀"))
        self.GrayerosionFunction.setText(_translate("MainWindow", "灰度腐蚀"))
        self.GrayopeningFunction.setText(_translate("MainWindow", "灰度开运算"))
        self.GrayclosingFunction.setText(_translate("MainWindow", "灰度闭运算"))
        self.TopleftLabel.setText(_translate("MainWindow", "1"))
        self.TopRightLabel.setText(_translate("MainWindow", "2"))
        self.BottomLeftLabel.setText(_translate("MainWindow", "3"))
        self.BottomRightLabel.setText(_translate("MainWindow", "4"))
        self.SkeletonFunction.setText(_translate("MainWindow", "Skeleton"))
        self.SkeletonRestoration.setText(_translate("MainWindow", "SR"))
        self.binaryImage.setText(_translate("MainWindow", "二值化图像"))
        self.otherImage.setText(_translate("MainWindow", "其他图像"))
        self.DTLabel.setText(_translate("MainWindow", "DT"))
        self.EdgeDetection.setText(_translate("MainWindow", "形态学边缘检测"))
        self.Gradient.setText(_translate("MainWindow", "形态学梯度"))
        self.ConditionalDilation.setText(_translate("MainWindow", "Conditional D"))
        self.GrayReconstructionLabel.setText(_translate("MainWindow", "灰度重建"))
        self.OBRFunction.setText(_translate("MainWindow", "OBR"))
        self.CBRFunction.setText(_translate("MainWindow", "CBR"))
        self.OBRSE.setText(_translate("MainWindow", "SE:"))
        self.EntropyThreshold.setText(_translate("MainWindow", "最大熵阈值："))
        self.OSTUThreshold.setText(_translate("MainWindow", "OSTU阈值："))
        self.EntropyThresholdShow.setText(_translate("MainWindow", "0"))
        self.OSTUThresholdShow.setText(_translate("MainWindow", "0"))
        self.ThresholdSliderLabel.setText(_translate("MainWindow", "滑动阈值："))
        self.ThresholdSliderShow.setText(_translate("MainWindow", "0"))
        self.menuFile.setTitle(_translate("MainWindow", "File(&F)"))
        self.fileOpenAction.setText(_translate("MainWindow", "打开文件"))
        self.fileOpenAction.setShortcut(_translate("MainWindow", "Ctrl+O"))