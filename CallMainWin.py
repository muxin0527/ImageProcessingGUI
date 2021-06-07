import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QSlider
from PyQt5.QtGui import QPixmap, QImage, QPainter
from PyQt5.QtCore import Qt
from MainWin import *
import cv2
import numpy as np
from skimage import morphology


class fileOpenException(Exception):
    def __init__(self, text):
        self.text = text


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.label_width = self.resultLabel_1.width()
        self.label_height = self.resultLabel_1.height()
        self.initUI()
        self.initSlider()
        self.initSpinBox()
        self.initRadioButton()
        self.sig_slot_connect()

    # 设置背景色为黑色
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setBrush(Qt.black)
        painter.drawRect(177, 35, 463, 415)
        painter.drawRect(648, 35, 467, 415)
        painter.drawRect(177, 460, 463, 420)
        painter.drawRect(648, 460, 467, 420)

    # 初始化一些Qt Designer没有的界面设置
    def initUI(self):
        # 将Label的边框设置为红色
        self.resultLabel_1.setStyleSheet("border: 2px solid red")
        self.resultLabel_2.setStyleSheet("border: 2px solid red")
        self.resultLabel_3.setStyleSheet("border: 2px solid red")
        self.resultLabel_4.setStyleSheet("border: 2px solid red")
        self.TopleftLabel.setStyleSheet("color:white")
        self.TopRightLabel.setStyleSheet("color:white")
        self.BottomLeftLabel.setStyleSheet("color:white")
        self.BottomRightLabel.setStyleSheet("color:white")
        # 设置所有窗口控件初始不可见
        self.TopleftLabel.setVisible(False)
        self.TopRightLabel.setVisible(False)
        self.BottomLeftLabel.setVisible(False)
        self.BottomRightLabel.setVisible(False)
        # 作业1控件
        self.binaryFunction.setVisible(False)
        self.OSTUFunction.setVisible(False)
        self.thresholdSlider.setVisible(False)
        self.EntropyThreshold.setVisible(False)
        self.OSTUThreshold.setVisible(False)
        self.OSTUThresholdShow.setVisible(False)
        self.EntropyThresholdShow.setVisible(False)
        self.ThresholdSliderLabel.setVisible(False)
        self.ThresholdSliderShow.setVisible(False)
        # 作业2控件
        self.RobertsOperator.setVisible(False)
        self.PrewittOperator.setVisible(False)
        self.SobelOperator.setVisible(False)
        self.GaussianFilter.setVisible(False)
        self.MedianFilter.setVisible(False)
        self.Edgelabel.setVisible(False)
        self.Noiselabel.setVisible(False)
        self.GassianspinBox.setVisible(False)
        # 作业3控件
        self.dilationFunction.setVisible(False)
        self.erosionFunction.setVisible(False)
        self.openingFunction.setVisible(False)
        self.closingFunction.setVisible(False)
        # 作业4控件
        self.SkeletonFunction.setVisible(False)
        self.SkeletonRestoration.setVisible(False)
        self.binaryImage.setVisible(False)
        self.otherImage.setVisible(False)
        self.DTLabel.setVisible(False)
        # 作业5控件
        self.GraydilationFunction.setVisible(False)
        self.GrayerosionFunction.setVisible(False)
        self.GrayopeningFunction.setVisible(False)
        self.GrayclosingFunction.setVisible(False)
        self.GraydilationFunctionSlider.setVisible(False)
        self.GrayerosionFunctionSlider.setVisible(False)
        self.GrayopeningFunctionSlider.setVisible(False)
        self.GrayclosingFunctionSlider.setVisible(False)
        # 作业6控件
        self.EdgeDetection.setVisible(False)
        self.Gradient.setVisible(False)
        self.ConditionalDilationSlider.setVisible(False)
        self.ConditionalDilation.setVisible(False)
        self.GrayReconstructionLabel.setVisible(False)
        self.OBRFunction.setVisible(False)
        self.CBRFunction.setVisible(False)
        self.OBRSE.setVisible(False)
        self.OBRSpinBox.setVisible(False)

    # 初始化滑动条
    def initSlider(self):
        # 二值化滑动条
        self.thresholdSlider.setMinimum(0)
        self.thresholdSlider.setMaximum(255)
        self.thresholdSlider.setSingleStep(10)
        self.thresholdSlider.setValue(100)
        self.thresholdSlider.setTickPosition(QSlider.TicksBelow)
        self.thresholdSlider.setTickInterval(50)
        # conditional dilation滑动条
        self.ConditionalDilationSlider.setMinimum(1)
        self.ConditionalDilationSlider.setMaximum(30)
        self.ConditionalDilationSlider.setSingleStep(1)
        self.ConditionalDilationSlider.setValue(25)
        self.ConditionalDilationSlider.setTickPosition(QSlider.TicksBelow)
        self.ConditionalDilationSlider.setTickInterval(5)

        self.GraydilationFunctionSlider.setMinimum(1)
        self.GraydilationFunctionSlider.setMaximum(30)
        self.GraydilationFunctionSlider.setSingleStep(1)
        self.GraydilationFunctionSlider.setValue(10)
        self.GraydilationFunctionSlider.setTickPosition(QSlider.TicksBelow)
        self.GraydilationFunctionSlider.setTickInterval(5)

        self.GrayerosionFunctionSlider.setMinimum(1)
        self.GrayerosionFunctionSlider.setMaximum(30)
        self.GrayerosionFunctionSlider.setSingleStep(1)
        self.GrayerosionFunctionSlider.setValue(10)
        self.GrayerosionFunctionSlider.setTickPosition(QSlider.TicksBelow)
        self.GrayerosionFunctionSlider.setTickInterval(5)

        self.GrayopeningFunctionSlider.setMinimum(1)
        self.GrayopeningFunctionSlider.setMaximum(30)
        self.GrayopeningFunctionSlider.setSingleStep(1)
        self.GrayopeningFunctionSlider.setValue(10)
        self.GrayopeningFunctionSlider.setTickPosition(QSlider.TicksBelow)
        self.GrayopeningFunctionSlider.setTickInterval(5)

        self.GrayclosingFunctionSlider.setMinimum(1)
        self.GrayclosingFunctionSlider.setMaximum(30)
        self.GrayclosingFunctionSlider.setSingleStep(1)
        self.GrayclosingFunctionSlider.setValue(10)
        self.GrayclosingFunctionSlider.setTickPosition(QSlider.TicksBelow)
        self.GrayclosingFunctionSlider.setTickInterval(5)

    # 初始化计数器
    def initSpinBox(self):
        self.GassianspinBox.setMinimum(1)
        self.GassianspinBox.setMaximum(15)
        self.GassianspinBox.setSingleStep(2)
        self.GassianspinBox.setValue(1)
        self.OBRSpinBox.setMinimum(5)
        self.OBRSpinBox.setMaximum(30)
        self.OBRSpinBox.setSingleStep(5)
        self.OBRSpinBox.setValue(10)

    # 初始化RadioButton
    def initRadioButton(self):
        self.otherImage.setChecked(True)

    # 信号与槽的连接函数
    def sig_slot_connect(self):
        # 点击作业1checkbox显示控件信号
        self.checkBox_1.clicked['bool'].connect(self.binaryFunction.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.OSTUFunction.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.thresholdSlider.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.OSTUThreshold.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.EntropyThreshold.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.OSTUThresholdShow.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.EntropyThresholdShow.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.ThresholdSliderShow.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.ThresholdSliderLabel.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.TopleftLabel.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.TopRightLabel.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.BottomRightLabel.setVisible)
        self.checkBox_1.clicked['bool'].connect(self.BottomLeftLabel.setVisible)
        # 点击作业2checkbox显示控件信号
        self.checkBox_2.clicked['bool'].connect(self.RobertsOperator.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.PrewittOperator.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.SobelOperator.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.GaussianFilter.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.MedianFilter.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.Edgelabel.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.Noiselabel.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.GassianspinBox.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.TopleftLabel.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.TopRightLabel.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.BottomRightLabel.setVisible)
        self.checkBox_2.clicked['bool'].connect(self.BottomLeftLabel.setVisible)
        # 点击作业3checkbox显示控件信号
        self.checkBox_3.clicked['bool'].connect(self.dilationFunction.setVisible)
        self.checkBox_3.clicked['bool'].connect(self.erosionFunction.setVisible)
        self.checkBox_3.clicked['bool'].connect(self.openingFunction.setVisible)
        self.checkBox_3.clicked['bool'].connect(self.closingFunction.setVisible)
        self.checkBox_3.clicked['bool'].connect(self.TopleftLabel.setVisible)
        self.checkBox_3.clicked['bool'].connect(self.TopRightLabel.setVisible)
        self.checkBox_3.clicked['bool'].connect(self.BottomRightLabel.setVisible)
        self.checkBox_3.clicked['bool'].connect(self.BottomLeftLabel.setVisible)
        # 点击作业4checkbox显示控件信号
        self.checkBox_4.clicked['bool'].connect(self.SkeletonFunction.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.SkeletonRestoration.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.binaryImage.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.otherImage.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.DTLabel.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.TopleftLabel.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.TopRightLabel.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.BottomRightLabel.setVisible)
        self.checkBox_4.clicked['bool'].connect(self.BottomLeftLabel.setVisible)
        # 点击作业5checkbox显示控件信号
        self.checkBox_5.clicked['bool'].connect(self.GraydilationFunction.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.GrayerosionFunction.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.GrayopeningFunction.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.GrayclosingFunction.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.GraydilationFunctionSlider.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.GrayerosionFunctionSlider.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.GrayopeningFunctionSlider.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.GrayclosingFunctionSlider.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.TopleftLabel.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.TopRightLabel.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.BottomRightLabel.setVisible)
        self.checkBox_5.clicked['bool'].connect(self.BottomLeftLabel.setVisible)
        # 点击作业6checkbox显示控件信号
        self.checkBox_6.clicked['bool'].connect(self.EdgeDetection.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.Gradient.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.ConditionalDilation.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.ConditionalDilationSlider.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.GrayReconstructionLabel.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.OBRFunction.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.CBRFunction.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.OBRSpinBox.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.OBRSE.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.TopleftLabel.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.TopRightLabel.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.BottomRightLabel.setVisible)
        self.checkBox_6.clicked['bool'].connect(self.BottomLeftLabel.setVisible)
        # 打开文件夹信号
        self.fileOpenAction.triggered.connect(self.openOriginImg)
        # HW1 点击OSTU信号
        self.OSTUFunction.clicked.connect(self.homeWork1_OSTU)
        # HW1 点击Entropy信号
        self.binaryFunction.clicked.connect(self.homeWork1_Binary)
        # HW1 滑动条变化阈值二值化
        self.thresholdSlider.valueChanged.connect(self.homeWork1_BinaryChange)
        # 清除所有Label内容
        self.clearFunction.clicked.connect(self.clearLabel)
        # HW2 Roberts算子边缘检测
        self.RobertsOperator.clicked.connect(self.homeWork2_Roberts)
        # HW2 Prewitt算子边缘检测
        self.PrewittOperator.clicked.connect(self.homeWork2_Prewitt)
        # HW2 Sobel算子边缘检测
        self.SobelOperator.clicked.connect(self.homeWork2_Sobel)
        # HW2 Gaussian滤波
        self.GaussianFilter.clicked.connect(self.homeWork2_Gaussian)
        # HW2 Gaussian滤波数值变化
        self.GassianspinBox.valueChanged.connect(self.homeWork2_GaussianFilterChange)
        # HW2 Median滤波
        self.MedianFilter.clicked.connect(self.homeWork2_Median)
        # HW3 dilation运算
        self.dilationFunction.clicked.connect(self.homeWork3_Dilation)
        # HW3 erosion运算
        self.erosionFunction.clicked.connect(self.homeWork3_Erosion)
        # HW3 opening运算
        self.openingFunction.clicked.connect(self.homeWork3_Opening)
        # HW3 closing运算
        self.closingFunction.clicked.connect(self.homeWork3_Closing)
        # HW4 distanceTransform运算
        self.binaryImage.toggled.connect(lambda: self.homeWork4_DistanceTransform(self.binaryImage))
        self.otherImage.toggled.connect(lambda: self.homeWork4_DistanceTransform(self.otherImage))
        # HW4 skeleton运算
        self.SkeletonFunction.clicked.connect(self.homeWork4_Skeleton)
        # HW4 skeleton restoration运算
        self.SkeletonRestoration.clicked.connect(self.homeWork4_SkeletonRestoration)
        # HW5 Gray dilation运算
        self.GraydilationFunction.clicked.connect(self.homeWork5_Dilation)
        self.GraydilationFunctionSlider.valueChanged.connect(self.homeWork5_DilationChange)
        # HW5 Gray erosion运算
        self.GrayerosionFunction.clicked.connect(self.homeWork5_Erosion)
        self.GrayerosionFunctionSlider.valueChanged.connect(self.homeWork5_ErosionChange)
        # HW5 Gray opening运算
        self.GrayopeningFunction.clicked.connect(self.homeWork5_Opening)
        self.GrayopeningFunctionSlider.valueChanged.connect(self.homeWork5_OpeningChange)
        # HW5 Gray closing运算
        self.GrayclosingFunction.clicked.connect(self.homeWork5_Closing)
        self.GrayclosingFunctionSlider.valueChanged.connect(self.homeWork5_ClosingChange)
        # HW6 Morphological Edge Detection运算
        self.EdgeDetection.clicked.connect(self.homeWork6_MorEdgeDetection)
        # HW6 Morphological Gradient运算
        self.Gradient.clicked.connect(self.homeWork6_MorGradient)
        # HW6 Conditional Dilation运算
        self.ConditionalDilation.clicked.connect(self.homeWork6_ConditionalDilation)
        # HW6 Conditional Dilation Slider运算
        self.ConditionalDilationSlider.valueChanged.connect(self.homeWork6_ConditionalDilationSlider)
        # HW6 OBR运算
        self.OBRFunction.clicked.connect(self.homeWork6_OBR)
        # HW6 CBR运算
        self.CBRFunction.clicked.connect(self.homeWork6_CBR)
        # HW6 OBR运算变化
        self.OBRSpinBox.valueChanged.connect(self.homeWork6_OBRChange)

    # 槽函数——从文件夹中选取图片并将其显示在Label_1中
    def openOriginImg(self):
        self.file, ok = QFileDialog.getOpenFileName(self, "打开", "C:/Users/74706/Desktop/ImageProcess_test/",
                                                    "All Files(*);;JPG Files(*.jpg);;PNG Files(*.png);;TIF Files(*.tif)")
        img = cv2.imread(self.file)
        origin = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        temp_origin = QImage(origin[:], origin.shape[1], origin.shape[0], origin.shape[1] * 3, QImage.Format_RGB888)
        pixmap_origin = QPixmap.fromImage(temp_origin).scaled(self.label_width, self.label_height)
        self.resultLabel_1.setPixmap(pixmap_origin)
        self.TopleftLabel.setText("Origin")

    # 槽函数——清除所有Label中的内容
    def clearLabel(self):
        self.resultLabel_1.clear()
        self.resultLabel_2.clear()
        self.resultLabel_3.clear()
        self.resultLabel_4.clear()
        self.TopleftLabel.setText("1")
        self.TopRightLabel.setText("2")
        self.BottomLeftLabel.setText("3")
        self.BottomRightLabel.setText("4")
        self.file = ""

    # 槽函数——HW1 OSTU算法
    def homeWork1_OSTU(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, ostu = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        temp_ostu = QImage(ostu[:], ostu.shape[1], ostu.shape[0], ostu.shape[1], QImage.Format_Grayscale8)
        pixmap_ostu = QPixmap.fromImage(temp_ostu).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_ostu)
        self.TopRightLabel.setText("OSTU")
        self.OSTUThresholdShow.setText(str(ret))

    # 槽函数——HW1 最大熵算法
    def homeWork1_Binary(self):
        img = cv2.imread(self.file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        def calculate_current_entropy(hist, threshold):
            data_hist = hist.copy()
            background_sum = 0.
            target_sum = 0.
            for i in range(256):
                if i < threshold:  # 累积背景
                    background_sum += data_hist[i]
                else:  # 累积目标
                    target_sum += data_hist[i]
            background_ent = 0.
            target_ent = 0.
            for i in range(256):
                if i < threshold:  # 计算背景熵
                    if data_hist[i] == 0:
                        continue
                    ratio1 = data_hist[i] / background_sum
                    background_ent -= ratio1 * np.log2(ratio1)
                else:
                    if data_hist[i] == 0:
                        continue
                    ratio2 = data_hist[i] / target_sum
                    target_ent -= ratio2 * np.log2(ratio2)
            return target_ent + background_ent

        def max_entropy_segmentation(image):
            channels = [0]
            hist_size = [256]
            prange = [0, 256]
            hist = cv2.calcHist(image, channels, None, hist_size, prange)
            hist = np.reshape(hist, [-1])
            max_ent = 0.
            max_index = 0
            for i in range(256):
                cur_ent = calculate_current_entropy(hist, i)
                if cur_ent > max_ent:
                    max_ent = cur_ent
                    max_index = i
            ret, th = cv2.threshold(image, max_index, 255, cv2.THRESH_BINARY)
            self.EntropyThresholdShow.setText(str(ret))
            return th
        binary = max_entropy_segmentation(img)
        temp_binary = QImage(binary[:], binary.shape[1], binary.shape[0], binary.shape[1], QImage.Format_Grayscale8)
        pixmap_binary = QPixmap.fromImage(temp_binary).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_binary)
        self.BottomLeftLabel.setText("Entropy")

    # 槽函数——HW1 滑动改变阈值
    def homeWork1_BinaryChange(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        threshold = self.thresholdSlider.value()
        ret, binaryChange = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        temp_binaryChange = QImage(binaryChange[:], binaryChange.shape[1], binaryChange.shape[0], binaryChange.shape[1],
                                   QImage.Format_Grayscale8)
        pixmap_binaryChange = QPixmap.fromImage(temp_binaryChange).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_binaryChange)
        self.BottomRightLabel.setText("Change")
        self.ThresholdSliderShow.setText(str(threshold))

    # 槽函数——HW2 Roberts算子
    def homeWork2_Roberts(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
        kernely = np.array([[0, -1], [1, 0]], dtype=int)
        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        temp_Roberts = QImage(Roberts[:], Roberts.shape[1], Roberts.shape[0], Roberts.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_Roberts = QPixmap.fromImage(temp_Roberts).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_Roberts)
        self.TopRightLabel.setText("Roberts")

    # 槽函数——HW2 Prewitt算子
    def homeWork2_Prewitt(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
        kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
        x = cv2.filter2D(gray, cv2.CV_16S, kernelx)
        y = cv2.filter2D(gray, cv2.CV_16S, kernely)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        temp_Prewitt = QImage(Prewitt[:], Prewitt.shape[1], Prewitt.shape[0], Prewitt.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_Prewitt = QPixmap.fromImage(temp_Prewitt).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_Prewitt)
        self.BottomLeftLabel.setText("Prewitt")

    # 槽函数——HW2 Sobel算子
    def homeWork2_Sobel(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x = cv2.Sobel(gray, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(gray, cv2.CV_16S, 0, 1)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        temp_Sobel = QImage(Sobel[:], Sobel.shape[1], Sobel.shape[0], Sobel.shape[1],
                            QImage.Format_Grayscale8)
        pixmap_Sobel = QPixmap.fromImage(temp_Sobel).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_Sobel)
        self.BottomRightLabel.setText("Sobel")

    # 槽函数——HW2 Gaussian滤波
    def homeWork2_Gaussian(self):
        img = cv2.imread(self.file)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Gaussian = cv2.GaussianBlur(img1, (3, 3), 0)
        temp_Gaussian = QImage(Gaussian[:], Gaussian.shape[1], Gaussian.shape[0], Gaussian.shape[1] * 3,
                               QImage.Format_RGB888)
        pixmap_Gaussian = QPixmap.fromImage(temp_Gaussian).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_Gaussian)
        self.TopRightLabel.setText("Gaussian")

    # 槽函数——HW2 Gaussian滤波数值变化
    def homeWork2_GaussianFilterChange(self):
        img = cv2.imread(self.file)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Gaussian = cv2.GaussianBlur(img1, (self.GassianspinBox.value(), self.GassianspinBox.value()), 0)
        temp_Gaussian = QImage(Gaussian[:], Gaussian.shape[1], Gaussian.shape[0], Gaussian.shape[1] * 3,
                               QImage.Format_RGB888)
        pixmap_Gaussian = QPixmap.fromImage(temp_Gaussian).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_Gaussian)
        self.BottomRightLabel.setText("GaussianC")

    # 槽函数——HW2 Median滤波
    def homeWork2_Median(self):
        img = cv2.imread(self.file)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Median = cv2.medianBlur(img1, 3)
        temp_Median = QImage(Median[:], Median.shape[1], Median.shape[0], Median.shape[1] * 3,
                             QImage.Format_RGB888)
        pixmap_Median = QPixmap.fromImage(temp_Median).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_Median)
        self.BottomLeftLabel.setText("Median")

    # 槽函数——HW3 Dilation运算
    def homeWork3_Dilation(self):
        img = cv2.imread(self.file)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Dilation = cv2.dilate(img1, kernel=np.ones((10, 10), np.uint8))
        temp_Dilation = QImage(Dilation[:], Dilation.shape[1], Dilation.shape[0], Dilation.shape[1] * 3,
                               QImage.Format_RGB888)
        pixmap_Dilation = QPixmap.fromImage(temp_Dilation).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_Dilation)
        self.TopRightLabel.setText("Dilation")

    # 槽函数——HW3 Erosion运算
    def homeWork3_Erosion(self):
        img = cv2.imread(self.file)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Erosion = cv2.erode(img1, kernel=np.ones((5, 5), np.uint8))
        temp_Erosion = QImage(Erosion[:], Erosion.shape[1], Erosion.shape[0], Erosion.shape[1] * 3,
                              QImage.Format_RGB888)
        pixmap_Erosion = QPixmap.fromImage(temp_Erosion).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_Erosion)
        self.BottomLeftLabel.setText("Erosion")

    # 槽函数——HW3 Opening运算
    def homeWork3_Opening(self):
        img = cv2.imread(self.file)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Opening = cv2.morphologyEx(img1, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
        temp_Opening = QImage(Opening[:], Opening.shape[1], Opening.shape[0], Opening.shape[1] * 3,
                              QImage.Format_RGB888)
        pixmap_Opening = QPixmap.fromImage(temp_Opening).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_Opening)
        self.BottomRightLabel.setText("Opening")

    # 槽函数——HW3 Closing运算
    def homeWork3_Closing(self):
        img = cv2.imread(self.file)
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        Closing = cv2.morphologyEx(img1, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
        temp_Closing = QImage(Closing[:], Closing.shape[1], Closing.shape[0], Closing.shape[1] * 3,
                              QImage.Format_RGB888)
        pixmap_Closing = QPixmap.fromImage(temp_Closing).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_Closing)
        self.TopRightLabel.setText("Closing")

    # 槽函数——HW4 DistanceTransform运算
    def homeWork4_DistanceTransform(self, btn):
        img = cv2.imread(self.file, cv2.IMREAD_GRAYSCALE)
        if btn.text() == "二值化图像":
            if btn.isChecked() == True:
                pass
            else:
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if btn.text() == "其他图像":
            if btn.isChecked() == True:
                ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            else:
                pass
        dist_transform0 = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        dist_transform1 = cv2.convertScaleAbs(dist_transform0)
        dist_transform = cv2.normalize(dist_transform1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        temp_dist_transform = QImage(dist_transform[:], dist_transform.shape[1], dist_transform.shape[0],
                                     dist_transform.shape[1],
                                     QImage.Format_Grayscale8)
        pixmap_dist_transform = QPixmap.fromImage(temp_dist_transform).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_dist_transform)
        self.TopRightLabel.setText("DT")

    # 槽函数——HW4 Skeleton运算
    def homeWork4_Skeleton(self):
        img = cv2.imread(self.file, 0)
        ret, binary = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
        binary[binary == 255] = 1
        skeleton0 = morphology.skeletonize(binary)
        skeleton = skeleton0.astype(np.uint8) * 255
        cv2.imwrite("skeleton.jpg", skeleton)
        sk = cv2.imread("skeleton.jpg")
        img1 = cv2.cvtColor(sk, cv2.COLOR_BGR2GRAY)
        temp_skeleton = QImage(img1[:], img1.shape[1], img1.shape[0], img1.shape[1], QImage.Format_Grayscale8)
        pixmap_skeleton = QPixmap.fromImage(temp_skeleton).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_skeleton)
        self.BottomLeftLabel.setText("Skeleton")

    # 槽函数——HW4 Skeleton Restoration运算
    def homeWork4_SkeletonRestoration(self):
        img = cv2.imread(self.file, cv2.IMREAD_GRAYSCALE)
        dist_transform0 = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        dist_transform1 = cv2.convertScaleAbs(dist_transform0)
        dist_transform = cv2.normalize(dist_transform1, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        temp_dist_transform = QImage(dist_transform[:], dist_transform.shape[1], dist_transform.shape[0],
                                     dist_transform.shape[1],
                                     QImage.Format_Grayscale8)
        pixmap_dist_transform = QPixmap.fromImage(temp_dist_transform).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_dist_transform)
        self.BottomRightLabel.setText("SR")

    # 槽函数——HW5 GrayDilation运算
    def homeWork5_Dilation(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        GDilation = cv2.dilate(grey, kernel=np.ones((10, 10), np.uint8))
        temp_GDilation = QImage(GDilation[:], GDilation.shape[1], GDilation.shape[0], GDilation.shape[1],
                                QImage.Format_Grayscale8)
        pixmap_GDilation = QPixmap.fromImage(temp_GDilation).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_GDilation)
        self.TopRightLabel.setText("GDilation")

    def homeWork5_DilationChange(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = self.GraydilationFunctionSlider.value()
        GDilation = cv2.dilate(grey, kernel=np.ones((size, size), np.uint8))
        temp_GDilation = QImage(GDilation[:], GDilation.shape[1], GDilation.shape[0], GDilation.shape[1],
                                QImage.Format_Grayscale8)
        pixmap_GDilation = QPixmap.fromImage(temp_GDilation).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_GDilation)
        self.TopRightLabel.setText("GDilation")

    # 槽函数——HW5 Gray Erosion运算
    def homeWork5_Erosion(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        GErosion = cv2.erode(grey, kernel=np.ones((5, 5), np.uint8))
        temp_GErosion = QImage(GErosion[:], GErosion.shape[1], GErosion.shape[0], GErosion.shape[1],
                               QImage.Format_Grayscale8)
        pixmap_GErosion = QPixmap.fromImage(temp_GErosion).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_GErosion)
        self.BottomLeftLabel.setText("GErosion")

    def homeWork5_ErosionChange(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = self.GrayerosionFunctionSlider.value()
        GErosion = cv2.erode(grey, kernel=np.ones((size, size), np.uint8))
        temp_GErosion = QImage(GErosion[:], GErosion.shape[1], GErosion.shape[0], GErosion.shape[1],
                               QImage.Format_Grayscale8)
        pixmap_GErosion = QPixmap.fromImage(temp_GErosion).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_GErosion)
        self.BottomLeftLabel.setText("GErosion")

    # 槽函数——HW5 Gray Opening运算
    def homeWork5_Opening(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        GOpening = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel=np.ones((5, 5), np.uint8))
        temp_GOpening = QImage(GOpening[:], GOpening.shape[1], GOpening.shape[0], GOpening.shape[1],
                               QImage.Format_Grayscale8)
        pixmap_GOpening = QPixmap.fromImage(temp_GOpening).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_GOpening)
        self.BottomRightLabel.setText("GOpening")

    def homeWork5_OpeningChange(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = self.GrayopeningFunctionSlider.value()
        GOpening = cv2.morphologyEx(grey, cv2.MORPH_OPEN, kernel=np.ones((size, size), np.uint8))
        temp_GOpening = QImage(GOpening[:], GOpening.shape[1], GOpening.shape[0], GOpening.shape[1],
                               QImage.Format_Grayscale8)
        pixmap_GOpening = QPixmap.fromImage(temp_GOpening).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_GOpening)
        self.BottomRightLabel.setText("GOpening")

    # 槽函数——HW5 Gray Closing运算
    def homeWork5_Closing(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        GClosing = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel=np.ones((5, 5), np.uint8))
        temp_GClosing = QImage(GClosing[:], GClosing.shape[1], GClosing.shape[0], GClosing.shape[1],
                               QImage.Format_Grayscale8)
        pixmap_GClosing = QPixmap.fromImage(temp_GClosing).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_GClosing)
        self.TopRightLabel.setText("GClosing")

    def homeWork5_ClosingChange(self):
        img = cv2.imread(self.file)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = self.GrayclosingFunctionSlider.value()
        GClosing = cv2.morphologyEx(grey, cv2.MORPH_CLOSE, kernel=np.ones((size, size), np.uint8))
        temp_GClosing = QImage(GClosing[:], GClosing.shape[1], GClosing.shape[0], GClosing.shape[1],
                               QImage.Format_Grayscale8)
        pixmap_GClosing = QPixmap.fromImage(temp_GClosing).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_GClosing)
        self.TopRightLabel.setText("GClosing")

    # 槽函数——HW6 Morphological Edge Detection
    def homeWork6_MorEdgeDetection(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilation = cv2.dilate(gray, kernel=np.ones((5, 5), np.uint8))
        erosion = cv2.erode(gray, kernel=np.ones((5, 5), np.uint8))
        standard_result = cv2.subtract(dilation, erosion)
        external_result = cv2.subtract(dilation, gray)
        internal_result = cv2.subtract(gray, erosion)
        # standard边缘检测
        temp_standard_result = QImage(standard_result[:], standard_result.shape[1], standard_result.shape[0],
                                      standard_result.shape[1], QImage.Format_Grayscale8)
        pixmap_standard_result = QPixmap.fromImage(temp_standard_result).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_standard_result)
        self.TopRightLabel.setText("Standard")
        # external边缘检测
        temp_external_result = QImage(external_result[:], external_result.shape[1], external_result.shape[0],
                                      external_result.shape[1], QImage.Format_Grayscale8)
        pixmap_external_result = QPixmap.fromImage(temp_external_result).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_external_result)
        self.BottomLeftLabel.setText("External")
        # internal边缘检测
        temp_internal_result = QImage(internal_result[:], internal_result.shape[1], internal_result.shape[0],
                                      internal_result.shape[1], QImage.Format_Grayscale8)
        pixmap_internal_result = QPixmap.fromImage(temp_internal_result).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_internal_result)
        self.BottomRightLabel.setText("Internal")

    # 槽函数——HW6 Morphological Gradient
    def homeWork6_MorGradient(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dilation = cv2.dilate(gray, kernel=np.ones((5, 5), np.uint8))
        erosion = cv2.erode(gray, kernel=np.ones((5, 5), np.uint8))
        standard_result = cv2.subtract(dilation, erosion)
        external_result = cv2.subtract(dilation, gray)
        internal_result = cv2.subtract(gray, erosion)
        size = np.shape(standard_result)
        standard_gradient = np.zeros((size[0], size[1]), dtype=np.uint8)
        external_gradient = np.zeros((size[0], size[1]), dtype=np.uint8)
        internal_gradient = np.zeros((size[0], size[1]), dtype=np.uint8)
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                standard_gradient[i][j] = (1 / 2) * standard_result[i][j]
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                external_gradient[i][j] = (1 / 2) * external_result[i][j]
        for i in range(0, size[0]):
            for j in range(0, size[1]):
                internal_gradient[i][j] = (1 / 2) * internal_result[i][j]
        # standard梯度
        temp_standard_gradient = QImage(standard_gradient[:], standard_gradient.shape[1], standard_gradient.shape[0],
                                        standard_gradient.shape[1], QImage.Format_Grayscale8)
        pixmap_standard_gradient = QPixmap.fromImage(temp_standard_gradient).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_standard_gradient)
        self.TopRightLabel.setText("Standard")
        # external梯度
        temp_external_gradient = QImage(external_gradient[:], external_gradient.shape[1], external_gradient.shape[0],
                                        external_gradient.shape[1], QImage.Format_Grayscale8)
        pixmap_external_gradient = QPixmap.fromImage(temp_external_gradient).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_external_gradient)
        self.BottomLeftLabel.setText("External")
        # internal梯度
        temp_internal_gradient = QImage(internal_gradient[:], internal_gradient.shape[1], internal_gradient.shape[0],
                                        internal_gradient.shape[1], QImage.Format_Grayscale8)
        pixmap_internal_gradient = QPixmap.fromImage(temp_internal_gradient).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_internal_gradient)
        self.BottomRightLabel.setText("Internal")

    # 槽函数——HW6 Conditional Dilation
    def homeWork6_ConditionalDilation(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        x, y = gray.shape
        gray = np.array(gray, dtype=np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel=np.ones((25, 25), np.uint8))
        temp_opening = QImage(opening[:], opening.shape[1], opening.shape[0], opening.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_opening = QPixmap.fromImage(temp_opening).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_opening)
        self.BottomLeftLabel.setText("Marker")

        reconstruct = np.zeros([x, y], dtype=np.uint8)
        reconstruct = opening
        while True:
            opening = cv2.dilate(opening, kernel=np.ones((4, 4), np.uint8))
            opening = opening & gray
            if (opening == reconstruct).all():
                break
            reconstruct = opening
        temp_gray = QImage(gray[:], gray.shape[1], gray.shape[0], gray.shape[1], QImage.Format_Grayscale8)
        pixmap_gray = QPixmap.fromImage(temp_gray).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_gray)
        self.TopRightLabel.setText("Threshold")

        temp_reconstruct = QImage(reconstruct[:], reconstruct.shape[1], reconstruct.shape[0], reconstruct.shape[1],
                                  QImage.Format_Grayscale8)
        pixmap_reconstruct = QPixmap.fromImage(temp_reconstruct).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_reconstruct)
        self.BottomRightLabel.setText("Reconstruction")

    # 槽函数——HW6 Conditional Dilation Slider
    def homeWork6_ConditionalDilationSlider(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        x, y = gray.shape
        gray = np.array(gray, dtype=np.uint8)
        size = self.ConditionalDilationSlider.value()
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel=np.ones((size, size), np.uint8))
        temp_opening = QImage(opening[:], opening.shape[1], opening.shape[0], opening.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_opening = QPixmap.fromImage(temp_opening).scaled(self.label_width, self.label_height)
        self.resultLabel_3.setPixmap(pixmap_opening)
        self.BottomLeftLabel.setText("Marker")

        reconstruct = np.zeros([x, y], dtype=np.uint8)
        reconstruct = opening
        while True:
            opening = cv2.dilate(opening, kernel=np.ones((4, 4), np.uint8))
            opening = opening & gray
            if (opening == reconstruct).all():
                break
            reconstruct = opening
        temp_gray = QImage(gray[:], gray.shape[1], gray.shape[0], gray.shape[1], QImage.Format_Grayscale8)
        pixmap_gray = QPixmap.fromImage(temp_gray).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_gray)
        self.TopRightLabel.setText("Threshold")

        temp_reconstruct = QImage(reconstruct[:], reconstruct.shape[1], reconstruct.shape[0], reconstruct.shape[1],
                                  QImage.Format_Grayscale8)
        pixmap_reconstruct = QPixmap.fromImage(temp_reconstruct).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_reconstruct)
        self.BottomRightLabel.setText("Reconstruction")

    # 槽函数——HW6 OBR
    def homeWork6_OBR(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = gray.shape
        gray = np.array(gray, dtype=np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel=np.ones((20, 20), np.uint8))
        temp_opening = QImage(opening[:], opening.shape[1], opening.shape[0], opening.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_opening = QPixmap.fromImage(temp_opening).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_opening)
        self.TopRightLabel.setText("Marker")
        reconstruct = np.empty([x, y], dtype=np.uint8)
        diff = 0
        while True:
            opening = cv2.dilate(opening, kernel=np.ones((3, 3), np.uint8))
            w, h = opening.shape
            reconstruct = opening.copy()
            for i in range(w):
                for j in range(h):
                    if opening[i][j] > gray[i][j]:
                        reconstruct[i][j] = gray[i][j]
            diff += 1
            if diff > 20:
                break
        temp_reconstruct = QImage(reconstruct[:], reconstruct.shape[1], reconstruct.shape[0], reconstruct.shape[1],
                                  QImage.Format_Grayscale8)
        pixmap_reconstruct = QPixmap.fromImage(temp_reconstruct).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_reconstruct)
        self.BottomRightLabel.setText("Reconstruction")

    # 槽函数——HW6 OBR
    def homeWork6_CBR(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = gray.shape
        gray = np.array(gray, dtype=np.uint8)
        closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel=np.ones((20, 20), np.uint8))
        temp_closing = QImage(closing[:], closing.shape[1], closing.shape[0], closing.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_closing = QPixmap.fromImage(temp_closing).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_closing)
        self.TopRightLabel.setText("Marker")
        reconstruct = np.empty([x, y], dtype=np.uint8)
        diff = 0
        while True:
            closing = cv2.dilate(closing, kernel=np.ones((3, 3), np.uint8))
            w, h = closing.shape
            reconstruct = closing.copy()
            for i in range(w):
                for j in range(h):
                    if closing[i][j] > gray[i][j]:
                        reconstruct[i][j] = gray[i][j]
            diff += 1
            if diff > 20:
                break
        temp_reconstruct = QImage(reconstruct[:], reconstruct.shape[1], reconstruct.shape[0], reconstruct.shape[1],
                                  QImage.Format_Grayscale8)
        pixmap_reconstruct = QPixmap.fromImage(temp_reconstruct).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_reconstruct)
        self.BottomRightLabel.setText("Reconstruction")

    # 槽函数——HW6 OBR Change
    def homeWork6_OBRChange(self):
        img = cv2.imread(self.file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x, y = gray.shape
        gray = np.array(gray, dtype=np.uint8)
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel=np.ones((self.OBRSpinBox.value(), self.OBRSpinBox.value()), np.uint8))
        temp_opening = QImage(opening[:], opening.shape[1], opening.shape[0], opening.shape[1],
                              QImage.Format_Grayscale8)
        pixmap_opening = QPixmap.fromImage(temp_opening).scaled(self.label_width, self.label_height)
        self.resultLabel_2.setPixmap(pixmap_opening)
        self.TopRightLabel.setText("Marker")
        reconstruct = np.empty([x, y], dtype=np.uint8)
        diff = 0
        while True:
            opening = cv2.dilate(opening, kernel=np.ones((3, 3), np.uint8))
            w, h = opening.shape
            reconstruct = opening.copy()
            for i in range(w):
                for j in range(h):
                    if opening[i][j] > gray[i][j]:
                        reconstruct[i][j] = gray[i][j]
            diff += 1
            if diff > 20:
                break
        temp_reconstruct = QImage(reconstruct[:], reconstruct.shape[1], reconstruct.shape[0], reconstruct.shape[1],
                                  QImage.Format_Grayscale8)
        pixmap_reconstruct = QPixmap.fromImage(temp_reconstruct).scaled(self.label_width, self.label_height)
        self.resultLabel_4.setPixmap(pixmap_reconstruct)
        self.BottomRightLabel.setText("Reconstruction")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWin = MyMainWindow()
    myWin.show()
    sys.exit(app.exec_())
