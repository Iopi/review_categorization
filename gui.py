import os
import pickle

import pandas as pd
from PyQt5 import QtCore, QtGui, QtWidgets

from enums.language_enum import Language
from preprocessing import preprocessing_methods
from models.saved_model import SavedModels
import util
from models.classifier_model import Classifier
import constants
from enums.sentiment_enum import Sentiment


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(818, 473)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.sentence_text_edit = QtWidgets.QPlainTextEdit(self.centralwidget)
        self.sentence_text_edit.setGeometry(QtCore.QRect(80, 80, 561, 81))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.sentence_text_edit.setFont(font)
        self.sentence_text_edit.setPlainText("")
        self.sentence_text_edit.setObjectName("sentence_text_edit")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(90, 60, 47, 13))
        self.label.setObjectName("label")
        self.language_combo_box = QtWidgets.QComboBox(self.centralwidget)
        self.language_combo_box.setGeometry(QtCore.QRect(680, 110, 81, 22))
        self.language_combo_box.setObjectName("language_combo_box")
        self.language_combo_box.addItem("")
        self.language_combo_box.addItem("")
        self.language_combo_box.addItem("")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(670, 60, 111, 16))
        self.label_2.setObjectName("label_2")
        self.run_button = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.run_classification())
        self.run_button.setGeometry(QtCore.QRect(360, 190, 111, 41))
        self.run_button.setObjectName("run_button")
        self.save_button = QtWidgets.QPushButton(self.centralwidget, clicked=lambda: self.save_classifier())
        self.save_button.setGeometry(QtCore.QRect(500, 190, 80, 41))
        self.save_button.setObjectName("save_button")
        self.result_table = QtWidgets.QTableWidget(self.centralwidget)
        self.result_table.setEnabled(False)
        self.result_table.setGeometry(QtCore.QRect(80, 290, 664, 86))
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.result_table.sizePolicy().hasHeightForWidth())
        self.result_table.setSizePolicy(sizePolicy)
        self.result_table.setMaximumSize(QtCore.QSize(664, 86))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setStrikeOut(False)
        font.setKerning(False)
        self.result_table.setFont(font)
        self.result_table.setFocusPolicy(QtCore.Qt.NoFocus)
        self.result_table.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.result_table.setAutoFillBackground(False)
        self.result_table.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.result_table.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.result_table.setSizeAdjustPolicy(QtWidgets.QAbstractScrollArea.AdjustIgnored)
        self.result_table.setAlternatingRowColors(True)
        self.result_table.setShowGrid(True)
        self.result_table.setWordWrap(True)
        self.result_table.setObjectName("result_table")
        self.result_table.setColumnCount(9)
        self.result_table.setRowCount(2)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(True)
        item.setFont(font)
        self.result_table.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setItalic(True)
        item.setFont(font)
        self.result_table.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setWeight(75)
        item.setFont(font)
        self.result_table.setHorizontalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 4, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 5, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 6, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 7, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(0, 8, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.result_table.setItem(1, 0, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        self.result_table.setItem(1, 1, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(1, 2, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(1, 3, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(1, 4, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(1, 5, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(1, 6, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(1, 7, item)
        item = QtWidgets.QTableWidgetItem()
        item.setTextAlignment(QtCore.Qt.AlignCenter)
        font = QtGui.QFont()
        font.setPointSize(10)
        item.setFont(font)
        self.result_table.setItem(1, 8, item)
        self.result_table.horizontalHeader().setVisible(True)
        self.result_table.horizontalHeader().setCascadingSectionResizes(True)
        self.result_table.horizontalHeader().setDefaultSectionSize(65)
        self.result_table.horizontalHeader().setHighlightSections(True)
        self.result_table.horizontalHeader().setMinimumSectionSize(10)
        self.result_table.horizontalHeader().setSortIndicatorShown(True)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.result_table.verticalHeader().setVisible(True)
        self.result_table.verticalHeader().setDefaultSectionSize(30)
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(190, 250, 450, 20))
        font = QtGui.QFont()
        font.setPointSize(13)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 818, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.device = util.device_recognition()
        self.models = SavedModels("cs")
        self.prepare_classifiers()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def run_classification(self):
        self.clean_table()
        # sentence preprocessing
        sentence = self.sentence_text_edit.toPlainText()
        if sentence == "":
            return
        self.label_3.setText("Sentence preprocessing...")
        self.label_3.repaint()
        sent_language = self.language_combo_box.currentText().upper()
        sentence = preprocessing_methods.split_line(sentence.lower(), Language[sent_language].value)
        words = sentence.split(" ")
        preprocessing_methods.remove_bad_words([words], Language[sent_language].value)
        target_language = Language.CZECH.name

        # train data preprocessing
        self.label_3.setText("Training reviews preprocessing...")
        self.label_3.repaint()
        train_reviews = self.models.load_reviews(Language[target_language].value)

        # vector models preparing
        self.label_3.setText("Loading vector models...")
        self.label_3.repaint()
        target_model = self.models.prepare_vec_model(Language[target_language].value)
        if Language[sent_language].value != Language[target_language].value:
            source_model = self.models.prepare_vec_model(Language[sent_language].value)
        else:
            source_model = target_model

        # transform matrix computing
        trans_matrix = None
        transform_method = "orto1"
        if Language[sent_language].value != Language[target_language].value:
            self.label_3.setText("Transformation matrix computing...")
            self.label_3.repaint()
            trans_matrix = self.models.compute_transform_matrix(transform_method, Language[target_language].value,
                                                                Language[sent_language].value)

        classifier_method = "lstm"
        self.classification_process(classifier_method, train_reviews, trans_matrix, target_model, source_model,
                                    Language[sent_language].value, Language[target_language].value, words)
        self.label_3.setText(f"All done")
        self.label_3.repaint()

    def save_classifier(self):
        classifier_method = "lstm"
        with open(constants.CLASSIFIER_LSTM_CS, 'wb') as f:
            pickle.dump(self.classifiers[classifier_method], f)

    def load_classifier(self, classifier_method):
        with open(constants.CLASSIFIER_LSTM_CS, 'rb') as f:
            self.classifiers[classifier_method] = pickle.load(f)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label.setText(_translate("MainWindow", "Sentence:"))
        self.language_combo_box.setItemText(0, _translate("MainWindow", "czech"))
        self.language_combo_box.setItemText(1, _translate("MainWindow", "english"))
        self.language_combo_box.setItemText(2, _translate("MainWindow", "german"))
        self.label_2.setText(_translate("MainWindow", "Language of sentence:"))
        self.run_button.setText(_translate("MainWindow", "Run"))
        self.save_button.setText(_translate("MainWindow", "Save"))
        item = self.result_table.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Sentiment"))
        item = self.result_table.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Probability"))
        item = self.result_table.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "General"))
        item = self.result_table.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Food"))
        item = self.result_table.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Drink"))
        item = self.result_table.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Staff"))
        item = self.result_table.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Speed"))
        item = self.result_table.horizontalHeaderItem(5)
        item.setText(_translate("MainWindow", "Cleaness"))
        item = self.result_table.horizontalHeaderItem(6)
        item.setText(_translate("MainWindow", "Prices"))
        item = self.result_table.horizontalHeaderItem(7)
        item.setText(_translate("MainWindow", "Environ."))
        item = self.result_table.horizontalHeaderItem(8)
        item.setText(_translate("MainWindow", "Occup."))
        __sortingEnabled = self.result_table.isSortingEnabled()
        self.result_table.setSortingEnabled(False)
        item = self.result_table.item(0, 0)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 1)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 2)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 3)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 4)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 5)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 6)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 7)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(0, 8)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 0)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 1)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 2)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 3)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 4)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 5)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 6)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 7)
        item.setText(_translate("MainWindow", "-"))
        item = self.result_table.item(1, 8)
        item.setText(_translate("MainWindow", "-"))
        self.result_table.setSortingEnabled(__sortingEnabled)
        # self.label_3.setText(_translate("MainWindow", "Loading..."))

    def prepare_classifiers(self):
        self.classifiers = dict()
        if os.path.exists(constants.CLASSIFIER_LSTM_CS):
            self.load_classifier("lstm")
        else:
            self.classifiers["lstm"] = Classifier("lstm")

        if os.path.exists(constants.CLASSIFIER_CNN_CS):
            self.load_classifier("cnn")
        else:
            self.classifiers["cnn"] = Classifier("cnn")


    def classification_process(self, classifier_method, train_reviews, trans_matrix, target_model, source_model,
                               sent_language, target_language, words):
        index = 0
        max_sen_len = train_reviews.tokens.map(len).max()
        if len(words) > max_sen_len:
            self.label_3.setText(
                f"Count of sentence words after preprocessing ({len(words)}) can not be higher then {max_sen_len}.")
            self.label_3.repaint()
            return

        for category_name, cat_models in self.classifiers[classifier_method].categories.items():
            if cat_models['existence'] is None:
                self.label_3.setText(f"Training {classifier_method} - existence for category {category_name}...")
                self.label_3.repaint()
                temp_data = train_reviews.copy()
                preprocessing_methods.map_annotated(temp_data)
                cat_models['existence'] = self.classifiers[classifier_method].train_model(classifier_method, temp_data,
                                                                                          trans_matrix, target_model,
                                                                                          self.device, sent_language,
                                                                                          target_language,
                                                                                          category_name,
                                                                                          max_sen_len, self.models)
            self.label_3.setText(f"Classification for category {constants.CATEGORIES[index]}...")
            self.label_3.repaint()
            result = self.classifiers[classifier_method].test_model(classifier_method,
                                                                    cat_models['existence'], source_model, self.device,
                                                                    max_sen_len, words)

            if result >= 0.5:
                if cat_models['sentiment'] is None:
                    self.label_3.setText(f"Training {classifier_method} - sentiment for category {category_name}...")
                    self.label_3.repaint()
                    temp_data = train_reviews.copy()
                    preprocessing_methods.map_sentiment(temp_data)
                    temp_data = temp_data[temp_data[category_name] != 2]
                    cat_models['sentiment'] = self.classifiers[classifier_method].train_model(classifier_method,
                                                                                              temp_data,
                                                                                              trans_matrix,
                                                                                              target_model,
                                                                                              self.device,
                                                                                              sent_language,
                                                                                              target_language,
                                                                                              category_name,
                                                                                              max_sen_len, self.models)
                self.label_3.setText(f"Classification for category {constants.CATEGORIES[index]}...")
                self.label_3.repaint()
                result = self.classifiers[classifier_method].test_model(classifier_method,
                                                                        cat_models['sentiment'], source_model,
                                                                        self.device, max_sen_len, words)

                if result >= 0.5:
                    self.result_table.item(0, index).setText(Sentiment.POSITIVE.value)
                    self.result_table.item(1, index).setText(str(round(result, 2)))
                else:
                    self.result_table.item(0, index).setText(Sentiment.NEGATIVE.value)
                    self.result_table.item(1, index).setText(str(round(1 - result, 2)))
                self.result_table.repaint()
            else:
                self.result_table.item(0, index).setText("x")
                self.result_table.item(1, index).setText("x")
                self.result_table.repaint()
            index += 1

    def clean_table(self):
        for row in range(self.result_table.rowCount()):
            for col in range(self.result_table.columnCount()):
                item = self.result_table.item(row, col)
                item.setText("-")
        self.result_table.repaint()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
