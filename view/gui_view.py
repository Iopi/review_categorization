import os
import pickle

from PyQt5 import QtCore, QtGui, QtWidgets

import constants
import util
from view import app_output
from model.enums.language_enum import Language
from model.enums.sentiment_enum import Sentiment
from model.gui_model.classifier_model import Classifier
from model.gui_model.saved_model import SavedModels
from controller import preprocessing


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        '''
        Set up of user interface
        :param MainWindow: Main window of ui
        :return:
        '''
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(818, 473)
        MainWindow.setMinimumSize(818, 473)
        MainWindow.setMaximumSize(818, 473)
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
        self.models = SavedModels()

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def run_classification(self):
        '''
        Action after trigger run button.
        Runs the classification.
        :return:
        '''
        self.clean_table()
        # sentence preprocessing
        sentence = self.sentence_text_edit.toPlainText()
        if sentence == "":
            return
        self.label_3.setText("Sentence preprocessing...")
        self.label_3.repaint()
        sent_language = self.language_combo_box.currentText().upper()
        sentence = preprocessing.split_line(sentence.lower(), Language[sent_language].value)
        words = sentence.split(" ")
        preprocessing.remove_bad_words([words], Language[sent_language].value)
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
        if Language[sent_language].value != Language[target_language].value:
            transform_method = constants.DEFAULT_TRANS_METHOD
            self.label_3.setText("Transformation matrix computing...")
            self.label_3.repaint()
            trans_matrix = self.models.compute_transform_matrix(transform_method, Language[target_language].value,
                                                                Language[sent_language].value)

        classifier_method = "lstm"
        self.classification_process(classifier_method, train_reviews, trans_matrix, target_model, source_model,
                                    Language[sent_language].value, Language[target_language].value, words)
        self.label_3.setText(f"All done")
        self.label_3.repaint()

    def save_classifier(self, filename, classifier):
        '''
        Save classifier object to file
        :param filename: filename of saving file
        :param classifier: classifier object to be saved
        :return:
        '''
        with open(filename, 'wb') as f:
            pickle.dump(classifier, f)

    def load_classifier(self, filename):
        '''
        Load classifier object from file
        :param filename: filename of loading file
        :return:
        '''
        with open(filename, 'rb') as f:
            classifier = pickle.load(f)
        return classifier

    def retranslateUi(self, MainWindow):
        '''
        Specifies component names
        :param MainWindow: Main window of ui
        :return:
        '''
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Customer feedback categorization demonstrator"))
        self.label.setText(_translate("MainWindow", "Text:"))
        self.language_combo_box.setItemText(0, _translate("MainWindow", "czech"))
        self.language_combo_box.setItemText(1, _translate("MainWindow", "english"))
        self.language_combo_box.setItemText(2, _translate("MainWindow", "german"))
        self.label_2.setText(_translate("MainWindow", "Language of text:"))
        self.run_button.setText(_translate("MainWindow", "Run"))
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

    def classification_process(self, classifier_method, train_reviews, trans_matrix, target_model, source_model,
                               sent_language, target_language, words):
        '''

        :param classifier_method:
        :param train_reviews:
        :param trans_matrix:
        :param target_model:
        :param source_model:
        :param sent_language:
        :param target_language:
        :param words:
        :return:
        '''
        index = 0
        max_sen_len = train_reviews.tokens.map(len).max()
        if len(words) > max_sen_len:
            self.label_3.setText(
                f"Count of sentence words after preprocessing ({len(words)}) can not be higher then {max_sen_len}.")
            self.label_3.repaint()
            return

        for category_name in constants.CATEGORIES:
            classifier = self.load_models(classifier_method, category_name, train_reviews, trans_matrix, target_model,
                                          sent_language, target_language, max_sen_len)

            self.label_3.setText(f"Classification for category {constants.CATEGORIES[index]}...")
            self.label_3.repaint()
            result = classifier.test_model(classifier_method,
                                           classifier.category_models['existence'], source_model, self.device,
                                           max_sen_len, words)

            if result >= 0.5:

                self.label_3.setText(f"Classification for category {constants.CATEGORIES[index]}...")
                self.label_3.repaint()
                result = classifier.test_model(classifier_method,
                                               classifier.category_models['sentiment'], source_model,
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

    def load_models(self, classifier_method, category_name, train_reviews, trans_matrix, target_model,
                    sent_language, target_language, max_sen_len):
        if classifier_method == "lstm":
            models_filename = f"{constants.CLASSIFIER_LSTM}{category_name}_{self.device}_{target_language}_{sent_language}.bin"
        else:
            app_output.exception(f"Unknown classifier method {classifier_method}")

        # if exists - load
        if os.path.exists(models_filename):
            classifier = self.load_classifier(models_filename)

        # if not - create and save
        else:
            if not os.path.exists(constants.CLASSIFIER_FOLDER):
                os.makedirs(constants.CLASSIFIER_FOLDER)

            self.label_3.setText(f"Training {classifier_method} - existence for category {category_name}...")
            self.label_3.repaint()
            temp_data = train_reviews.copy()
            preprocessing.map_category_existence(temp_data)
            classifier = Classifier(classifier_method)
            classifier.category_models['existence'] = classifier.train_model(classifier_method, temp_data,
                                                                             trans_matrix, target_model,
                                                                             self.device, sent_language,
                                                                             target_language,
                                                                             category_name,
                                                                             max_sen_len, self.models)
            self.label_3.setText(f"Training {classifier_method} - sentiment for category {category_name}...")
            self.label_3.repaint()
            temp_data = train_reviews.copy()
            preprocessing.map_sentiment(temp_data)
            temp_data = temp_data[temp_data[category_name] != 2]
            classifier.category_models['sentiment'] = classifier.train_model(classifier_method,
                                                                             temp_data,
                                                                             trans_matrix,
                                                                             target_model,
                                                                             self.device,
                                                                             sent_language,
                                                                             target_language,
                                                                             category_name,
                                                                             max_sen_len, self.models)

            self.save_classifier(models_filename, classifier)

        return classifier


def run_gui():
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
