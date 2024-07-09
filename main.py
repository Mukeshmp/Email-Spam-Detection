from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtGui import *
import sys
# For ML
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class MyWindow(QMainWindow):
    def __init__(self):
        super(MyWindow, self).__init__()
        self.initUI()

    def button_clicked(self):
        data = self.textArea.toPlainText()
        input_your_mail = [data]
        input_data_features = feature_extraction.transform(input_your_mail)
        prediction = model.predict(input_data_features)
        if prediction[0] != 1:
            self.text.setText("SPAM")
            self.text.move(440, 490)
            self.text.setStyleSheet('font-weight: bold; color : Red;')
            self.text.adjustSize()
        else:
            self.text.setText("NOT SPAM")
            self.text.move(410, 490)
            self.text.setStyleSheet('font-weight: bold; color : Green;')
            self.text.adjustSize()

    def initUI(self):
        self.setGeometry(500, 200, 1000, 650)
        self.setWindowTitle("Email Spam Detection")

        self.label = QtWidgets.QLabel(self)
        self.label.setText("Enter Text Here : ")
        self.label.setStyleSheet("font-weight: bold")
        self.label.adjustSize()
        self.label.move(250, 65)

        self.textArea = QtWidgets.QPlainTextEdit(self)
        self.textArea.resize(500, 300)
        # self.textArea.adjustSize()
        self.textArea.move(250, 100)

        self.b1 = QtWidgets.QPushButton(self)
        self.b1.setText("Test")
        self.b1.clicked.connect(self.button_clicked)
        self.b1.move(450, 430)
        # self.b1.setGeometry(450, 430, 60, 30)
        # self.b1.setStyleSheet("font-weight: bold")
        self.b1.setStyleSheet('QPushButton {font-weight: bold; color: blue; }')

        self.text = QtWidgets.QLabel(self)
        self.text.move(425, 490)
        self.text.setFont(QFont('Times', 20))


def window():
    app = QApplication(sys.argv)
    win = MyWindow()
    win.show()
    sys.exit(app.exec_())


df = pd.read_csv("C:\\Users\\mukes\\OneDrive\\Desktop\\mail_data.csv")
data = df.where((pd.notnull(df)),  '')
data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category', ] = 1
x = data['Message']
y = data['Category']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=3)
feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

x_train_features = feature_extraction.fit_transform(x_train)
x_test_features = feature_extraction.transform(x_test)

y_train = y_train.astype('int')
y_test = y_test.astype('int')
model = LogisticRegression()
model.fit(x_train_features, y_train)
prediction_on_training_data = model.predict(x_train_features)
accuracy_on_training_data = accuracy_score(y_train, prediction_on_training_data)
print('Acc on training data : ', accuracy_on_training_data)
prediction_on_training_data = model.predict(x_test_features)
accuracy_on_training_data = accuracy_score(y_test, prediction_on_training_data)
print('Acc on test data : ', accuracy_on_training_data)

window()



# Hey, you won $400000 prize, please contact 1234 for claiming? -- Spam
# Dear Mukesh,	logo Recruiter from Flight To Sucess Immigration Llp is actively hiring for Multiple Roles and your profile seems to be a good match.
