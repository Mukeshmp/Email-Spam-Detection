﻿# Email-Spam-Detection

The provided solution for the *E-mail Spam Detection* project consists of a GUI-based application implemented using Python, PyQt5 for the interface, and machine learning with Logistic Regression for classification.

### Key components of the solution:

1. **Data Preprocessing**:
   - The dataset is loaded, containing emails with categories labeled as "spam" or "ham" (not spam). The data is cleaned, converting empty values to a blank string, and spam/ham labels are converted to numerical values (spam = 0, ham = 1).
   - The dataset is split into training and testing sets using an 80-20 split.

2. **Feature Extraction**:
   - TF-IDF vectorization is used to convert email text into numerical features, which capture the importance of terms while ignoring common words (using stop words).

3. **Model Training**:
   - Logistic Regression is used to train the model on the transformed TF-IDF features.
   - The model is evaluated on both the training and testing data, and accuracy is printed for both.

4. **GUI Application**:
   - A simple graphical interface is built using PyQt5. The user can input email text into a text area, and upon clicking the "Test" button, the input email is classified as either "SPAM" or "NOT SPAM" based on the trained model.
   - The classification result is displayed on the screen with bold text, either in red (for spam) or green (for not spam).

### Results:
The model's performance is displayed via the accuracy on training and testing data.

This approach provides an interactive tool where users can test email messages and get immediate feedback on whether they are likely spam.
