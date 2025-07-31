# 📧 Email Phishing Detection (Machine Learning Project)

This project is a **machine learning model** that detects whether an email is **phishing** or **safe** based on various **numeric features** extracted from the email.

It uses **Logistic Regression** and **Gaussian Naive Bayes** for classification.

---

## 📂 Dataset

The dataset contains emails with **numeric features**, such as:

- `num_words` → Total words in the email
- `num_unique_words` → Unique words count
- `num_stopwords` → Stopwords count
- `num_links` → Number of links
- `num_unique_domains` → Unique domains in links
- `num_email_addresses` → Email addresses count
- `num_spelling_errors` → Spelling mistakes
- `num_urgent_keywords` → Count of urgent words like _urgent_, _win_, _click_
- `label` → **0 = Safe email, 1 = Phishing email**

---

## 🛠️ Tech Stack

- **Python 3**
- **Pandas, NumPy** → Data processing
- **Matplotlib, Seaborn** → Visualization
- **Scikit-learn** → ML models and evaluation

---

## ⚡ Project Workflow

1. **Import Libraries**
2. **Load Dataset** (emails.csv)
3. **Exploratory Data Analysis (EDA)**
   - Class distribution plot
   - Correlation heatmap
4. **Preprocessing**
   - Define features (X) and target (y)
5. **Train-Test Split**
6. **Train Models**
   - Logistic Regression
   - Gaussian Naive Bayes
7. **Model Evaluation**
   - Accuracy
   - Confusion Matrix
   - Classification Report
8. **Make Predictions** for new email feature vectors

---

## 📊 Example Output

----- Logistic Regression Performance -----
Accuracy: 0.93
Confusion Matrix:
[[480  12]
 [ 25 483]]
Classification Report:
              precision    recall  f1-score   support
           0       0.95      0.98      0.97       492
           1       0.98      0.95      0.96       508

----- Gaussian Naive Bayes Performance -----
Accuracy: 0.91
...

---

## 🧪 Predicting a New Email

## Example feature vector:
 [num_words, num_unique_words, num_stopwords, num_links,
  num_unique_domains, num_email_addresses, num_spelling_errors, num_urgent_keywords]

new_email_features = [[50, 40, 10, 2, 1, 0, 3, 1]]
prediction = lr_model.predict(new_email_features)

print("Prediction:", "Phishing" if prediction[0] == 1 else "Safe")

---

## 📌 How to Run

1. Clone the repository or download the project
2. Install required libraries:

pip install pandas numpy seaborn matplotlib scikit-learn

3. Run the script:

python email_phishing_detection.py

---

## 🎯 Results

- Achieved **90–95% accuracy** on test data
- Logistic Regression performed slightly better than Naive Bayes
