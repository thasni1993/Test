# Using the dataset "Predict students' dropout and academic success" and predict the accuracy by training the data. The datset taken from "http://archive.ics.uci.edu/dataset/697/predict+students+dropout+and+academic+success"

#import the packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

#Load the dataset and then preprocess the data
stud = pd.read_csv('Student_Data.csv')
print(stud)

#It appears that there are no nulls or duplicates, but we can check and deal with them if necessary.
print(stud.isnull().sum())

#To check the duplicate values
print(stud.duplicated().sum())

#Only the target column is non-numeric which we can convert to numeric
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

# Split the dataset into features (X) and the target variable (y)
X = df.drop('Target', axis=1)
y = df['Target']

# Count the number of instances for each class
class_counts = df['Target'].value_counts()
print("Class Distribution:")
print(class_counts)

# Perform undersampling on the majority class
rus = RandomUnderSampler(random_state=42)
X_undersampled, y_undersampled = rus.fit_resample(X, y)

# Perform oversampling on the minority class
ros = RandomOverSampler(random_state=42)
X_oversampled, y_oversampled = ros.fit_resample(X, y)

# Perform SMOTE (Synthetic Minority Over-sampling Technique)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X, y)

# Calculate class distribution before and after resampling
class_counts_before = y.value_counts()
class_counts_undersampled = pd.Series(y_undersampled).value_counts()
class_counts_oversampled = pd.Series(y_oversampled).value_counts()
class_counts_smote = pd.Series(y_smote).value_counts()

# Plot class distribution before and after resampling
plt.figure(figsize=(16, 4))

plt.subplot(1, 4, 1)
plt.bar(class_counts_before.index, class_counts_before.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution Before Resampling')

plt.subplot(1, 4, 2)
plt.bar(class_counts_undersampled.index, class_counts_undersampled.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution After Undersampling')

plt.subplot(1, 4, 3)
plt.bar(class_counts_oversampled.index, class_counts_oversampled.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution After Oversampling')

plt.subplot(1, 4, 4)
plt.bar(class_counts_smote.index, class_counts_smote.values)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Class Distribution After SMOTE')

plt.tight_layout()
plt.show()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print('------------------------------------------------------------------------------')
print(X_test)
print('------------------------------------------------------------------------------')
print(y_train)
print('------------------------------------------------------------------------------')
print(y_test)

# Scale the features using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(X_train_scaled)
print('-------------------------------------------------------------------')
print(X_test_scaled)

# Train and evaluate the Decision Tree classifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train_scaled, y_train)
dt_pred = dt_clf.predict(X_test_scaled)
dt_accuracy = accuracy_score(y_test, dt_pred)
dt_cm = confusion_matrix(y_test, dt_pred)
print("Decision Tree Accuracy:", dt_accuracy)
print("Decision Tree Confusion Matrix:")
print(dt_cm)

# Plot the Decision Tree Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(dt_cm, annot=True, cmap="Blues", fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Train and evaluate the Support Vector Machine (SVM) classifier
svm_clf = SVC()
svm_clf.fit(X_train_scaled, y_train)
svm_pred = svm_clf.predict(X_test_scaled)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_cm = confusion_matrix(y_test, svm_pred)
print("SVM Accuracy:", svm_accuracy)
print("SVM Confusion Matrix:")
print(svm_cm)

# Plot the SVM Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(svm_cm, annot=True, cmap="Blues", fmt="d")
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

# Train and evaluate the Random Forest classifier
rf_clf = RandomForestClassifier()
rf_clf.fit(X_train_scaled, y_train)
rf_pred = rf_clf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
print("Random Forest Accuracy:", rf_accuracy)
print("Random Forest Confusion Matrix:")
print(rf_cm)

# Plot the Random Forest Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(rf_cm, annot=True, cmap="Blues", fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.show()

#Plotting these three classifier accuracy in one plot
# Accuracy values
accuracy_values = [dt_accuracy, svm_accuracy, rf_accuracy]

# Classifier names
classifier_names = ['Decision Tree', 'SVM', 'Random Forest']

# Plotting
plt.figure(figsize=(8, 6))
plt.bar(classifier_names, accuracy_values)
plt.title('Accuracy Comparison')
plt.xlabel('Classifier')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.show()

