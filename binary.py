import numpy as np # Numerical computations
import pandas as pd # Data handling and analysis
import matplotlib.pyplot as plt # Plotting graphs
# Scikit-learn utilities
from sklearn.model_selection import train_test_split # Dataset splitting
from sklearn.preprocessing import MinMaxScaler # Feature scaling
from sklearn.tree import DecisionTreeClassifier # Decision Tree model
from sklearn.metrics import ( accuracy_score, precision_score, recall_score,
f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report,
roc_curve, roc_auc_score, auc)
df = pd.read_csv('heart.csv') # Read dataset from CSV file
X = df.drop(columns=['output']) # X contains all feature columns
y = df['output'] # y contains the target class (0: Normal, 1: Abnormal)
# 80% training data, 20% testing data
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,
random_state=0)
# Initialize MinMaxScaler
scaler = MinMaxScaler()
# Fit scaler only on training data to avoid data leakage
X_train = scaler.fit_transform(X_train)
# Apply the same transformation to test data
X_test = scaler.transform(X_test)
# Create Decision Tree model using Gini index
dt_model = DecisionTreeClassifier(random_state=42)
# Train the model using scaled training data
dt_model.fit(X_train, y_train.ravel())
# Predict class labels for test samples
preds = dt_model.predict(X_test)
# Print evaluation metrics
print("Accuracy :", accuracy_score(y_test, preds))
print("Precision:", precision_score(y_test, preds))
print("Recall :", recall_score(y_test, preds))
print("F1 Score :", f1_score(y_test, preds))
# Define class labels
target_names = ['Normal', 'Abnormal']
# Print detailed classification report
print(classification_report(y_test, preds, target_names=target_names))
# Compute confusion matrix
cm = confusion_matrix(y_test, preds)
# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=target_names
)
disp.plot()
plt.title("Confusion Matrix - Decision Tree")
plt.show()
# Compute ROC-AUC score
roc_auc = roc_auc_score(y_test, preds)
# Compute False Positive Rate and True Positive Rate
fpr, tpr, thresholds = roc_curve(y_test, preds)
# Compute AUC value
auc_score = auc(fpr, tpr)
# Plot ROC Curve
plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree')
plt.legend(loc='best')
plt.show()
