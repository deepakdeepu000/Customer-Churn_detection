# Import all needed packages
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso
from numpy import absolute, mean, std
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



# Load the dataset
df = pd.read_excel("/content/drive/MyDrive/E Commerce Dataset.xlsx", sheet_name=1)
df.head()

# Remove unnecessary columns
df = df.drop(['CustomerID'], axis=1)

# Check for duplicate rows
df.duplicated().any()

# Define numerical and categorical columns
cat_columns = df.select_dtypes(include="O").columns
num_columns = []
for col in df.columns:
    if col not in cat_columns:
        num_columns.append(col)

print("Numerical columns: ", num_columns)
print("Categorical columns: ", cat_columns)


# Look into the unique values of each categorical column
for col in cat_columns:
    print(col, df[col].unique())



# Merge categories in categorical columns
df.replace(['Mobile Phone', 'Credit Card', 'Mobile Phone'], ['Phone', 'CC', 'Mobile'], inplace=True)

# Check for NULL values
df.isna().sum()



def train_and_test(X, y, classifier, test_size, enable_print=True):
    # Get categorical and numerical columns
    cat_columns = X.select_dtypes(include="O").columns
    num_columns = [col for col in X.columns if col not in cat_columns]
    
    # Encode categorical columns using OneHotEncoder
    categorical_col = Pipeline(steps=[
        ('encoding', OneHotEncoder())
    ])
    
    # Fill numerical NULL values with the mean value of the relevant column
    numerical_col = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy="mean")),
        ('scaler', StandardScaler())
    ])
    
    # Transform data
    transformer = ColumnTransformer(transformers=[
        ('categorical_col', categorical_col, cat_columns),    
        ('numerical_col', numerical_col, num_columns)
    ])
    
    # Create model
    model = Pipeline([
        ('transformer', transformer),
        ('classifier', classifier)
    ])

    # Prepare training and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Fit the model
    model.fit(X_train, y_train)
    
    # Calculate the training data accuracy and confusion matrix
    pred = model.predict(X_train)
    accuracy = accuracy_score(y_train, pred)
    matrix = confusion_matrix(y_train, pred)
    if enable_print:
        print('Train confusion matrix:\n', matrix)
        print('Train accuracy: ', accuracy * 100, '%')
    
    # Calculate the testing data accuracy and confusion matrix
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    matrix = confusion_matrix(y_test, pred)
    if enable_print: 
        print('Test confusion matrix:\n', matrix)
        print('Test accuracy: ', accuracy * 100, '%')
        print("=====================================================================")
  
    return accuracy

# Prepare X and y
X = df.drop(columns=["Churn"])
y = df["Churn"]
test_size = 0.2

# Training the model

# Logistic Classifier
print("========================== Logistic Classifier ==========================")
model = train_and_test(X, y, LogisticRegression(solver='liblinear'), test_size)

# Decision Tree Classifier
print("========================== DecisionTreeClassifier ==========================")
model = train_and_test(X, y, DecisionTreeClassifier(random_state=42), test_size)

# XGBClassifier
print("========================== XGBClassifier ==========================")
model = train_and_test(X, y, XGBClassifier(), test_size)

# Hypertuning XGBClassifier params

# Tree max depth hypertuning
max_depth = []
accuracies = []

for i in range(4, 30, 2):
    grid = {'max_depth': i}
    clf = XGBClassifier()
    clf.set_params(**grid)
    accuracy = train_and_test(X, y, clf, test_size, False)
    
    max_depth.append(i)
    accuracies.append(accuracy)

plt.title("Max Depth hypertuning")
plt.xlabel("XGBClassifier Max Depth")
plt.ylabel("Test Accuracy")
plt.plot(max_depth, accuracies, 'b')
plt.grid()
plt.show()


plt.plot(max_depth, accuracies, marker='o', linestyle='-', color='green')
plt.title("Max Depth Hypertuning")
plt.xlabel("XGBClassifier Max Depth")
plt.ylabel("Test Accuracy")
plt.grid(True)
plt.show()




# Learning rate hypertuning
learning_rates = []
accuracies = []

for i in range(0, 10, 1):
    eta = i / 10
    grid = {'max_depth': 12, 'eta': eta}
    clf = XGBClassifier()
    clf.set_params(**grid)
    accuracy = train_and_test(X, y, clf, test_size, False)
    
    learning_rates.append(i)
    accuracies.append(accuracy)

plt.title("Learning Rate hypertuning")
plt.xlabel("XGBClassifier Learning Rate")
plt.ylabel("Test Accuracy")
plt.plot(learning_rates, accuracies, 'b')
plt.grid()



# Number of estimators hypertuning
n_estimators = []
accuracies = []

for i in range(1, 1000, 50):
    grid = {'max_depth': 12, 'eta': 0.4, 'n_estimators': i}
    clf = XGBClassifier()
    clf.set_params(**grid)
    accuracy = train_and_test(X, y, clf, test_size, False)
    
    n_estimators.append(i)
    accuracies.append(accuracy)

plt.title("Number of estimators hypertuning")
plt.xlabel("XGBClassifier Number of estimators")
plt.ylabel("Test Accuracy")
plt.plot(n_estimators, accuracies, 'g')
plt.grid()


sns.heatmap(np.array(n_estimators).reshape(-1, 1), annot=True, cmap='coolwarm', xticklabels=['Test Accuracy'], yticklabels=accuracies)
plt.title("Max Depth Hypertuning")
plt.xlabel("Metric")
plt.ylabel("XGBClassifier Max Depth")
plt.show()



# Final model with the hypertuned params
print("========================== XGBClassifier with hypertuned params ==========================")
grid = {'max_depth': 16, 'eta': 0.4, 'alpha': 0, 'lambda': 1, 'n_estimators': 400}
clf = XGBClassifier()
clf.set_params(**grid)
accuracy = train_and_test(X, y, clf, test_size)


