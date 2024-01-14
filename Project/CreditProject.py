# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy
import sys
numpy.set_printoptions(threshold=sys.maxsize)

# Training & Validation
train_file_path = "C:\\Users\\Sydney\\Documents\\GradSchool\\CS_7050 Data_Warehousing_Mining\\Project\\GiveMeSomeCredit-training.csv"
train_data = pd.read_csv(train_file_path)

# Testing
test_file_path = 'C:\\Users\\Sydney\\Documents\\GradSchool\\CS_7050 Data_Warehousing_Mining\\Project\\GiveMeSomeCredit-testing.csv'
test_data = pd.read_csv(test_file_path)

# Explore the datasets
print("Training Dataset:\n", train_data.head())
print("\nTesting Dataset:\n", test_data.head())


# Split Feature & Target
X_train = train_data.drop('SeriousDlqin2yrs', axis=1) #features
y_train = train_data['SeriousDlqin2yrs'] #targetn variable

# Split training into training & validation
X_train_split, X_val, y_train_split, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val)

# Create and train a logistic regression model on the training set

model = LogisticRegression(random_state=42)
model.fit(X_train_scaled, y_train_split)


y_val_pred = model.predict(X_val_scaled)

# Evaluate the model on the validation set
accuracy_val = accuracy_score(y_val, y_val_pred)
conf_matrix_val = confusion_matrix(y_val, y_val_pred)
classification_rep_val = classification_report(y_val, y_val_pred)

print("Validation Set Evaluation:")
print(f"Accuracy: {accuracy_val}")
print("Confusion Matrix:\n", conf_matrix_val)
print("Classification Report:\n", classification_rep_val)
#-------------------------------------------------------------


X_test = test_data.drop('SeriousDlqin2yrs', axis=1)# features only in testing set
X_test_scaled = scaler.transform(X_test)
y_test_pred = model.predict(X_test_scaled)

# Display the predictions
print("Predictions on Testing Set:")
print(y_test_pred)



#--------------------------------------------------------------------
# tak in user input to see if they are a risk 


# Process the user input with same scaler used above 
def preprocess_data(user_input, scaler):
    input_data = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_data)
    return input_scaled

# Function to get user input
def get_user_input():
    user_input = {}

    # same feature names as in database excel file
    feature_names = ['Unnamed: 0','RevolvingUtilizationOfUnsecuredLines', 'age', 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']

    for feature in feature_names:
        value = float(input(f"Enter value for {feature}: "))
        user_input[feature] = value

    return user_input

#execute
user_input_data = get_user_input()
user_input_scaled = preprocess_data(user_input_data, scaler)

# Make prediction using the trained model
prediction = model.predict(user_input_scaled)

# Display the prediction
print(f"The model predicts: {prediction}")
