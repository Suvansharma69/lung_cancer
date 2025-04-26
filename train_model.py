import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Load the dataset
df = pd.read_csv('D:/lung cancer gpt/your_actual_dataset.csv')  # Replace with your actual dataset file path

# Check the columns
print("Columns in dataset:", df.columns)

# Convert categorical columns to category type
categorical_columns = ['gender', 'country', 'cancer_stage', 'family_history', 'smoking_status', 'treatment_type']
for col in categorical_columns:
    df[col] = df[col].astype('category')

# Feature engineering: Drop non-numerical columns and target
X = df.drop(['id', 'diagnosis_date', 'end_treatment_date', 'survived'], axis=1)  # Features
y = df['survived']  # Target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoost model
model = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=6, cat_features=[0, 1, 2, 3, 4, 5], verbose=100)

# Train the model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Output the results
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Save the trained model
model.save_model('models/catboost_lung_model.pkl')
