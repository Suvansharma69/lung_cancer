import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

# 📥 Load Dataset
data = pd.read_csv('your_dataset.csv')  # Replace with actual dataset name
print("📜 Dataset loaded successfully.")

# 🧩 Define Categorical Features
categorical_features = [
    'gender', 'country', 'diagnosis_date', 'cancer_stage',
    'family_history', 'smoking_status', 'treatment_type', 'end_treatment_date'
]

# 🛡️ Handle Missing Target Labels
if data['survived'].isnull().sum() > 0:
    data = data.dropna(subset=['survived'])

# 🚑 Handle Missing Values in Features
for col in data.columns:
    if data[col].isnull().sum() > 0:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

# 🎯 Features and Labels
X = data.drop(['id', 'survived'], axis=1)
y = data['survived']

# ✨ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🧠 Initialize CatBoost Classifier
model = CatBoostClassifier(
    iterations=500,
    depth=6,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=42,
    cat_features=categorical_features,
    verbose=20
)

# 🚀 Train Model
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 📈 Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Test Accuracy: {accuracy:.4f}")

# 💾 Save Trained Model
model.save_model('lung_cancer_model.cbm')
print("💾 Model saved successfully.")
