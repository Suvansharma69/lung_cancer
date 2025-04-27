import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import joblib

# 📥 Load Dataset
data = pd.read_csv('your_dataset.csv')  # <-- Hide real filename
print("📜 Dataset columns:", data.columns)

# 🧩 Categorical Features
categorical_features = [
    'gender', 'country', 'diagnosis_date', 'cancer_stage',
    'family_history', 'smoking_status', 'treatment_type', 'end_treatment_date'
]
print("🧩 Categorical Features:", categorical_features)

# 🎯 Check survived values
print("🧹 Unique survived values:", data['survived'].unique())

# 🛡️ Check if survived has NaNs
if data['survived'].isnull().sum() > 0:
    data = data.dropna(subset=['survived'])

# 🚑 Handle Missing Features
if data.isnull().sum().sum() > 0:
    print("⚠️ Warning: Missing values detected! Filling with mode/median.")
    for col in data.columns:
        if data[col].dtype == 'object':
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

# 🔥 Features and Labels
X = data.drop(['id', 'survived'], axis=1)
y = data['survived']

print(f"✅ Final Samples for Training: {len(X)}")

# ✨ Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 🧠 Initialize CatBoost
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

# 🚀 Train
model.fit(X_train, y_train, eval_set=(X_test, y_test))

# 📈 Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Final Test Accuracy: {accuracy:.4f}")

# 💾 Save
model.save_model('model.cbm')  # <-- Hide real filename
print("💾 Model saved successfully.")
