# 🫁 Lung Cancer Survival Prediction System

A machine learning-powered application that predicts the likelihood of lung cancer patient survival based on clinical and demographic details. This project combines data preprocessing, CatBoost model training, and a modern graphical user interface (GUI) to deliver predictions in an accessible format.

---

## 🔍 Project Overview

This system:

- Trains a robust classification model using the [CatBoost](https://catboost.ai/) algorithm.
- Handles preprocessing, missing values, and categorical variables efficiently.
- Offers an intuitive GUI for medical professionals to input patient data and receive survival predictions.
- Includes calendar date selection, dark mode, tooltips, and progress animations.

---

## 📁 File Structure

| File/Folder                          | Description                                           |
|-------------------------------------|-------------------------------------------------------|
| `train_model.py`                    | Script for preprocessing data and training the model |
| `gui_predict.py`                    | GUI application for making predictions               |
| `lung_cancer_survival_model.cbm`    | Trained CatBoost model file                          |
| `README.md`                         | Project documentation (this file)                    |

---

## 🧠 Model Details

- **Algorithm**: CatBoostClassifier
- **Hyperparameters**:
  - Iterations: 500
  - Depth: 6
  - Learning rate: 0.1
- **Preprocessing**:
  - Missing values filled with median/mode
  - Date fields converted to number of days
  - Categorical features passed natively to CatBoost

---

## 🧪 Features Used

- Age, gender, country
- Diagnosis date and treatment end date
- Cancer stage and treatment type
- Family history and smoking status
- Comorbidities (hypertension, asthma, cirrhosis, other cancer)
- BMI and cholesterol level

---

## 🎯 Prediction Output

The GUI returns a result such as:

✅ **Patient SURVIVED**  
or  
❌ **Patient DID NOT SURVIVE**

It also shows a progress bar and animated feedback during prediction.

---

## 🖥️ GUI Features

- Fully interactive Tkinter interface
- Dropdowns for categorical fields (no typing needed)
- Date pickers for calendar-based input
- Dark Mode toggle
- Reset Button to clear form
- Tooltips for field explanations
- Voice-ready visual feedback (optionally extendable)

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
