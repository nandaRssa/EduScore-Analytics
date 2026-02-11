# -*- coding: utf-8 -*-
"""Final Tugas Kelompok 3_IT_BOOTCAMP - Versi Diperbaiki"""

# --- Business Understanding ---
# Dataset ini bertujuan untuk mengidentifikasi variabel yang memengaruhi skor ujian siswa
# Untuk membantu sekolah atau pembuat kebijakan meningkatkan performa akademik siswa.

# --- Import Library ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
import pickle

# --- Load Dataset ---
df = pd.read_csv('exams.csv')

# --- Data Overview ---
df.rename(columns={
    'race/ethnicity': 'race_ethnicity',
    'parental level of education': 'parental_level_of_education',
    'test preparation course': 'test_preparation_course',
    'math score': 'math_score',
    'reading score': 'reading_score',
    'writing score': 'writing_score'
}, inplace=True)

target_column_name = 'math_score'
categorical_features = ['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

# --- EDA: Korelasi ---
plt.figure(figsize=(10,5))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Variable')
plt.show()

# --- Distribusi Skor Matematika ---
plt.figure(figsize=(10,6))
sns.histplot(df['math_score'], kde=True, bins=20, color='blue')
plt.title('Distribusi Skor Matematika')
plt.xlabel('Skor Matematika')
plt.ylabel('Frekuensi')
plt.grid(axis='y', linestyle='--')
plt.show()

# --- Visualisasi Kategori ---
for col in categorical_features:
    plt.figure(figsize=(10,5))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribusi {col}')
    plt.show()

# --- Cek Duplikasi & Missing Values ---
df_cleaned = df.drop_duplicates()
missing_data_info = pd.DataFrame({
    'missing_value': df_cleaned.isnull().sum(),
    'percentage_missing': df_cleaned.isnull().sum()/len(df_cleaned)*100
})
print(missing_data_info)

# --- Outlier (Boxplot) ---
numeric_cols = df.select_dtypes(include=['int64']).columns
fig, axs = plt.subplots(ncols=len(numeric_cols), figsize=(5*len(numeric_cols),5))
for i, col in enumerate(numeric_cols):
    sns.boxplot(y=df[col], ax=axs[i])
    axs[i].set_title(f'Box Plot for {col}')
plt.tight_layout()
plt.show()

# --- Modeling Bagian 1: Klasifikasi Skor Matematika ---
df_class = df_cleaned.copy()
bins = [0, 59, 79, 100]
labels = ['Rendah', 'Sedang', 'Tinggi']
df_class['score_class'] = pd.cut(df_class['math_score'], bins=bins, labels=labels, include_lowest=True, right=True)

X_class = df_class[categorical_features]
y_class = df_class['score_class']

preprocessor_class = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
])

rf_class_pipeline = Pipeline([
    ('preprocessor', preprocessor_class),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42, stratify=y_class
)

rf_class_pipeline.fit(X_train_class, y_train_class)
y_pred_class = rf_class_pipeline.predict(X_test_class)

print(classification_report(y_test_class, y_pred_class))
cm_class = confusion_matrix(y_test_class, y_pred_class, labels=rf_class_pipeline.named_steps['classifier'].classes_)
ConfusionMatrixDisplay(cm_class, display_labels=rf_class_pipeline.named_steps['classifier'].classes_).plot(cmap='Blues')
plt.show()

# --- Modeling Bagian 2: Status Kelulusan ---
df_status = df_cleaned.copy()
df_status['status'] = df_status['math_score'].apply(lambda x: 'Lulus' if x >= 60 else 'Tidak Lulus')

X_status = df_status[categorical_features]
y_status = df_status['status']

rf_status_pipeline = Pipeline([
    ('preprocessor', preprocessor_class),
    ('classifier', RandomForestClassifier(random_state=42))
])

X_train_status, X_test_status, y_train_status, y_test_status = train_test_split(
    X_status, y_status, test_size=0.2, random_state=42, stratify=y_status
)

rf_status_pipeline.fit(X_train_status, y_train_status)
y_pred_status = rf_status_pipeline.predict(X_test_status)
print(classification_report(y_test_status, y_pred_status, target_names=sorted(y_status.unique())))
ConfusionMatrixDisplay(confusion_matrix(y_test_status.map({'Lulus':1,'Tidak Lulus':0}), 
                                       y_pred_status.map({'Lulus':1,'Tidak Lulus':0}))).plot(cmap='Greens')
plt.show()

# --- Modeling Bagian 3: Regresi Math Score ---
df_reg = df_cleaned.copy()
X_reg_categorical_features = categorical_features
X_reg_numerical_features = ['reading_score', 'writing_score']
X_reg = df_reg[X_reg_categorical_features + X_reg_numerical_features]
y_reg = df_reg['math_score']

preprocessor_reg = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), X_reg_categorical_features),
    ('num', StandardScaler(), X_reg_numerical_features)
])

lr_reg_pipeline = Pipeline([
    ('preprocessor', preprocessor_reg),
    ('regressor', LinearRegression())
])

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

lr_reg_pipeline.fit(X_train_reg, y_train_reg)

# --- Evaluasi Regresi ---
y_pred_reg = lr_reg_pipeline.predict(X_test_reg)
mse_reg = mean_squared_error(y_test_reg, y_pred_reg)
r2_reg = r2_score(y_test_reg, y_pred_reg)

print(f"Mean Squared Error: {mse_reg:.2f}")
print(f"R2 Score: {r2_reg:.2f}")

# --- Cross Validation R2 ---
scores = cross_val_score(lr_reg_pipeline, X_reg, y_reg, cv=5, scoring='r2')
print("R2 Cross Validation (5-fold):", scores.mean())

# --- Simpan Model ---
pickle.dump(lr_reg_pipeline, open("model.pkl", "wb"))

# --- Testing Data Dummy Regresi ---
data_dummy_reg = pd.DataFrame({
    'gender': ['female', 'male'],
    'race_ethnicity': ['group C', 'group A'],
    'parental_level_of_education': ["bachelor's degree", 'high school'],
    'lunch': ['standard', 'free/reduced'],
    'test_preparation_course': ['none', 'completed'],
    'reading_score': [70, 55],
    'writing_score': [75, 50]
})

dummy_pred = lr_reg_pipeline.predict(data_dummy_reg)
data_dummy_reg['Predicted Math Score'] = np.round(dummy_pred, 2)
print(data_dummy_reg)
