import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("dataset_final.csv")

# Step 1: Hitung Rata-rata Tertimbang (Weighting)
print("Recalculating weighted average (RATA2)...")
df['RATA2'] = (
    df['PU'] * 1.0 +
    df['PPU'] * 1.0 +
    df['PBM'] * 1.0 +
    df['PK'] * 1.2 +
    df['LBI'] * 0.9 +
    df['LBI2'] * 0.9 +
    df['PM'] * 1.3
) / 7.3

# Step 2: Buat Ulang Label berdasarkan threshold 620
print("Re-labeling STATUS...")
df['STATUS'] = df['RATA2'].apply(
    lambda x: 'BERPELUANG'
    if x >= 620
    else 'KURANG_BERPELUANG'
)

# Fitur dan Target
features = ['PU', 'PPU', 'PBM', 'PK', 'LBI', 'LBI2', 'PM']
X = df[features]
y = df['STATUS']

# Train-Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit Scaler pada data latih
print("Fitting StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Inisialisasi dan Train Model KNN (K=7, matching original model settings)
print("Training KNN Model (K=7)...")
model = KNeighborsClassifier(n_neighbors=7, metric='minkowski')
model.fit(X_train_scaled, y_train)

# Prediksi untuk Evaluasi
y_pred = model.predict(X_test_scaled)

# Step 4: Cek Hasil Akurasi
print("\n=== EVALUASI MODEL KNN ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='BERPELUANG') * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label='BERPELUANG') * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Save Model Final
print("Saving scaler.pkl and model_knn.pkl...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model_knn.pkl')
print("Model and scaler saved successfully!")
