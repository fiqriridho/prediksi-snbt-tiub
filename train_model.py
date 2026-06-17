import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
print("Loading dataset...")
df = pd.read_csv("dataset_final.csv")

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

# Inisialisasi dan Train Model KNN (Tuned: K=25, metric='euclidean', weights='distance')
print("Training KNN Model (K=25, metric='euclidean', weights='distance')...")
model = KNeighborsClassifier(n_neighbors=25, metric='euclidean', weights='distance')
model.fit(X_train_scaled, y_train)

# Prediksi untuk Evaluasi
y_pred = model.predict(X_test_scaled)

# Step 4: Cek Hasil Akurasi
print("\n=== EVALUASI MODEL KNN ===")
print(f"Accuracy:  {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"Precision: {precision_score(y_test, y_pred, pos_label='BERPELUANG') * 100:.2f}%")
print(f"Recall:    {recall_score(y_test, y_pred, pos_label='BERPELUANG') * 100:.2f}%")
print(f"F1-Score:  {f1_score(y_test, y_pred, pos_label='BERPELUANG') * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 5: Save Model Final
print("Saving scaler.pkl and model_knn.pkl...")
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(model, 'model_knn.pkl')
print("Model and scaler saved successfully!")
