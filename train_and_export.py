# ===========================================================
# train_and_export.py — Train and Export Predictive Model
# ===========================================================
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

# ===========================================================
# 1️⃣ Load Dataset
# ===========================================================
file_path = "predictive_maintenance.csv"   # keep your dataset in this same folder
df = pd.read_csv(file_path)

print("✅ Dataset Loaded Successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# ===========================================================
# 2️⃣ Identify Target
# ===========================================================
target_candidates = ['Target', 'Failure Type', 'Failure', 'Machine failure']
target_column = None
for col in target_candidates:
    if col in df.columns:
        target_column = col
        break
if target_column is None:
    target_column = df.columns[-1]

print(f"🎯 Target Column: {target_column}")

# Split data
X = df.drop(columns=[target_column])
y = df[target_column]

if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# ===========================================================
# 3️⃣ Preprocessing
# ===========================================================
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"🔢 Numeric Features: {numeric_features}")
print(f"🔤 Categorical Features: {categorical_features}")

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# ===========================================================
# 4️⃣ Split Train/Test
# ===========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"📊 Train Size: {X_train.shape[0]} | Test Size: {X_test.shape[0]}")

# Preprocess
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# ===========================================================
# 5️⃣ Train 3 Models (Logistic, Decision Tree, Random Forest)
# ===========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_prep, y_train)
    y_pred = model.predict(X_test_prep)
    acc = accuracy_score(y_test, y_pred)
    cv = cross_val_score(model, X_train_prep, y_train, cv=5).mean()
    print(f"✅ {name} → Accuracy: {acc:.4f} | CV Mean: {cv:.4f}")
    results[name] = (acc, model)

# ===========================================================
# 6️⃣ Select Best Model
# ===========================================================
best_model_name = max(results, key=lambda k: results[k][0])
best_model = results[best_model_name][1]

print(f"\n🏆 Best Model Selected: {best_model_name}")

# ===========================================================
# 7️⃣ Export Model & Preprocessor to Flask Folder
# ===========================================================
export_path = os.path.dirname(os.path.abspath(__file__))

joblib.dump(best_model, os.path.join(export_path, "best_predictive_model.pkl"))
joblib.dump(preprocessor, os.path.join(export_path, "preprocessor.pkl"))

print(f"\n💾 Model saved as: {os.path.join(export_path, 'best_predictive_model.pkl')}")
print(f"💾 Preprocessor saved as: {os.path.join(export_path, 'preprocessor.pkl')}")
print("\n🎉 Export Completed Successfully!")
