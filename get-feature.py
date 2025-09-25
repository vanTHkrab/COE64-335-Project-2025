import joblib

model = joblib.load("models/best_rf_model.joblib")

# ถ้า Pipeline มีขั้นตอน preprocessing เช่น ColumnTransformer
if hasattr(model, "feature_names_in_"):
    print("Features:", model.feature_names_in_)
else:
    print("โมเดลนี้ไม่เก็บชื่อ feature ไว้ (เช็คตอน preprocess)")