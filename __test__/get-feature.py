import joblib

model = joblib.load("../models/best_rf_model.joblib")

if hasattr(model, "feature_names_in_"):
    print("Features:", model.feature_names_in_)
else:
    print("The model does not have feature names information.")