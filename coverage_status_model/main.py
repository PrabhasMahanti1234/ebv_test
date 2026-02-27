import joblib
import pandas as pd
import numpy as np
import json
from datetime import datetime

def predict_coverage_status(saved_model, payer_name, state_name, acronym, expansion, explanation):
    """
    Predict coverage status using the saved model.
    Returns a dictionary with all prediction details.
    """
    # Prepare input
    combined_text = f"{expansion} {explanation}".lower().strip()
    acronym_clean = acronym.strip().upper()
    
    # Transform text
    text_features = saved_model['tfidf'].transform([combined_text]).toarray()
    
    # Transform categorical
    cat_df = pd.DataFrame({
        'PAYER NAME': [payer_name],
        'STATE NAME': [state_name],
        'ACRONYM_clean': [acronym_clean]
    })
    cat_features = saved_model['onehot'].transform(cat_df)
    
    # Combine
    X = np.hstack([text_features, cat_features])
    
    # Predict
    prediction_encoded = saved_model['model'].predict(X)
    prediction = saved_model['label_encoder'].inverse_transform(prediction_encoded)
    
    # Get prediction probabilities if available
    probabilities = {}
    if hasattr(saved_model['model'], 'predict_proba'):
        proba = saved_model['model'].predict_proba(X)[0]
        classes = saved_model['label_encoder'].classes_
        probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba)}
    
    # Build result dictionary
    result = {
        "prediction": {
            "coverage_status": prediction[0],
            "confidence": probabilities
        },
        "input": {
            "payer_name": payer_name,
            "state_name": state_name,
            "acronym": acronym,
            "expansion": expansion,
            "explanation": explanation
        },
        "metadata": {
            "model_type": type(saved_model['model']).__name__,
            "timestamp": datetime.now().isoformat(),
            "total_features": X.shape[1]
        },
        "model_evaluation": saved_model.get('metrics', {})
    }
    
    return result


# Load the saved model from models directory
saved_model = joblib.load(r"c:\Users\VH0000812\Desktop\Coverage\models\coverage_model.pkl")

# Make a prediction
result = predict_coverage_status(
    saved_model,
    payer_name="cook childrens health plan",
    state_name="Texas",
    acronym="NPD",
    expansion="Non-preferred drug.",
    explanation="Refers to brand-name drugs on a formulary that are not in the preferred tier, usually resulting in higher copayments or coinsurance compared to preferred brand-name drugs.")
# Output as formatted JSON
print(json.dumps(result, indent=2))

	