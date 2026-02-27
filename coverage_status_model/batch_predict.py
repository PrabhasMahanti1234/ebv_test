"""
Batch Prediction Script
========================
Predict coverage status for multiple acronyms with the same payer.
"""

import joblib
import numpy as np
import pandas as pd
import json

def load_model():
    """Load the saved model."""
    model_path = r"c:\Users\VH0000812\Desktop\Coverage\models\coverage_model.pkl"
    model_data = joblib.load(model_path)
    return model_data

def predict_batch(model_data, payer_name, state_name, acronym_data):
    """
    Predict coverage status for multiple acronyms.
    
    Parameters:
    -----------
    payer_name : str
        Payer name (same for all)
    state_name : str
        State name (same for all)
    acronym_data : list of dict
        List of {'acronym': str, 'expansion': str, 'explanation': str}
    
    Returns:
    --------
    list of predictions with confidence scores
    """
    model = model_data['model']
    tfidf = model_data['tfidf']
    onehot = model_data['onehot']
    label_encoder = model_data['label_encoder']
    
    results = []
    
    for item in acronym_data:
        acronym = item['acronym']
        expansion = item.get('expansion', '')
        explanation = item.get('explanation', '')
        
        # Prepare features
        combined_text = f"{expansion} {explanation}".lower().strip()
        text_features = tfidf.transform([combined_text]).toarray()
        
        cat_df = pd.DataFrame({
            'PAYER NAME': [payer_name],
            'STATE NAME': [state_name],
            'ACRONYM_clean': [acronym.upper()]
        })
        cat_features = onehot.transform(cat_df)
        X = np.hstack([text_features, cat_features])
        
        # Predict
        prediction = model.predict(X)
        result = label_encoder.inverse_transform(prediction)[0]
        
        # Get confidence scores
        confidence = {}
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            for cls, prob in zip(label_encoder.classes_, proba):
                confidence[cls] = round(prob * 100, 2)
        
        results.append({
            'acronym': acronym,
            'expansion': expansion,
            'prediction': result,
            'confidence': confidence.get(result, 0),
            'all_scores': confidence
        })
    
    return results


if __name__ == "__main__":
    # Load model
    model_data = load_model()
    
    # === CONFIGURE YOUR INPUT HERE ===
    
    # Payer and State (same for all acronyms)
    PAYER_NAME = "CenCal"
    STATE_NAME = "California"
    
    # Multiple acronyms with their expansion and explanation
    ACRONYM_DATA = [
        {
            "acronym": "AL",
            "expansion": "Age Limit",
            "explanation": "Minimum and maximum age requirements set by an insurance policy or plan."
        },
        {
            "acronym": "PA",
            "expansion": "Prior Authorization",
            "explanation": "You need to get approval from the plan before filling prescription."
        },
        {
            "acronym": "QL",
            "expansion": "Quantity Limit",
            "explanation": "Limits on the amount of medication you can get at one time."
        },
        {
            "acronym": "ST",
            "expansion": "Step Therapy",
            "explanation": "You must try a less expensive drug before the plan covers the prescribed drug."
        },
        {
            "acronym": "NC",
            "expansion": "Not Covered",
            "explanation": "This drug is not covered by the plan."
        }
    ]
    
    # === END CONFIGURATION ===
    
    print("=" * 70)
    print(f"BATCH PREDICTIONS FOR: {PAYER_NAME} ({STATE_NAME})")
    print("=" * 70)
    
    # Run predictions
    results = predict_batch(model_data, PAYER_NAME, STATE_NAME, ACRONYM_DATA)
    
    # Display results
    print(f"\n{'Acronym':<10} {'Expansion':<25} {'Prediction':<25} {'Confidence':<10}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['acronym']:<10} {r['expansion']:<25} {r['prediction']:<25} {r['confidence']:.2f}%")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    from collections import Counter
    status_counts = Counter([r['prediction'] for r in results])
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    
    # Output as JSON
    print("\n" + "=" * 70)
    print("JSON OUTPUT")
    print("=" * 70)
    
    output = {
        "payer_name": PAYER_NAME,
        "state_name": STATE_NAME,
        "predictions": results
    }
    print(json.dumps(output, indent=2))
