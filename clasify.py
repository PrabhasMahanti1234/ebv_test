# import json
# import os
# import pandas as pd
# import numpy as np
# from datetime import datetime
# import joblib
# import logging

# # Setup logging
# logger = logging.getLogger(__name__)

# # ============================
# # ✅ Coverage Status ML Model Loader (one-time)
# # ============================

# _COVERAGE_MODEL = None

# def load_coverage_model():
#     """
#     Loads the ML coverage model once and caches it.
#     """
#     global _COVERAGE_MODEL
#     if _COVERAGE_MODEL is None:
#         # Use absolute path to avoid relative path issues
#         base_dir = os.path.dirname(os.path.abspath(__file__))
#         model_path = os.path.join(base_dir, "models", "coverage_model.pkl")
        
#         if not os.path.exists(model_path):
#             logger.error(f"Coverage model not found at: {model_path}")
#             return None
            
#         try:
#             _COVERAGE_MODEL = joblib.load(model_path)
#             logger.info("Successfully loaded coverage model.")
#         except Exception as e:
#             logger.error(f"Error loading coverage model: {e}")
#             return None
#     return _COVERAGE_MODEL

# def predict_coverage_status(saved_model, payer_name, state_name, acronym, expansion, explanation):
#     """
#     Predict coverage status using the saved model for a single input.
#     Returns a dictionary with all prediction details.
#     """
#     if saved_model is None:
#         return None

#     # Prepare input
#     combined_text = f"{expansion or ''} {explanation or ''}".lower().strip()
#     acronym_clean = (acronym or "").strip().upper()

#     # Transform text
#     text_features = saved_model['tfidf'].transform([combined_text]).toarray()

#     # Transform categorical
#     cat_df = pd.DataFrame({
#         'PAYER NAME': [payer_name or ""],
#         'STATE NAME': [state_name or ""],
#         'ACRONYM_clean': [acronym_clean]
#     })
#     cat_features = saved_model['onehot'].transform(cat_df)

#     # Combine
#     X = np.hstack([text_features, cat_features])

#     # Predict
#     prediction_encoded = saved_model['model'].predict(X)
#     prediction = saved_model['label_encoder'].inverse_transform(prediction_encoded)

#     # Get prediction probabilities if available
#     probabilities = {}
#     if hasattr(saved_model['model'], 'predict_proba'):
#         proba = saved_model['model'].predict_proba(X)[0]
#         classes = saved_model['label_encoder'].classes_
#         probabilities = {cls: round(float(prob), 4) for cls, prob in zip(classes, proba)}

#     # Build result dictionary
#     result = {
#         "prediction": {
#             "coverage_status": prediction[0],
#             "confidence": probabilities
#         },
#         "input": {
#             "payer_name": payer_name,
#             "state_name": state_name,
#             "acronym": acronym,
#             "expansion": expansion,
#             "explanation": explanation
#         },
#         "metadata": {
#             "model_type": type(saved_model['model']).__name__,
#             "timestamp": datetime.now().isoformat(),
#             "total_features": X.shape[1]
#         },
#         "model_evaluation": saved_model.get('metrics', {})
#     }

#     return result

# def predict_coverage_status_batch(saved_model, inputs):
#     """
#     Predict coverage status for a batch of inputs using the saved model.
#     'inputs' should be a list of dictionaries, each containing:
#     payer_name, state_name, acronym, expansion, explanation
#     """
#     if not inputs or saved_model is None:
#         return []

#     # 1. Prepare text features
#     combined_texts = [f"{inp.get('expansion') or ''} {inp.get('explanation') or ''}".lower().strip() for inp in inputs]
#     text_features = saved_model['tfidf'].transform(combined_texts).toarray()

#     # 2. Prepare categorical features
#     cat_data = {
#         'PAYER NAME': [inp.get('payer_name') or "" for inp in inputs],
#         'STATE NAME': [inp.get('state_name') or "" for inp in inputs],
#         'ACRONYM_clean': [(inp.get('acronym') or "").strip().upper() for inp in inputs]
#     }
#     cat_df = pd.DataFrame(cat_data)
#     cat_features = saved_model['onehot'].transform(cat_df)

#     # 3. Combine
#     X = np.hstack([text_features, cat_features])

#     # 4. Predict
#     prediction_encoded = saved_model['model'].predict(X)
#     predictions = saved_model['label_encoder'].inverse_transform(prediction_encoded)

#     # 5. Probabilities
#     all_probabilities = []
#     if hasattr(saved_model['model'], 'predict_proba'):
#         probas = saved_model['model'].predict_proba(X)
#         classes = saved_model['label_encoder'].classes_
#         for i in range(len(inputs)):
#             all_probabilities.append({cls: round(float(probas[i][j]), 4) for j, cls in enumerate(classes)})
#     else:
#         all_probabilities = [{} for _ in range(len(inputs))]

#     # 6. Build results
#     results = []
#     for i in range(len(inputs)):
#         results.append({
#             "prediction": {
#                 "coverage_status": predictions[i],
#                 "confidence": all_probabilities[i]
#             },
#             "input": inputs[i],
#             "metadata": {
#                 "model_type": type(saved_model['model']).__name__,
#                 "timestamp": datetime.now().isoformat(),
#                 "total_features": X.shape[1]
#             }
#         })

#     return results

# def ml_predict_coverage_status(payer_name, state_name, acronym, expansion, explanation):
#     """
#     Wrapper for pdf_processing integration.
#     Returns: (predicted_label, confidence_score)
#     """
#     saved_model = load_coverage_model()
#     if saved_model is None:
#         return None, 0.0

#     result = predict_coverage_status(
#         saved_model, payer_name, state_name, acronym, expansion, explanation
#     )
    
#     if result is None:
#         return None, 0.0
        
#     pred_label = result["prediction"]["coverage_status"]
#     conf_probs = result["prediction"]["confidence"]
#     confidence_score = max(conf_probs.values()) if conf_probs else 0.0

#     return pred_label, confidence_score

# def ml_predict_coverage_status_batch(inputs):
#     """
#     Wrapper for batch prediction.
#     'inputs' is a list of dicts with keys: payer_name, state_name, acronym, expansion, explanation
#     Returns: List of (predicted_label, confidence_score) tuples
#     """
#     saved_model = load_coverage_model()
#     if saved_model is None or not inputs:
#         return [(None, 0.0)] * len(inputs) if inputs else []

#     results = predict_coverage_status_batch(saved_model, inputs)
    
#     output = []
#     for res in results:
#         pred_label = res["prediction"]["coverage_status"]
#         conf_probs = res["prediction"]["confidence"]
#         confidence_score = max(conf_probs.values()) if conf_probs else 0.0
#         output.append((pred_label, confidence_score))
        
#     return output
