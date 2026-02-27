"""
Model Evaluation Script
========================
Load saved model and check for overfitting/underfitting without retraining.
Generates a comprehensive evaluation report.
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# =============================================================================
# LOAD MODEL AND DATA
# =============================================================================

def load_saved_model(model_path):
    """Load the saved model and its components."""
    print("=" * 60)
    print("LOADING SAVED MODEL")
    print("=" * 60)
    
    model_data = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Model type: {model_data['metrics']['model_name']}")
    print(f"Saved Test Accuracy: {model_data['metrics']['test_accuracy']:.4f}")
    print(f"Saved CV Accuracy: {model_data['metrics']['cv_accuracy_mean']:.4f}")
    
    return model_data


def load_and_prepare_data(data_path, model_data):
    """Load data and prepare features using saved transformers."""
    print("\n" + "=" * 60)
    print("LOADING AND PREPARING DATA")
    print("=" * 60)
    
    # Load data
    try:
        df = pd.read_csv(data_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(data_path, encoding='cp1252')
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding='latin-1')
    
    df.columns = df.columns.str.strip()
    df['Coverage Status'] = df['Coverage Status'].str.strip()
    
    print(f"Total records: {len(df)}")
    print(f"\nCoverage Status Distribution:")
    print(df['Coverage Status'].value_counts())
    
    # Prepare features
    df['EXPANSION'] = df['EXPANSION'].fillna('')
    df['Explanation'] = df['Explanation'].fillna('')
    df['combined_text'] = (df['EXPANSION'].astype(str) + ' ' + df['Explanation'].astype(str)).str.lower().str.strip()
    df['ACRONYM_clean'] = df['ACRONYM'].str.strip().str.upper()
    
    # Transform using saved transformers
    tfidf = model_data['tfidf']
    onehot = model_data['onehot']
    label_encoder = model_data['label_encoder']
    
    X_text_tfidf = tfidf.transform(df['combined_text']).toarray()
    X_cat = df[['PAYER NAME', 'STATE NAME', 'ACRONYM_clean']]
    X_cat_encoded = onehot.transform(X_cat)
    X_combined = np.hstack([X_text_tfidf, X_cat_encoded])
    
    y_encoded = label_encoder.transform(df['Coverage Status'])
    
    print(f"\nFeature dimensions: {X_combined.shape[1]}")
    
    return df, X_combined, y_encoded, label_encoder


# =============================================================================
# FIT STATUS CHECK
# =============================================================================

def check_fit_status(model, X_combined, y_encoded, label_encoder, test_size=0.2):
    """
    Check if model is underfitted, overfitted, or well-fitted.
    """
    print("\n" + "=" * 60)
    print("MODEL FIT STATUS ANALYSIS")
    print("=" * 60)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_encoded, 
        test_size=test_size, 
        random_state=42, 
        stratify=y_encoded
    )
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Get predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"\nPerformance Metrics:")
    print(f"  Training Accuracy:    {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Test Accuracy:        {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Calculate gap
    train_test_gap = train_accuracy - test_accuracy
    
    print(f"\nAccuracy Gap:")
    print(f"  Train - Test Gap:     {train_test_gap:.4f} ({train_test_gap*100:.2f}%)")
    
    # Determine fit status
    fit_status = "WELL-FITTED"
    fit_details = []
    
    # Check for underfitting
    if train_accuracy < 0.85:
        fit_status = "UNDERFITTED"
        fit_details.append("Training accuracy is below 85%")
    if test_accuracy < 0.85:
        fit_status = "UNDERFITTED"
        fit_details.append("Test accuracy is below 85%")
    
    # Check for overfitting
    if train_test_gap > 0.05:
        fit_status = "OVERFITTED"
        fit_details.append(f"Train-Test gap ({train_test_gap:.2%}) exceeds 5% threshold")
    
    # Well-fitted conditions
    if not fit_details:
        fit_details.append("Training and Test accuracies are consistent")
        fit_details.append("No significant accuracy gap detected")
        fit_details.append("Model generalizes well to unseen data")
    
    print("\n" + "-" * 60)
    print(f"FIT STATUS: {fit_status}")
    print("-" * 60)
    for detail in fit_details:
        print(f"  • {detail}")
    
    # Recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS:")
    print("-" * 60)
    
    if fit_status == "UNDERFITTED":
        print("  • Increase model complexity (more estimators, deeper trees)")
        print("  • Add more features or feature engineering")
        print("  • Reduce regularization parameters")
    elif fit_status == "OVERFITTED":
        print("  • Reduce model complexity")
        print("  • Increase regularization")
        print("  • Use more training data")
    else:
        print("  ✓ Model is well-fitted and ready for production")
        print("  ✓ No changes recommended")
    
    # Classification Report
    print("\n" + "=" * 60)
    print("CLASSIFICATION REPORT (Test Data)")
    print("=" * 60)
    
    unique_labels = np.unique(np.concatenate([y_test, y_test_pred]))
    target_names = [label_encoder.classes_[i] for i in unique_labels]
    print(classification_report(y_test, y_test_pred, labels=unique_labels, target_names=target_names))
    
    # Confusion Matrix
    print("CONFUSION MATRIX:")
    cm = confusion_matrix(y_test, y_test_pred, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    print(cm_df)
    
    # Return report data
    return {
        'fit_status': fit_status,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'train_test_gap': train_test_gap,
        'details': fit_details,
        'y_test': y_test,
        'y_test_pred': y_test_pred,
        'label_encoder': label_encoder,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def generate_evaluation_report(fit_report, df, model_name):
    """Generate and save evaluation report to file."""
    print("\n" + "=" * 60)
    print("GENERATING EVALUATION REPORT")
    print("=" * 60)
    
    # Create reports directory
    reports_dir = r"c:\Users\VH0000812\Desktop\Coverage\reports"
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(reports_dir, f"evaluation_report_{timestamp}.txt")
    
    label_encoder = fit_report['label_encoder']
    y_test = fit_report['y_test']
    y_pred = fit_report['y_test_pred']
    
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("COVERAGE CLASSIFICATION MODEL - EVALUATION REPORT\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {fit_report['timestamp']}\n")
        f.write(f"Model: {model_name}\n\n")
        
        # Data Summary
        f.write("-" * 70 + "\n")
        f.write("1. DATA SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total Records Evaluated: {len(df)}\n\n")
        f.write("Class Distribution:\n")
        for cls in label_encoder.classes_:
            count = (df['Coverage Status'] == cls).sum()
            pct = count / len(df) * 100
            f.write(f"  {cls}: {count} ({pct:.1f}%)\n")
        
        # Fit Status
        f.write("\n" + "-" * 70 + "\n")
        f.write("2. FIT STATUS ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Status: {fit_report['fit_status']}\n\n")
        f.write(f"Training Accuracy: {fit_report['train_accuracy']:.4f} ({fit_report['train_accuracy']*100:.2f}%)\n")
        f.write(f"Test Accuracy:     {fit_report['test_accuracy']:.4f} ({fit_report['test_accuracy']*100:.2f}%)\n")
        f.write(f"Train-Test Gap:    {fit_report['train_test_gap']:.4f} ({fit_report['train_test_gap']*100:.2f}%)\n\n")
        f.write("Analysis:\n")
        for detail in fit_report['details']:
            f.write(f"  • {detail}\n")
        
        # Classification Report
        f.write("\n" + "-" * 70 + "\n")
        f.write("3. CLASSIFICATION REPORT\n")
        f.write("-" * 70 + "\n")
        unique_labels = np.unique(np.concatenate([y_test, y_pred]))
        target_names = [label_encoder.classes_[i] for i in unique_labels]
        f.write(classification_report(y_test, y_pred, labels=unique_labels, target_names=target_names))
        
        # Confusion Matrix
        f.write("\n" + "-" * 70 + "\n")
        f.write("4. CONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")
        cm = confusion_matrix(y_test, y_pred, labels=unique_labels)
        cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
        f.write(cm_df.to_string() + "\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")
    
    print(f"Report saved to: {report_path}")
    return report_path


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Paths
    MODEL_PATH = r"c:\Users\VH0000812\Desktop\Coverage\models\coverage_model.pkl"
    DATA_PATH = r"c:\Users\VH0000812\Desktop\Coverage\data\Acronym_2026_cleaned.csv"
    
    # Load saved model
    model_data = load_saved_model(MODEL_PATH)
    
    # Load and prepare data
    df, X_combined, y_encoded, label_encoder = load_and_prepare_data(DATA_PATH, model_data)
    
    # Check fit status
    model = model_data['model']
    fit_report = check_fit_status(model, X_combined, y_encoded, label_encoder)
    
    # Generate report
    model_name = model_data['metrics']['model_name']
    report_path = generate_evaluation_report(fit_report, df, model_name)
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
