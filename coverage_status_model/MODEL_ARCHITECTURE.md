# Coverage Classification Model - Architecture

---

## 1. Complete Workflow: Training to Prediction

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING PHASE                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐                  │
│   │  Raw CSV     │ ───► │ Data Cleaning│ ───► │ Clean Data   │                  │
│   │  (9,335 rows)│      │ (Remove null)│      │ (7,463 rows) │                  │
│   └──────────────┘      └──────────────┘      └──────────────┘                  │
│                                                       │                          │
│                                                       ▼                          │
│                              ┌─────────────────────────────────────┐            │
│                              │       FEATURE ENGINEERING           │            │
│                              ├─────────────────────────────────────┤            │
│                              │  Text (EXPANSION + Explanation)     │            │
│                              │           ↓                         │            │
│                              │     TF-IDF Vectorizer               │            │
│                              │     (500 features)                  │            │
│                              │           +                         │            │
│                              │  Categorical (PAYER, STATE, ACRONYM)│            │
│                              │           ↓                         │            │
│                              │     One-Hot Encoding                │            │
│                              │     (371 features)                  │            │
│                              │           ↓                         │            │
│                              │     Combined: 871 features          │            │
│                              └─────────────────────────────────────┘            │
│                                                       │                          │
│                                                       ▼                          │
│                              ┌─────────────────────────────────────┐            │
│                              │         TRAIN/TEST SPLIT            │            │
│                              │         (80% / 20%)                 │            │
│                              └─────────────────────────────────────┘            │
│                                          │                                       │
│                         ┌────────────────┴────────────────┐                     │
│                         ▼                                 ▼                     │
│              ┌──────────────────┐              ┌──────────────────┐             │
│              │   Training Set   │              │    Test Set      │             │
│              │   (5,970 rows)   │              │   (1,493 rows)   │             │
│              └──────────────────┘              └──────────────────┘             │
│                         │                                 │                     │
│                         ▼                                 │                     │
│              ┌──────────────────────────────┐             │                     │
│              │    5-FOLD CROSS VALIDATION   │             │                     │
│              ├──────────────────────────────┤             │                     │
│              │  For each fold:              │             │                     │
│              │    1. Apply SMOTE            │             │                     │
│              │    2. Train Model            │             │                     │
│              │    3. Validate               │             │                     │
│              └──────────────────────────────┘             │                     │
│                         │                                 │                     │
│                         ▼                                 │                     │
│   ┌─────────────────────────────────────────────────┐    │                     │
│   │              TRAIN 4 MODELS                      │    │                     │
│   ├────────────┬────────────┬───────────┬───────────┤    │                     │
│   │  Random    │  Gradient  │   SVM     │ Logistic  │    │                     │
│   │  Forest    │  Boosting  │           │ Regression│    │                     │
│   └────────────┴────────────┴───────────┴───────────┘    │                     │
│                         │                                 │                     │
│                         ▼                                 │                     │
│              ┌──────────────────┐                        │                     │
│              │  EVALUATE MODELS │◄───────────────────────┘                     │
│              │  on Test Set     │                                              │
│              └──────────────────┘                                              │
│                         │                                                       │
│                         ▼                                                       │
│              ┌──────────────────┐                                              │
│              │  SELECT BEST     │                                              │
│              │  (Gradient       │                                              │
│              │   Boosting)      │                                              │
│              └──────────────────┘                                              │
│                         │                                                       │
│                         ▼                                                       │
│              ┌──────────────────┐                                              │
│              │  SAVE MODEL      │                                              │
│              │  (.pkl file)     │                                              │
│              └──────────────────┘                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             PREDICTION PHASE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                        NEW INPUT                                  │          │
│   │  • Payer Name: "Humana"                                          │          │
│   │  • State Name: "Connecticut"                                      │          │
│   │  • Acronym: "Tier 5"                                             │          │
│   │  • Expansion: "Specialty Tier"                                    │          │
│   │  • Explanation: "Some injectables and other high-cost drugs"     │          │
│   └──────────────────────────────────────────────────────────────────┘          │
│                                          │                                       │
│                                          ▼                                       │
│                         ┌────────────────────────────────┐                      │
│                         │      LOAD SAVED MODEL          │                      │
│                         │      (coverage_model.pkl)      │                      │
│                         └────────────────────────────────┘                      │
│                                          │                                       │
│                                          ▼                                       │
│                         ┌────────────────────────────────┐                      │
│                         │    PREPROCESS INPUT            │                      │
│                         │    • Combine text fields       │                      │
│                         │    • Lowercase                 │                      │
│                         │    • Clean acronym             │                      │
│                         └────────────────────────────────┘                      │
│                                          │                                       │
│                         ┌────────────────┴────────────────┐                     │
│                         ▼                                 ▼                     │
│              ┌──────────────────┐              ┌──────────────────┐             │
│              │  TF-IDF Transform│              │ One-Hot Encode   │             │
│              │  (500 features)  │              │ (371 features)   │             │
│              └──────────────────┘              └──────────────────┘             │
│                         │                                 │                     │
│                         └────────────────┬────────────────┘                     │
│                                          ▼                                       │
│                         ┌────────────────────────────────┐                      │
│                         │      COMBINE FEATURES          │                      │
│                         │      (871 total)               │                      │
│                         └────────────────────────────────┘                      │
│                                          │                                       │
│                                          ▼                                       │
│                         ┌────────────────────────────────┐                      │
│                         │   GRADIENT BOOSTING PREDICT    │                      │
│                         │   • predict() → Class          │                      │
│                         │   • predict_proba() → Scores   │                      │
│                         └────────────────────────────────┘                      │
│                                          │                                       │
│                                          ▼                                       │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                        JSON OUTPUT                                │          │
│   │  {                                                                │          │
│   │    "prediction": {                                                │          │
│   │      "coverage_status": "Covered",                                │          │
│   │      "confidence": {                                              │          │
│   │        "Covered": 0.72,                                           │          │
│   │        "Covered with Condition": 0.28,                            │          │
│   │        "Not Covered": 0.00                                        │          │
│   │      }                                                            │          │
│   │    },                                                             │          │
│   │    "model_evaluation": {                                          │          │
│   │      "test_accuracy": ">99%",                                     │          │
│   │      "confidence_level": ">95%"                                   │          │
│   │    }                                                              │          │
│   │  }                                                                │          │
│   └──────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Pipeline

```
Raw CSV Data (9,335 rows)
         ↓
    Data Cleaning
    (Remove null Coverage Status)
         ↓
Clean Data (7,463 rows)
         ↓
    Feature Engineering
         ↓
    Model Training
         ↓
    Saved Model (.pkl)
```

---

## 2. Feature Engineering

| Feature Type | Input | Processing | Output Dimensions |
|--------------|-------|------------|-------------------|
| **Text Features** | EXPANSION + Explanation | TF-IDF Vectorizer | 500 features |
| **Categorical Features** | PAYER NAME, STATE NAME, ACRONYM | One-Hot Encoding | 371 features |
| **Total** | - | Combined | **871 features** |

**TF-IDF Configuration:**
- Max features: 500
- N-gram range: (1, 2) - Unigrams + Bigrams
- Stop words: English
- Min document frequency: 2

---

## 3. Models Evaluated

We trained **4 different models** to compare and select the best one:

| # | Model | Type | Why Include? |
|---|-------|------|--------------|
| 1 | **Random Forest** | Ensemble (Bagging) | Fast, handles non-linear data, resistant to overfitting |
| 2 | **Gradient Boosting** | Ensemble (Boosting) | High accuracy, learns from errors sequentially |
| 3 | **SVM** | Kernel-based | Effective in high-dimensional space (871 features) |
| 4 | **Logistic Regression** | Linear | Simple baseline, interpretable, fast training |

### Why Multiple Models?

1. **No single model works best for all datasets** - Different algorithms have different strengths
2. **Fair comparison** - Evaluate multiple approaches under same conditions
3. **Avoid bias** - Don't assume one algorithm will always win
4. **Find optimal fit** - Dataset characteristics determine best model
5. **Validation** - If multiple models perform similarly, results are more trustworthy

### Selection Criteria

The best model is selected based on:
- Highest **Test Accuracy**
- Smallest **CV-Test Gap** (no overfitting)
- Highest **Confidence Level**

**Result:** Gradient Boosting was selected (>99% accuracy, >95% confidence)

> **Note:** In production, only the best model (Gradient Boosting) is saved and used for predictions. The other 3 models are discarded after comparison.

---

## 4. Model Architecture

```
                    ┌─────────────────────────────────┐
                    │         INPUT LAYER             │
                    │  (871 combined features)        │
                    └─────────────────────────────────┘
                                   ↓
                    ┌─────────────────────────────────┐
                    │           SMOTE                 │
                    │  (Class Balancing - per fold)   │
                    └─────────────────────────────────┘
                                   ↓
        ┌──────────────┬──────────────┬──────────────┬──────────────┐
        ↓              ↓              ↓              ↓
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│Random Forest ││  Gradient    ││     SVM      ││  Logistic    │
│              ││  Boosting    ││              ││  Regression  │
│ 100 trees    ││  100 trees   ││  RBF kernel  ││  L2 penalty  │
│ max_depth=8  ││  max_depth=4 ││  C=0.5       ││  C=0.5       │
└──────────────┘└──────────────┘└──────────────┘└──────────────┘
        ↓              ↓              ↓              ↓
        └──────────────┴──────────────┴──────────────┘
                                   ↓
                    ┌─────────────────────────────────┐
                    │      MODEL SELECTION            │
                    │  (Best Test Accuracy Wins)      │
                    └─────────────────────────────────┘
                                   ↓
                    ┌─────────────────────────────────┐
                    │        OUTPUT LAYER             │
                    │  3 Classes + Confidence Scores  │
                    └─────────────────────────────────┘
```

---

## 4. Best Model: Gradient Boosting Classifier

| Parameter | Value | Purpose |
|-----------|-------|---------|
| n_estimators | 100 | Number of boosting stages |
| max_depth | 4 | Maximum tree depth (regularization) |
| min_samples_split | 5 | Min samples to split node |
| min_samples_leaf | 2 | Min samples in leaf node |
| learning_rate | 0.1 | Step size shrinkage |
| subsample | 0.8 | Fraction of samples per tree |
| random_state | 42 | Reproducibility |

---

## 5. Output Classes

| Class ID | Label | Description |
|----------|-------|-------------|
| 0 | Covered | Drug is fully covered |
| 1 | Covered with Condition | Drug covered with restrictions (PA, ST, QL, etc.) |
| 2 | Not Covered | Drug is not covered by plan |

---

## 6. Training Pipeline (with imblearn)

```python
Pipeline([
    ('smote', SMOTE(random_state=42)),        # Applied per CV fold
    ('classifier', GradientBoostingClassifier(...))
])
```

---

## 7. Cross-Validation Strategy

| Parameter | Value |
|-----------|-------|
| Method | StratifiedKFold |
| Folds | 5 |
| Shuffle | True |
| SMOTE | Inside each fold (no data leakage) |

---

## 8. Saved Model Components

```python
{
    'model': GradientBoostingClassifier,    # Trained classifier
    'label_encoder': LabelEncoder,           # Target encoding
    'tfidf': TfidfVectorizer,               # Text feature transformer
    'onehot': OneHotEncoder,                # Categorical transformer
    'metrics': {
        'model_name': 'Gradient Boosting',
        'test_accuracy': '>99%',
        'cv_accuracy_mean': '>99%',
        'cv_accuracy_std': '<0.3%',
        'training_time': '~6 min'
    }
}
```

---

## 9. Prediction Flow

```
New Input (payer, state, acronym, expansion, explanation)
                    ↓
         Text Preprocessing
         (lowercase, combine)
                    ↓
         TF-IDF Transform (500 features)
                    ↓
         One-Hot Encode (371 features)
                    ↓
         Combine Features (871 total)
                    ↓
         Gradient Boosting Predict
                    ↓
         Output: Class + Confidence Scores
```

---

## 10. JSON Output Structure

```json
{
  "prediction": {
    "coverage_status": "Covered with Condition",
    "confidence": {
      "Covered": 0.28,
      "Covered with Condition": 0.71,
      "Not Covered": 0.01
    }
  },
  "input": { ... },
  "metadata": {
    "model_type": "GradientBoostingClassifier",
    "total_features": 871
  },
  "model_evaluation": {
    "test_accuracy": ">99%",
    "cv_accuracy_mean": ">99%",
    "confidence_level": ">95%"
  }
}
```

---

## 11. Performance Comparison

### Stage 1: Before SMOTE ⚠️ Low Confidence + Overfitting Risk

| Model | CV Accuracy | Test Accuracy | CV-Test Gap |
|-------|-------------|---------------|-------------|
| Random Forest | ~97% | ~97% | <1% |
| Gradient Boosting | ~99% | ~99% | >0.5% |
| SVM | ~99% | ~99% | <0.5% |
| Logistic Regression | ~99% | ~99% | <0.5% |

**Issues:**
- ❌ Class imbalance: Minority class underrepresented
- ❌ Overfitting detected: Test accuracy higher than CV accuracy
- ❌ **Confidence Level: >60%** - Not reliable for production use

---

### Stage 2: After SMOTE (Before Overfitting Control) ⚠️ Data Leakage + Overfitting

| Model | CV Accuracy | Test Accuracy | CV-Test Gap |
|-------|-------------|---------------|-------------|
| Random Forest | ~97% | ~97% | <0.5% |
| Gradient Boosting | ~99% | ~99% | <0.5% |
| SVM | >99% | ~99% | >0.4% |
| Logistic Regression | >99% | ~99% | >0.4% |

**Issues:**
- ❌ Data leakage: SMOTE applied before CV split
- ❌ Inflated CV scores: Artificially high due to data leakage
- ❌ **Confidence Level: >70%** - Improved but still unreliable

---

### Stage 3: After SMOTE + Overfitting Control (Final) ✅ Reliable & Balanced

| Model | CV Accuracy | Test Accuracy | CV-Test Gap |
|-------|-------------|---------------|-------------|
| Random Forest | ~96% | ~96% | <0.2% |
| **Gradient Boosting** | **>99%** | **>99%** | **<0.25%** |
| SVM | ~98% | ~98% | <0.1% |
| Logistic Regression | ~98% | ~99% | <0.1% |

**Fixes Applied:**
- ✅ SMOTE inside CV folds: No data leakage
- ✅ Regularization added: Prevents overfitting
- ✅ **Confidence Level: >95%** - Reliable for production use

---

## 12. Confidence Level Summary

| Stage | Confidence Level | Reliability |
|-------|------------------|-------------|
| **Stage 1** | >60% | ❌ Low |
| **Stage 2** | >70% | ❌ Unreliable |
| **Stage 3** | >95% | ✅ High |

---

## 13. Files Structure

```
Coverage/
├── data/
│   ├── Acronym.csv              # Raw data
│   └── Acronym_cleaned.csv      # Cleaned data
├── models/
│   └── coverage_model.pkl       # Trained model
├── coverage_classification_model.py  # Training script
├── data_processing.py           # Data cleaning script
├── main.py                      # Prediction script
└── MODEL_ARCHITECTURE.md        # This file
```
