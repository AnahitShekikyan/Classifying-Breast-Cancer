# Breast Cancer Classification with Machine Learning

## Overview
This project classifies breast tumors as **malignant** or **benign** using supervised machine learning on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset. The analysis emphasizes **clinical interpretability** and **high recall** for screening-oriented decision support.
Six classification algorithms were rigorously evaluated using stratified 5-fold cross-validation and hold-out testing. **Logistic Regression** emerged as the optimal model, balancing strong predictive performance with transparent feature interpretation for clinical deployment.

---

## Key Results 

### Test Performance (Logistic Regression)

| Metric | Value |
|---|---:|
| Accuracy | 95.61% |
| Precision | 93.02% |
| Recall (Sensitivity) | 95.24% |
| F1 Score | 94.12% |
| ROC-AUC | 0.9960 |


### Confusion Matrix (Test Set, N=114)

```text
                Predicted
              Benign  Malignant
Actual Benign    69        3
Actual Malignant  2       40
```

These results indicate strong detection of malignant cases while maintaining high overall discrimination.

**Clinical Interpretation:**
- **40 of 42 malignant cases correctly identified** (95.24% recall)
- **2 false negatives** (missed cancer cases)
- **3 false positives** (~4.17% false-positive rate)
- **Near-perfect discrimination** (ROC-AUC = 0.9960)

---

## Dataset
- **Name** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Primary Source:** UCI Machine Learning Repository
- **Samples:** 569 observations
- **Task:** Binary classification (Malignant vs Benign)
- **Class Distribution:** 62.7% Benign, 37.3% Malignant
- **Target variable:** `diagnosis`
- **Features Used:** 3 carefully selected FNA-derived features
  - `area_worst` - Largest tumor area
  - `smoothness_worst` - Surface texture irregularity
  - `texture_mean` - Cell texture variation

---

## Models Evaluated

All models were trained using **stratified 5-fold cross-validation** and evaluated on a **held-out test set (20%)**:

| Model | CV Accuracy | Test Accuracy | Test Recall | Notes |
|-------|-------------|---------------|-------------|-------|
| **Logistic Regression** | **96.49%** | **95.61%** | **95.24%** | Selected |
| Neural Network (MLP) | 97.19% | - | - | Highest CV accuracy |
| Random Forest | 96.14% | - | - | Strong ensemble |
| Naïve Bayes | 95.43% | - | - | Fast baseline |
| Decision Tree (Gini) | 94.56% | - | - | Interpretable |
| Decision Tree (Entropy) | 94.21% | - | - | Alternative split |

**Selection Rationale:**  
Logistic Regression was chosen for deployment due to:
- Excellent recall (95.24% - critical for cancer screening)
- High interpretability (transparent feature coefficients)
- Computational efficiency (real-time predictions)
- Clinical alignment (features match medical understanding)
- Stable generalization (minimal overfitting)

---

## Feature Importance

Analysis of standardized Logistic Regression coefficients:

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| `area_worst` | +5.22 | Larger tumor area → malignancy |
| `smoothness_worst` | +1.50 | Irregular surface → malignancy |
| `texture_mean` | +1.04 | Heterogeneous texture → malignancy |

All coefficients align with established medical knowledge of tumor characteristics.

---

## Repository Contents

breast-cancer-classification/
├── Classifying_Breast_Cancer.ipynb                  # Complete analysis notebook
├── Breast_Cancer_Combined_Report.pdf                # Full technical report  with notebook
├── README.md                            
├── requirements.txt                                 # Python dependencies
└── LICENSE                                          # MIT License

---

## License

This project is licensed under the MIT License

---

## Acknowledgments

- Wisconsin Breast Cancer Database (Dr. William H. Wolberg, University of Wisconsin)
- UCI Machine Learning Repository for dataset hosting
- Scikit-learn development team for excellent ML tools
---

## Getting Started

### Prerequisites

```bash
Python 3.8+
pip

```

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/AnahitShekikyan/Classifying-Breast-Cancer.git

```

2. **Install dependencies:**
```bash
pip install -r requirements.txt

```

3. **Run the notebook:**
```bash
jupyter notebook Classifying_Breast_Cancer.ipynb

```

## Technologies Used

- **Python 3.8+** - Core programming language
- **scikit-learn** - Machine learning algorithms and evaluation
- **pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib / Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development environment
  
---

## Methodology Highlights

- **Train/Test Split:** 80/20 stratified split (random_state=42)
- **Cross-Validation:** Stratified 5-fold CV to preserve class distribution
- **Feature Scaling:** StandardScaler applied in pipelines
- **Class Balancing:** Balanced class weights to address imbalance
- **Evaluation Focus:** Recall prioritized (minimize false negatives)
- **Validation:** Both cross-validation and independent test set
  
---

## Documentation

- **Full report** (`Breast_Cancer_Combined_Report.pdf`): complete methodology, results, implications, limitations, and references
- **Notebook** (`Classifying_Breast_Cancer.ipynb`): full reproducible workflow, visualizations, and model implementation 

---

## Key Findings

1. **Parsimonious models can achieve excellent performance** - Only 3 features needed for 95%+ accuracy
2. **Interpretability matters in healthcare** - Logistic Regression's transparency outweighed marginal accuracy gains from complex models
3. **High recall is achievable** - 95.24% sensitivity minimizes dangerous false negatives
4. **Feature importance aligns with biology** - Data-driven findings match clinical understanding of malignancy
---

## Limitations & Future Work

### Limitations
- Single-institution dataset (generalization uncertain)
- Binary classification only (no borderline cases)
- Limited to FNA-derived features
- No temporal validation

### Future Work
- External validation on multi-institutional datasets
- Multimodal integration (imaging + clinical variables)
- Ensemble methods for improved robustness
- Deployment as clinical decision support system
- Clinical decision-support deployment pathway
  
---

  ## References

1. **Wolberg, W. H., Street, W. N., & Mangasarian, O. L.** (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B 

2. **Street, W. N., Wolberg, W. H., & Mangasarian, O. L.** (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE 1993 International Symposium on Electronic Imaging: Science and Technology*, 1905, 861–870.

3. **Centers for Disease Control and Prevention** (2024). U.S. Cancer Statistics: Breast Cancer Stat Bite.
   
---


