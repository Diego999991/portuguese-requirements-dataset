# Datareq Classification Code

Implementation of machine learning models for software requirements classification using the Datareq dataset.

## 📋 Overview

This code implements the three classification tasks described in the paper:

1. **Functionality Classification**: Classifies requirements as Functional or Non-Functional
2. **Single Category Classification**: Classifies requirements into one of 13 categories
3. **Multi-Category Classification (Top-2)**: Predicts the two most likely categories for each requirement

## 🔧 Requirements

### Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.3.0
scikit-learn>=1.0.0
numpy>=1.20.0
```

### Dataset Structure

The code expects a CSV file (`datareq.csv`) with the following columns:

- `description`: Text description of the requirement
- `functionality`: Label indicating if requirement is "Functional" or "Non-Functional"
- `category_1`: Primary category (one of 13 categories)
- `category_2`: Secondary category (optional, for multi-label tasks)

**Note:** Adjust column names in the code if your dataset uses different naming.

## 🚀 Usage

### Basic Usage

```python
from datareq_classifier import DatareqClassifier

# Initialize classifier
classifier = DatareqClassifier('datareq.csv')

# Preprocess data
classifier.preprocess_data(text_column='description')

# Run classification tasks
results_functionality = classifier.functionality_classification()
results_category = classifier.category_classification()
results_top2 = classifier.multi_category_classification_top2()
```

### Running All Tasks

```bash
python datareq_classifier.py
```

This will execute all three classification tasks and display:
- Accuracy and F1-Score for each model
- Classification reports
- Confusion matrices
- Summary of results

## 📊 Models Implemented

### Support Vector Machine (SVM)
- **Kernel**: Linear
- **Hyperparameter C**: 3 (empirically determined)
- **Best performing model** in the paper

### Naive Bayes
- **Variants**: Multinomial and Bernoulli
- Both variants showed similar performance

## 🎯 Classification Tasks

### Task 1: Functionality Classification

Classifies requirements as Functional or Non-Functional using only the requirement description.

**Expected Results (from paper):**
- SVM: 83% accuracy, 80% F1-score
- Naive Bayes: 83% accuracy, 80% F1-score

```python
results = classifier.functionality_classification()
```

### Task 2: Single Category Classification

Classifies requirements into one of 13 categories using both the description and functionality label.

**Categories:**
- Operational
- Usability
- Security
- Accessibility
- Legal
- Portability
- Scalability
- Fault Tolerance
- Maintainability
- Performance
- Lifecycle
- Use Cases
- Compliance

**Expected Results (from paper):**
- SVM: 48.5% accuracy, 46.8% F1-score
- Naive Bayes: 44% accuracy, 38% F1-score

```python
results = classifier.category_classification()
```

### Task 3: Multi-Category Classification (Top-2)

Predicts the two most likely categories for each requirement, accounting for ambiguity.

**Expected Results (from paper):**
- SVM: 75% accuracy
- Naive Bayes: 65% accuracy

```python
results = classifier.multi_category_classification_top2()
```

## ⚖️ Class Imbalance Mitigation

The code includes a method to handle class imbalance using balanced class weights:

```python
results = classifier.class_imbalance_mitigation()
```

This approach increases the importance of minority classes during training, as discussed in Section 4.5 of the paper. With balanced weights, SVM achieved 50% accuracy on category classification.

## 📈 Text Preprocessing

The code uses **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization:

- **Max features**: 1000
- **N-gram range**: (1, 2) - unigrams and bigrams
- Captures important terms while reducing dimensionality

## 🔍 Model Evaluation

The code uses:
- **Train/Test Split**: 75% training, 25% testing
- **Stratified sampling**: Maintains class distribution
- **Random state**: 42 (for reproducibility)

**Metrics:**
- Accuracy
- F1-Score (weighted)
- Classification Report
- Confusion Matrix

## 📁 Project Structure

```
datareq/
├── datareq.csv              # Dataset file
├── datareq_classifier.py    # Main classification code
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🎓 Example Output

```
Dataset loaded: 1502 requirements
Text vectorized: 1000 features

============================================================
TASK 1: FUNCTIONALITY CLASSIFICATION
============================================================

--- SVM (Linear Kernel, C=3) ---
Accuracy: 83.00%
F1-Score: 80.00%

Classification Report:
              precision    recall  f1-score   support
  Functional       0.85      0.95      0.90       300
Non-Functional     0.75      0.50      0.60        76

    accuracy                           0.83       376
```

## 🛠️ Customization

### Adjusting Hyperparameters

```python
# Change SVM C parameter
svm_model = SVC(kernel='linear', C=5, random_state=42)

# Try different kernels
svm_model = SVC(kernel='rbf', C=3, random_state=42)
```

### Using Different Text Features

```python
# Adjust TF-IDF parameters
vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 3),
    min_df=2,
    max_df=0.95
)
```

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 4-fold cross-validation (as mentioned in paper)
scores = cross_val_score(svm_model, X, y, cv=4, scoring='accuracy')
print(f"CV Accuracy: {scores.mean():.2%} (+/- {scores.std():.2%})")
```

## ⚠️ Known Limitations

- **Class Imbalance**: Dataset has significant bias toward certain categories (Use Cases, Usability, Operational)
- **Language**: All requirements use system-level (technical) language
- **Functional Bias**: 80% of requirements are functional vs. 20% non-functional

These limitations are discussed in detail in Section 4.5 of the paper.

## 🔮 Future Improvements

- [ ] Implement data augmentation for minority classes
- [ ] Test with transformer models (BERT, BERTimbau)
- [ ] Add undersampling techniques
- [ ] Implement ensemble methods
- [ ] Add visualization of decision boundaries
- [ ] Create web interface for real-time classification

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{datareq2024,
  title={Datareq: um dataset de requisitos em português},
  author={Santos, John V. and Primo, Antônio K. C. and Ribeiro, Gisele P. and Mariano, Diego N.},
  booktitle={Conference Proceedings},
  year={2024},
  organization={Federal University of Ceará}
}
```

## 📧 Contact

For questions or issues with the code, please open an issue in the repository or contact the authors.

## 📄 License

[Specify license - should match dataset license]

---

**Federal University of Ceará (UFC)**  
Fortaleza - CE - Brazil