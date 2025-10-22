# Datareq: Portuguese Software Requirements Dataset

A Brazilian dataset of software requirements for requirements engineering and natural language processing (NLP).

## 📋 About the Dataset

**Datareq** is a dataset containing **1,502 software requirements** in Brazilian Portuguese, extracted from 8 public notices available online. Each requirement was manually labeled with three main attributes, making it ideal for supervised learning tasks and automatic requirements classification.

### Key Features

- **1,502 software requirements** in Brazilian Portuguese
- **8 sources**: public notices (.GOV and .ORG)
- **Period**: 2011-2024
- **Manual labeling** with multiple attributes
- **Format**: Structured spreadsheet

## 🏷️ Data Structure

Each requirement is labeled with the following attributes:

### 1. Functionality
- **Functional** (1,200 requirements - 80%): Describes what the system does
- **Non-Functional** (302 requirements - 20%): Describes how the system should behave

### 2. Type (Language)
- **System**: Detailed technical language (100% of requirements)
- **User**: Language comprehensible to end users

### 3. Categories (up to 2 per requirement)

| Category | Description |
|----------|-------------|
| **Operational** | Support for operating systems, networks, or technologies |
| **Usability** | System functions that users can interact with |
| **Security** | Permission management and protection against attackers |
| **Accessibility** | Ensuring access for people with different abilities |
| **Legal** | Legal or regulatory compliance |
| **Portability** | Ability to transfer to different environments |
| **Scalability** | Capacity growth without performance loss |
| **Fault Tolerance** | Continued operation in case of failures |
| **Maintainability** | Maintenance, corrections, and updates |
| **Performance** | Performance and response times |
| **Lifecycle** | Maintenance, discontinuation, or updates |
| **Use Cases** | Specific interactions between users and system |
| **Compliance** | Code standards or style |

## 📊 Data Distribution

- **Functional**: 80% (1,200 requirements)
- **Non-Functional**: 20% (302 requirements)
- **Most frequent categories**: Use Cases, Usability, Operational

## 🎯 Applications

Datareq can be used for:

- ✅ Automatic requirements classification
- ✅ Training NLP models
- ✅ Functional vs. non-functional requirements identification
- ✅ Multi-category requirements classification
- ✅ Specification document analysis
- ✅ Requirements engineering research

## 🔬 Benchmark Results

Tested models: **SVM** (linear kernel) and **Naive Bayes** (Multinomial and Bernoulli)

### Support Vector Machine (SVM) - Best Performance

| Task | Accuracy | F1-Score |
|------|----------|----------|
| Functionality Classification | 83% | 80% |
| Category Classification (single) | 48.5% | 46.8% |
| Multi-Category Classification (Top-2) | 75% | N/A |

### Naive Bayes

| Task | Accuracy | F1-Score |
|------|----------|----------|
| Functionality Classification | 83% | 80% |
| Category Classification (single) | 44% | 38% |
| Multi-Category Classification (Top-2) | 65% | N/A |

## 🚀 How to Use

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('datareq.csv')

# View structure
print(df.head())
print(df.columns)

# Basic analysis
print(f"Total requirements: {len(df)}")
print(f"Functionality distribution:\n{df['functionality'].value_counts()}")
print(f"Primary categories:\n{df['category_1'].value_counts()}")
```

## ⚠️ Known Limitations

- **Distribution bias**: Predominance of functional requirements (80%)
- **Language**: All requirements use system language (technical)
- **Imbalance**: Categories like Use Cases and Usability are much more frequent
- **Source**: Concentrated on government procurement notices

## 🔮 Future Work

- [ ] Include requirements with user language (non-technical)
- [ ] Balance distribution across categories
- [ ] Increase representation of non-functional requirements
- [ ] Generate synthetic data for minority categories
- [ ] Explore diverse sources (manuals, project documentation)

## 📚 Citation

If you use Datareq in your research, please cite:

```bibtex
@inproceedings{datareq2024,
  title={Datareq: um dataset de requisitos em português},
  author={Santos, John V. and Primo, Antônio K. C. and Ribeiro, Gisele P. and Mariano, Diego N.},
  booktitle={Conference Proceedings},
  year={2024},
  organization={Federal University of Ceará}
}
```

## 👥 Authors

- **John V. Santos** - Technology Center, UFC
- **Antônio K. C. Primo** - Science Center, UFC
- **Gisele P. Ribeiro** - Technology Center, UFC
- **Diego N. Mariano** - Science Center, UFC

## 📄 License

[Specify dataset license]

## 🤝 Contributing

Contributions are welcome! Feel free to:

- Report bugs or inconsistencies
- Suggest improvements
- Add new requirements
- Propose new classification tasks

## 📧 Contact

For questions or suggestions, contact us at [insert contact email].

---

**Federal University of Ceará (UFC)**  
Fortaleza - CE - Brazil