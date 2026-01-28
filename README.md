# Fake Job Posting Detection

This project focuses on detecting **fraudulent job postings** by combining **structured behavioral features** with **text-based semantic analysis** using **TF-IDF** and **linear machine learning models**.

The goal is not only **high accuracy**, but a **balanced and explainable fraud detection system** suitable for real-world usage.

---

## Problem Statement

Online job portals frequently contain **fraudulent postings** that attempt to scam applicants. These fake jobs are often **well-written** and difficult to identify using simple rules.

This project aims to classify job postings as **real or fake** while addressing:

- Severe class imbalance  
- High-effort fraud that mimics real jobs  
- Trade-offs between missed frauds and false alarms  

---

## Dataset

- **Total samples:** 17,880  
- **Fraudulent jobs:** 866 (~5%)

### Data Includes:
- Job descriptions  
- Company profile information  
- Employment metadata  
- Platform-level indicators (logo, questions, telecommuting)

 Due to heavy class imbalance, **accuracy is not a reliable metric** for this task.

---

## Exploratory Data Analysis (EDA)

### Key Findings:
- Fraudulent postings frequently **lack company profile information**
- Fake jobs tend to mention **organizations and locations less often**
- Company profile length differs significantly between **real and fake jobs**
- Simple keyword-based heuristics are **insufficient** for detecting high-effort fraud

EDA showed that **text semantics are essential**, motivating the use of **TF-IDF**.

---

## Feature Engineering

### Structured Features:
- Company profile missing indicator  
- Log-transformed company profile length  
- Organization and location entity counts  
- Information completeness score  
- Employment type indicators  
- Platform metadata (logo, questions, telecommuting)

These features capture **behavioral and structural fraud signals**.

### Text Representation (TF-IDF):
- Applied TF-IDF to job descriptions  
- Removed stopwords  
- Filtered rare terms  
- Limited vocabulary to informative words  

TF-IDF allows the model to learn **language patterns directly from data**, rather than relying on handcrafted keyword lists.

---

## Models Evaluated

All models were trained on the **same TF-IDF + structured feature set**.

| Model               | Missed Frauds (FN) | False Positives (FP) |
|--------------------|-------------------|---------------------|
| Logistic Regression | 23                | 170                 |
| **Linear SVM (Final)** | 36                | 61                  |
| XGBoost             | 32                | 116                 |

---

## Final Model Selection

**Linear SVM** was selected as the final model.

Although **Logistic Regression** achieved higher fraud recall, it produced significantly more **false positives**.  
Linear SVM provided a **better balance**, reducing false alarms substantially while maintaining acceptable fraud detection performance.

---

## Model Interpretability

Because a **linear model** was used, feature weights were analyzed.

### Observations:
- Fraudulent jobs are associated with:
  - Money-related terms  
  - External links  
  - Vague work-from-home language  
- Real jobs emphasize:
  - Task descriptions  
  - Team structure  
  - Operational details  

This confirms that **semantic language differences** are key to detecting high-effort fraud.

---

## Limitations

- The model may struggle with **very short job descriptions**
- New scam language patterns may reduce effectiveness
- Linear SVM does **not provide probability estimates**

---

## Future Work

- Use contextual embeddings (e.g., **BERT**) for deeper semantic understanding  
- Incorporate **URL/domain reputation** features  
- Add **human-in-the-loop review** for borderline cases  

---

## Tech Stack

- Python  
- Pandas, NumPy  
- Scikit-learn  
- TF-IDF  
- Linear Support Vector Machine  
