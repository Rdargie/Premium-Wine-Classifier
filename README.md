👉 **[Click Here to view the Interactive Notebook and SHAP Charts in Google Colab](https://colab.research.google.com/github/Rdargie/Explainable-AI-Wine-Classifier/blob/main/Wine_Quality_XAI.ipynb)**

#  Predicting Premium Wine Quality: An Explainable AI (XAI) Approach

##  The Scientific Objective
In Viticulture and Enology, distinguishing a "Premium" vintage from a standard wine relies heavily on complex, non-linear biochemical interactions. The goal of this project is to build an advanced machine learning classifier capable of identifying premium wines (Quality Score >= 7) based purely on their physicochemical properties, while completely opening the algorithmic "black box" to understand the *why* behind the prediction.

##  Machine Learning Architecture
* **Algorithm:** Random Forest Classifier (`scikit-learn`)
* **Handling Imbalance:** Applied stratified splitting and balanced class weighting to isolate the rare "Premium" minority class.
* **Feature Scaling:** `StandardScaler` utilized to normalize vastly different measurement units (e.g., mg/L vs. g/cm³).
* **Evaluation Metric:** Optimized for ROC-AUC to ensure robust classification thresholding.

##  Explainable AI (SHAP) & Biochemical Insights
A model is only as useful as its transparency. Using **SHapley Additive exPlanations (SHAP)**, grounded in cooperative game theory, the model revealed the precise marginal contribution of each chemical feature:

1. **Alcohol (Primary Driver):** High alcohol content strongly pushes a wine into the premium category, serving as a proxy for optimal grape ripeness, sugar accumulation, and resulting mouthfeel.
2. **Sulphates (Secondary Driver):** Higher sulphate levels positively influence premium classification, mathematically validating their role as essential antioxidants and antimicrobials that preserve wine structure.
3. **Volatile Acidity (Primary Detractor):** The model correctly learned to heavily penalize high volatile acidity, capturing the detrimental effect of acetic acid (vinegar taint) on the palate.

##  Tech Stack
* **Language:** Python 3
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`
