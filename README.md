Medical Insurance Cost Prediction
Project Overview
Medical Insurance Cost Prediction is a capstone project aimed at building an interpretable and fairness-aware regression model for predicting medical insurance costs using demographic, health, and synthetic lifestyle data. The model focuses on robust performance, subgroup fairness (gender and region), and transparency through explainable machine learning.

Key Objectives
Develop a scalable, transparent predictive pipeline for insurance cost estimation

Increase fairness and interpretability using advanced auditing and SHAP analysis

Augment real datasets with synthetic clinical and lifestyle features for robustness

Enable insurers and stakeholders to access actionable, equitable insights.

Data Description
The dataset includes the following features:

Age, Sex, BMI, Number of Children, Smoking Status, Region, Charges (target)

Feature Engineering: age bins, BMI bins, risk flags, interaction terms

Synthetic Augmentation: Blood pressure, cholesterol, sum insured, lifestyle scores, etc.

Data sources:

GTS Medical Insurance Cost Prediction Dataset

Kaggle Medical Insurance Dataset

Modeling and Fairness Strategy
Baseline model: XGBoost regression

Fairness-aware modeling: Fairlearn Exponentiated Gradient constraints

Custom Hybrid Subgroup models for sensitive groups (smoker/non-smoker)

Post-processing bias correction using TrippleFactorCorrection.

Explainability via SHAP feature importance for stakeholder trust

Performance
Predictive accuracy up to 
R
2
=
0.98
R 
2
 =0.98 after fairness corrections and synthetic augmentation

Fairness audits confirm improved equity across gender and regions

SHAP analysis shows stable, consistent feature importance across modeling stages

Project Structure
explore.py – Exploratory data analysis and visualization

transformation.py – Data preprocessing, feature engineering, and augmentation

FairnessAnalysis.py – Auditing and fairness constraint integration

Capstone Final Presentation.pptx – Project overview and findings

Notebooks, scripts, and utils for modular execution

Getting Started
Clone this repository and place the data files as instructed.

Install dependencies:

text
pip install -r requirements.txt
Run exploratory and modeling scripts (see explore.py, transformation.py).

Review fairness and interpretability analyses in corresponding modules.

For deployment, follow future work guidelines for integration and monitoring.

Results Summary
Combining feature engineering, upsampling, and fairness constraints yields optimal accuracy and ethical deployment.

Fairness correction and SHAP-based explanations help regulatory compliance, consumer trust, and robust insurance pricing.

References
Please see the final presentation and reference section for all citation sources and inspiration.
