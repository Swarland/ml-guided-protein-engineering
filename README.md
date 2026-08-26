# ml-guided-protein-engineering

Predicting the effects of amino-acid substitution on protein stability using biochemical features, protein language model embeddings, and transfer learning. 

## Overview

Proteins stability is an important consideration when designing de novo proteins. However, testing all configurations experimentally is expensive and impractical. Using exisiting protein language model frameworks, transfer learning can be applied to adapt them to predict the stability of proteins and the effects of various amino-acid substitutions on this stability. This project develops a workflow for predicting the effects of mutation on protein stability (ΔΔG), and using these predictions to select for stabilizing mutations. 

## Workflow

1. **Data Preparation**
- Curated single amino-acid substitution dataset containing experimentally determined stability measurements (ΔΔG) from Tsuboyama et al. 2023 (https://zenodo.org/records/7992926).
- Parsed dataset and added additional biochemical features, mutation identity, residue position, etc


2. **Biochemical feature models**
- Split data into train, validate, test groups, based on WT protein identity to prevent variants of same protein in both train/test sets (data leakage)
- Engineered features containing various biochemical data (hydrophobicity, molecular weight, charge, etc)
- Evaulated using linear, tree based models

3. **Protein language model embeddings**
- Used contextual residue embeddings from ESM-2 model
- Represented mutations as difference between mutant and WT residue embeddings
- Trained Ridge and neural-network regressors on the frozen representations.

4. **Transfer learning**
- Fine tuned ESM-2 model using regression head for ΔΔG prediction
- Used protein-grouped train/validation/test splits and early stopping to train model to generalize to unseen proteins

5. **Variant design**
- Generated all possible single amino-acid substitutions for a held out protein
- Used trained model to predict ΔΔG for each candidate
Ranked candidates by stabilty scores and visualized mutational stability landscape

## Results

Model performance increased with increasingly informative protein representations

| Model | R² | Spearman ρ |
| --- | ---: | ---: |
| Biochemical Ridge | 0.13 | — |
| Biochemical Random Forest | 0.38 | 0.60 |
| Frozen ESM-2 + Ridge | 0.28 | 0.49 |
| Frozen ESM-2 + MLP | 0.45 | 0.64 |
| Fine-tuned ESM-2 | **0.53** | **0.70** |

The final model achieved an MAE of approximately **0.50 kcal/mol** and an RMSE of **0.69 kcal/mol** on unseen proteins during final evaluation. 

Validation curves flatlined while training curves continued decreasing, indicating substantial overfitting to the training-set. However the fine tuned model still substantially outperformed the other models. 

Hyperparameter tuning and unfreezing additional layers of the ESM-2 model although computationally intensive would likely improve R² further.

## In Silico Mutation Screen

For a selected held-out protein, at each residue all 19 possible amino-acid substitutions evaluated for their effect on the protein stability using the Fine-tuned ESM-2 model. 

![Heatmap of stability predictions](/figures/stability_pred_hm.png)

This figure represents the mutational landscape of the help out protein. For all residues in the protein (y-axis), the effects on stability of each possible amino acid substitution (x-axis) are shown. White squares represent the WT residue. 

Substitution in the chosen protein at residue positions (25, 38, 42) and to specific amino-acids (P) are predicted to be far more destabilizing than others. 

Predicted ΔΔG is highly assymetric. As expected by evolutionary theory, most residue mutations result in negative ΔΔG ranging from 0 to -5. However in the bar plot above, the most stabilizing mutation resulted in positive shift of 0.54. 


## Repository Structure

ml-guided-protein-engineering/
├── notebooks/
│   ├── 01_explore_stability_data.ipynb
│   ├── 02_sequence_features.ipynb
│   ├── 03_protein_embeddings.ipynb
│   ├── 04_model_comparison.ipynb
│   └── 05_variant_design.ipynb
├── src/
│   ├── data.py
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── features.py
│   ├── models.py
│   └── train.py
├── figures/
├── environment.yml
└── pyproject.toml

## Methods and Technologies

Python · PyTorch · scikit-learn · XGBoost · Hugging Face Transformers ·
ESM-2 · pandas · NumPy · matplotlib

## Data

Protein stability measurements and sequence data was derived from Tsuboyama et al. 2023 (https://zenodo.org/records/7992926). Additional biochemical features were taken from Kyte & Doolittle (1982) and the International Union of Pure and Applied Chemistry (IUPAC). 

Raw and processed data are excluded from this repository due to size constraints. 
