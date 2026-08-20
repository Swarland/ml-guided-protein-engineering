# ml-guided-protein-engineering
Develop machine learning workflow to predict experimentally measured protein stability and effects of amino acid variation on stability 

Data was taken from 
Tsuboyama et al. 2023
https://zenodo.org/records/7992926





August 13th 
- Engineered sequence and biochemical mutation features

August 14th 
- Established Ridge, Random Forest, and XGBoost baselines
- Generated residue-level ESM-2 embeddings for WT and mutant sequences

August 16th 
- Trained models on mutation-induced changes in ESM embeddings
- Frozen ESM embeddings + Ridge achieved R² = 0.28 on held-out protein backgrounds
- Frozen ESM embeddings + MLP achieved R² = 0.447 on held-out protein backgrounds


![ESM embeddings + MLP Prediction results](/figures/esm_net_predresult.png)


August 20th
Fine tuning the ESM-2 model by unfreezing layer and adding a regression head resulted in greatest improvment so far. However additional hyperparameter tuning will be quite computationally intensive. 

- Biochemical Ridge             R² ~0.13
- Biochemical Random Forest     R² ~0.38
- Frozen ESM + Ridge            R² ~0.28
- Frozen ESM + MLP              R² ~0.45
- Partial ESM fine-tuning       R² ~0.53

These results demonstrate that even the most basic transfer learning applied on the ESM-2 model yields by far the best performance.