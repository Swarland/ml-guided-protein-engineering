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