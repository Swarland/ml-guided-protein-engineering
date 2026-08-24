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




August 23rd 
Used tuned ESM-2 model to predict ddG for all possible mutations of a protein sequence.

Here are the top 10 most stabilizing substitutions. 
![Top 10 stabilizing substitutions](/figures/top10_stabilizing_subs.png)


However, as expected the vast majority of the residue mutations are destabilizing (negative ddG). 

![Heatmap of stability predictions](/figures/stability_pred_hm.png)

White squares represent the WT residue. We see some residue positions and specific substitutions have far more severe destabilizing predictions than others. Additionally it seems that mutations at the N- and C-terminus seemm to be less destabilizing than mutations in the middle of the sequence. 

Furthermore the Predicted ΔΔG is highly assymetric. As mentioned most residue mutations result in negative ΔΔG ranging from 0 to -5. However in the bar plot above, the most stabilizing mutation resulted in positive shift of 0.54. 

All this is inline with idea that proteins are evolved for stability, and similar to genes most mutations are deleterious (destabilizing).


Notebooks:
1. Curate mutation-stability data
2. Build biochemical ML baselines
3. Use frozen protein-language-model embeddings
4. Fine-tune ESM-2 for ΔΔG
5. Use the trained model to prioritize stabilizing variants