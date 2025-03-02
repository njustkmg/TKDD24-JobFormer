# JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer

This is the official PyTorch implementation of the paper "**JobFormer: Skill-Aware Job Recommendation with Semantic-Enhanced Transformer**". 

💥 In real-world management scenarios, job recommendation faces challenges due to the information deficit of JDs and the heterogeneity between JDs and user profiles. To tackle these challenges, we propose a skill-aware recommendation model based on a semantic-enhanced Transformer, which models JD-related items, leverages a local-global attention mechanism to capture intra- and inter-job dependencies, and adopts a two-stage learning strategy that utilizes skill distributions for JD representation learning in recall and integrates user profiles for final ranking.

### Requirements

- torch1.7.1+cu110
- torchvision 0.8.2
- torchaudio 0.7.2

### Train and Evaluation

python main.py # Recall Experiment
python finetune.py # Ranking Experiment

### Citation

If you find this code to be useful for your research, please consider citing.

```
@article{Jobformer,
  title={Jobformer: Skill-aware job recommendation with semantic-enhanced transformer},
  author={Zhihao Guan, Jia-Qi Yang, Yang Yang, Hengshu Zhu, Wenjie Li, and Hui Xiong},
  journal={ACM Transactions on Knowledge Discovery from Data},
  year={2024}
}
```
