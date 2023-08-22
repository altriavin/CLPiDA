# CLPiDA: A Contrastive Learning Approach for Predicting Potential PiRNA-Disease Associations
In this study, we present CLPiDA, a novel method for predicting potential piRNA-disease associations. CLPiDA begins by computing Gaussian kernel similarities for piRNA-piRNA and disease-disease pairs to establish initial embeddings for piRNAs and diseases. Subsequently, it employs a parameter-sharing online and target network, along with data augmentation techniques, to create a contrastive learning framework. This facilitates the generation of embeddings for piRNAs and diseases using piRNA-disease association pairs. Furthermore, CLPiDA employs a cross-prediction approach to determine association scores for specific piRNAs and diseases. Notably, CLPiDA introduces a novel approach by excluding negative samples, thereby avoiding the introduction of false negatives and enhancing both its reliability and predictive efficiency.
# Requirements
- torch 1.10.1
- python 3.7.13
- numpy 1.21.6
- scikit-learn 1.0.2
# Data
In this study, our model underwent a comprehensive evaluation and testing phase utilizing the MNDR v3.0 database. The dataset was meticulously curated to ensure data quality, involving the removal of duplicate associations. Specifically, we focused on human-related piRNA-disease associations, resulting in a refined dataset comprising 11,981 experimentally validated instances. This encompassed interactions between 10,149 distinct piRNAs and 19 distinct diseases, forming the foundation for our investigative efforts.

For a more robust validation of CLPiDA's performance, an independent test set was meticulously curated to conduct a comprehensive evaluation of the model's capabilities. A total of 2,489 piRNA-disease association pairs were meticulously collected from relevant literature sources. This independent test set encompassed interactions involving 2,415 distinct piRNAs and 13 distinct diseases.

# Run the demo
```
python main.py
```
