# ACP-CapsPred

**A model for ACP classification and feature extraction**

## abstract
Cancer is a severe illness that significantly threatens human life and health. Anticancer peptides (ACPs) represent a promising therapeutic strategy for combating cancer. In silico methods enable rapid and accurate identification of ACPs without extensive human and material resources. This study proposes a two-stage computational framework called ACP-CapsPred, which can accurately identify ACPs and characterise their functional activities across different cancer types. ACP-CapsPred integrates a protein language model with evolutionary information and physicochemical properties of peptides, constructing a comprehensive profile of peptides. ACP-CapsPred employs a next-generation neural network, specifically capsule networks, to construct predictive models. Experimental results demonstrate that ACP-CapsPred exhibits satisfactory predictive capabilities in both stages, reaching state-of-the-art performance. In the first stage, ACP-CapsPred achieves accuracies of 80.25\% and 95.71\%, as well as F1-scores of 79.86\% and 95.90\%, on benchmark datasets Set 1 and Set 2, respectively. In the second stage, tasked with characterising the functional activities of ACPs across five selected cancer types, ACP-CapsPred attains an average accuracy of 90.75\% and an F1-score of 92.38\%. Furthermore, ACP-CapsPred demonstrates excellent interpretability, revealing regions and residues associated with anticancer activity. Consequently, ACP-CapsPred presents a promising solution to expedite the development of ACPs and offers a novel perspective for other biological sequence analyses.

## Conclusion

In this study, we proposed a two-stage computational framework, ACP-CapsPred, for predicting ACPs and their targets across various cancers. The first stage of ACP-CapsPred is dedicated to identifying ACPs, while the second stage focuses on predicting the functional activity of ACPs against different types of cancer. ACP-CapsPred integrates protein language models to extract residue embeddings and incorporates the physicochemical properties and evolutionary features of peptides to construct a comprehensive profile of peptides. Employing a next-generation neural network, specifically capsule networks, ACP-CapsPred demonstrates  state-of-the-art performance in both stages. The model exhibits notable interpretability, aiding researchers in understanding regions within the sequences associated with anticancer activity. ACP-CapsPred's convincing performance in predicting ACPs and their targets presents a novel option to expedite the development of anticancer peptides.



## Model

![mode](model.jpg)

## 1.Train

```
python3 train.py
```



**please make sure that cuda is avaliable**

## 2.Evaluate

```
python3 evaluate.py
```

## 3.Best model

our model available: [Model](https://awi.cuhk.edu.cn/dbAMP/download/software/best_model.zip)

## 4.Dataset
```
Data\raw_data
```
