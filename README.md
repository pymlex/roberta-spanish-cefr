# Spanish CEFR Classification with BERTIN

## Model summary

`pymlex/roberta-spanish-cefr` is a Spanish text classifier fine-tuned from `bertin-project/bertin-roberta-base-spanish` for CEFR level prediction. It is intended for Spanish learner-text classification and readability-style proficiency assessment.

## Training data

The model was trained on `UniversalCEFR/caes_es`, a Spanish dataset of learner texts with CEFR annotations. The dataset viewer shows 31.1k rows and Spanish language.

<img width="691" height="365" alt="image" src="https://github.com/user-attachments/assets/73d297ad-8817-43c5-bfce-68ad0cb7e9b2" />

## Evaluation

Results for the test set:

* Accuracy: 0.9882
* Precision: 0.9896
* Recall: 0.9892
* F1: 0.9894

## Comparison with other CEFR Spanish classifiers

Our model's performance (F1: 0.9894) is SOTA. Most documented Spanish CEFR classifiers fall within the 0.75 – 0.88 F1-score range. The obtained results significantly outperform these common baselines:

| Model / Source | Task / Language | Accuracy | F1-Score |
|---|---|---|---|
| This model (BERTIN-RoBERTa) | Spanish CEFR (6 classes) | 0.9882 | 0.9894 |
| Spanish CEFR Fine-tuned[](https://www.researchgate.net/figure/Performance-metrics-of-the-fine-tuned-model-across-CEFR-levels_tbl4_398474670) | CEFR Spanish (General) | ~0.8500 | 0.83–0.85 |
| BETO/mBERT Baseline[](https://www.researchgate.net/figure/Performance-of-all-models-on-the-Spanish-language-dataset-for-skill-classification-ACC_tbl4_389648000) | Spanish Skill Classif. | 0.7800 | 0.7700 |
| CEFR-ASAG Benchmark[](https://www.cambridge.org/core/journals/recall/article/predicting-cefr-levels-in-learners-of-english-the-use-of-microsystem-criterial-features-in-a-machine-learning-approach/C915A35CD69168EDFB80DE8F57A4328C) | Multi-level (Cross-corpus) | 0.5100 | — |
| IberLEF / Shared Tasks[](https://ceur-ws.org/Vol-3202/parmex-paper3.pdf) | Related Spanish NLP tasks | 0.9373 | 0.9300 |

## Inference

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "pymlex/roberta-spanish-cefr"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.eval()

def predict_cefr(text, top_k=3):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    k = min(top_k, probs.numel())
    values, indices = torch.topk(probs, k=k)

    return [
        {
            "label": model.config.id2label[i.item()],
            "score": float(v.item()),
        }
        for i, v in zip(indices, values)
    ]

text = "Estimados señores, les escribo para solicitar información sobre el curso."
print(predict_cefr(text, top_k=3))
