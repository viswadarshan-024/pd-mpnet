# Fine-Tuned MPNet Model for Paraphrase Detection

### Model Description
This is a fine-tuned version of **MPNet-base** for **paraphrase detection**, trained on four benchmark datasets: **MRPC, QQP, PAWS-X, and PIT**. The model is optimized for **fast inference speed** while maintaining high accuracy, making it suitable for applications like **duplicate content detection, question answering, and semantic similarity analysis**.

- **Developed by:** Viswadarshan R R  
- **Model Type:** Transformer-based Sentence Pair Classifier  
- **Language:** English  
- **Finetuned from:** `microsoft/mpnet-base`

### Model Sources

- **Repository:** [Hugging Face Model Hub](https://huggingface.co/viswadarshan06/pd-mpnet/)  
- **Research Paper:** _Comparative Insights into Modern Architectures for Paraphrase Detection_ (Accepted at ICCIDS 2025)  
- **Demo:** (To be added upon deployment)

## Uses

### Direct Use
- Identifying **duplicate questions** in FAQs and customer support.  
- Enhancing **semantic search** in information retrieval systems.  
- Improving **document deduplication** and content moderation.

### Downstream Use
The model can be further fine-tuned on domain-specific paraphrase datasets (e.g., medical, legal, or finance).

### Out-of-Scope Use
- The model is not designed for multilingual paraphrase detection since it is trained only on English datasets.
- May not perform well on low-resource languages without additional fine-tuning.

## Bias, Risks, and Limitations

### Known Limitations
- Struggles with idiomatic expressions: The model finds it difficult to detect paraphrases in figurative language.
- Contextual ambiguity: May fail when sentences require deep contextual reasoning.

### Recommendations
Users should fine-tune the model with additional cultural and idiomatic datasets for improved generalization in real-world applications.

## How to Get Started with the Model

To use the model, install **transformers** and load the fine-tuned model as follows:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load the tokenizer and model
model_path = "viswadarshan06/pd-mpnet"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Encode sentence pairs
inputs = tokenizer("The car is fast.", "The vehicle moves quickly.", return_tensors="pt", padding=True, truncation=True)

# Get predictions
outputs = model(**inputs)
logits = outputs.logits
predicted_class = logits.argmax().item()
print("Paraphrase" if predicted_class == 1 else "Not a Paraphrase")
```

## Training Details
This model was trained using a combination of four datasets:

- **MRPC**: News-based paraphrases.
- **QQP**: Duplicate question detection.
- **PAWS-X**: Adversarial paraphrases for robustness testing.
- **PIT**: Short-text paraphrase dataset.

### Training Procedure

- **Tokenizer**: MPNetTokenizer
- **Batch Size**: 16
- **Optimizer**: AdamW
- **Loss Function**: Cross-entropy

#### Training Hyperparameters
- **Learning Rate**: 2e-5
- **Sequence Length**:
  - MRPC: 256
  - QQP: 336
  - PIT: 64
  - PAWS-X: 256

#### Speeds, Sizes, Times

- **GPU Used**: NVIDIA A100
- **Total Training Time**: ~6 hours
- **Compute Units Used**: 80

### Testing Data, Factors & Metrics
#### Testing Data

The model was tested on combined test sets and evaluated on:
- Accuracy
- Precision
- Recall
- F1-Score
- Runtime

### Results

## **MPNet Model Evaluation Metrics**
| Model   | Dataset     | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Runtime (sec) |
|---------|------------|-------------|--------------|------------|-------------|---------------|
| MPNet | MRPC Validation | 87.01 | 86.45 | 96.06 | 91.00 | 5.79 |
| MPNet | MRPC Test | 86.03 | 85.56 | 95.03 | 90.05 | 24.75 |
| MPNet | QQP Validation | 89.04 | 82.34 | 89.26 | 85.66 | 7.30 |
| MPNet | QQP Test | 88.98 | 82.95 | 88.65 | 85.70 | 17.77 |
| MPNet | PAWS-X Validation | 95.15 | 92.94 | 96.06 | 94.47 | 7.66 |
| MPNet | PAWS-X Test | 95.35 | 93.39 | 96.58 | 94.96 | 7.75 |
| MPNet | PIT Validation | 83.92 | 81.70 | 70.48 | 75.68 | 7.50 |
| MPNet | PIT Test | 89.50 | 75.74 | 73.14 | 74.42 | 1.57 |

### Summary
This **MPNet-based Paraphrase Detection Model** has been fine-tuned on four benchmark datasets: **MRPC, QQP, PAWS-X, and PIT**, enabling **fast and efficient paraphrase detection** across diverse linguistic structures. The model offers superior inference speed while maintaining high accuracy, making it ideal for applications requiring real-time **semantic similarity analysis and duplicate detection**.

### **Citation**  

If you use this model, please cite:  

```bibtex
@inproceedings{viswadarshan2025paraphrase,
   title={Comparative Insights into Modern Architectures for Paraphrase Detection},
   author={Viswadarshan R R, Viswaa Selvam S, Felcia Lilian J, Mahalakshmi S},
   booktitle={International Conference on Computational Intelligence, Data Science, and Security (ICCIDS)},
   year={2025},
   publisher={IFIP AICT Series by Springer}
}
```

## Model Card Contact

📧 Email: viswadarshanrramiya@gmail.com

🔗 GitHub: [Viswadarshan R R](https://github.com/viswadarshan-024)
