# PyTorch Fake News Detector

## Overview
This project implements a **Deep Learning-based Fake News Detection System** using PyTorch and LSTM (Long Short-Term Memory) networks. The model is trained to classify news articles as either **Real** or **Fake** based on their textual content and subject matter.

## Project Objectives
- Build and train a bidirectional LSTM model for binary classification (Real vs. Fake news)
- Perform comprehensive data preprocessing and exploratory data analysis
- Conduct hyperparameter tuning and model comparison
- Provide explainability insights using Integrated Gradients (XAI)
- Evaluate model performance through multiple metrics and error analysis

## Dataset

### Data Source
The project uses two separate datasets:
- **Fake News Dataset**: 23,481 articles labeled as fake
- **Real News Dataset**: 21,417 articles labeled as real
- **Total Combined Dataset**: 44,898 news articles

### Dataset Structure
Each article contains the following features:
- `title`: Article headline
- `text`: Full article content
- `subject`: Category (e.g., politics, news)
- `date`: Publication date
- `label`: Binary classification (0 = Fake, 1 = Real)

## Data Preprocessing & Exploration

### Exploratory Data Analysis (EDA)
1. **Class Distribution**: Analyzed the balance between fake (52.3%) and real (47.7%) news
2. **Text Length Distribution**: Examined the distribution of article lengths for both classes
3. **Subject Distribution**: Visualized the distribution of articles across different news categories

### Text Preprocessing Steps
1. Converted text to lowercase
2. Removed URLs and special characters
3. Removed punctuation and extra whitespace
4. Removed numbers and square brackets
5. Created combined text field merging title and text
6. Generated cleaned text for model input

## Model Architecture

### Core Model: Bidirectional LSTM
- **Type**: Bidirectional Long Short-Term Memory (Bi-LSTM)
- **Embedding Dimension**: 100
- **Hidden Dimension**: 256
- **Number of Layers**: 2
- **Dropout Rate**: 0.3
- **Bidirectional**: Yes (processes text in both directions)
- **Output Layer**: Fully connected layer with sigmoid activation for binary classification

### Model Performance
- **Best Validation Accuracy**: **99.88%**
- **Training Accuracy**: 99.72%
- **Loss Function**: Binary Cross-Entropy with Logits (BCEWithLogitsLoss)
- **Optimizer**: Adam (learning rate: 0.001)
- **Epochs**: 3 (achieved convergence quickly)

## Data Splitting
Data was split into three sets:
- **Training Set**: 60% (26,938 samples)
- **Validation Set**: 20% (8,980 samples)
- **Test Set**: 20% (8,980 samples)

## Model Comparison & Hyperparameter Tuning

Multiple LSTM configurations were tested:
1. **Unidirectional LSTM**: Baseline model
2. **Bidirectional LSTM**: Enhanced model with bidirectional processing
3. **Different Hidden Dimensions**: 128, 256, 512
4. **Varying Dropout Rates**: 0.2, 0.3, 0.5

**Result**: Bidirectional LSTM with hidden dimension 256 and dropout 0.3 achieved the best performance.

## Error Analysis

### Misclassification Summary
- **False Positives**: 3 (Real news incorrectly classified as Fake)
- **False Negatives**: 11 (Fake news incorrectly classified as Real)
- **Total Misclassifications**: 14 out of 8,980 test samples (0.156% error rate)

### Error Distribution Analysis
- Analyzed text length distribution of misclassified samples
- Identified patterns in false positives and false negatives
- Examined feature importance for incorrect predictions

## Ablation Study

### Experiment: Stop Words Removal
- **Original Model Accuracy**: 99.88%
- **Ablated Model (without stop word removal) Accuracy**: 99.86%
- **Finding**: Stop word removal has minimal impact on model performance, suggesting the LSTM effectively learns contextual importance

## Explainability & Interpretability (XAI)

### Integrated Gradients Analysis
Used Captum library to implement Integrated Gradients for model explainability:
- Identified which words contribute most to predictions
- Analyzed feature importance for misclassified samples
- Visualized attribution scores for individual tokens

### Example Analysis
- **False Positive Case**: Real news about "Trump pardons sheriff Joe Arpaio" was misclassified as fake
- **Attribution Insight**: Words like "breaking", "president", and "arpaio" had high influence on the prediction

## Technologies & Libraries

### Core Libraries
- **PyTorch**: Deep learning framework
- **TensorFlow/Keras**: Alternative deep learning components
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-learn**: Machine learning utilities and metrics
- **NLTK**: Natural Language Toolkit for text processing
- **Matplotlib & Seaborn**: Data visualization
- **Captum**: Model interpretability and explainability

## Key Findings

1. **High Accuracy**: Achieved 99.88% validation accuracy on fake news detection
2. **Bidirectional Processing**: Bi-LSTM significantly outperformed unidirectional variants
3. **Low Error Rate**: Only 0.156% misclassification rate on test set
4. **Robust Preprocessing**: Effective text cleaning and normalization
5. **Feature Importance**: Words related to sensational language and certain keywords strongly indicate fake news
6. **Context Matters**: LSTM learns contextual patterns effectively, making stop words less critical

## Model Insights

- Real news articles tend to follow journalistic standards with structured reporting
- Fake news often contains sensational language, emotional triggers, and controversial claims
- The model successfully captures linguistic patterns distinguishing the two classes
- Explainability analysis reveals interpretable decision-making process

## Usage

### Prerequisites
```bash
pip install torch pandas numpy scikit-learn nltk matplotlib seaborn captum
```

### Running the Project
1. Load and explore the fake and real news datasets
2. Preprocess text data (cleaning, tokenization, padding)
3. Build and train the Bi-LSTM model
4. Evaluate on validation and test sets
5. Perform error analysis on misclassified samples
6. Generate explainability visualizations using Integrated Gradients

## Results Summary

| Metric | Value |
|--------|-------|
| Vocabulary Size | 442 unique tokens |
| Maximum Sequence Length | 442 tokens |
| Training Samples | 26,938 |
| Validation Samples | 8,980 |
| Test Samples | 8,980 |
| Best Validation Accuracy | 99.88% |
| False Positives | 3 |
| False Negatives | 11 |
| Total Error Rate | 0.156% |

## Future Improvements

1. **Ensemble Methods**: Combine multiple models for even better performance
2. **Transfer Learning**: Use pre-trained language models (BERT, GPT)
3. **Multi-label Classification**: Classify into multiple fake news categories
4. **Real-time Detection**: Deploy as a web API for real-time fake news detection
5. **Data Augmentation**: Increase dataset size with additional sources
6. **Cross-domain Evaluation**: Test on news from different domains and time periods

## Project Structure

```
PyTorch-FakeNews-Detector/
├── pytorch_project_code.py
├── Fake.csv
├── True.csv
├── README.md
└── requirements.txt
```

## Conclusion

This project successfully demonstrates the application of deep learning for fake news detection. The Bi-LSTM model achieves exceptional accuracy (99.88%) while maintaining interpretability through integrated gradients analysis. The low error rate and comprehensive evaluation showcase the model's reliability for real-world deployment in misinformation detection systems.

## Author
**Rejuwanshamim**

## License
This project is open source and available for academic and research purposes.

---

**Last Updated**: December 28, 2025
