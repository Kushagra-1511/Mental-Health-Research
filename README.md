Mental Health Risk Assessment from Text: A Deep Learning Approach
This repository contains the code for a research paper focused on analyzing social media posts to assess mental health risk levels. The project uses natural language processing (NLP) and deep learning techniques to classify text into different risk categories and predict a numerical risk score.

Project Overview
The code implements a hybrid deep learning model that performs both classification and regression tasks:

Classification: Categorizing posts into mental health risk levels (Supportive, Ideation, Behavior, Attempt, Indicator)
Regression: Predicting a continuous risk score (0-1) indicating severity
Dataset
The model is trained on a dataset of Reddit posts labeled with mental health risk categories. The data includes:

500 Reddit users' posts
Risk labels (Supportive, Ideation, Behavior, Attempt, Indicator)
Text content of posts
Technical Implementation
The pipeline includes:

Data Preprocessing:

Text cleaning (URL removal, special character handling)
Emoji counting
Post expansion and normalization
Feature Extraction:

DistilBERT embeddings for semantic representation
Emoji count features
Model Architecture:

Multi-output neural network with shared layers
Two output heads:
Classification head (softmax for risk category)
Regression head (sigmoid for risk score)
Evaluation:

Classification metrics: accuracy, precision, recall, F1
Regression metrics: MSE, MAE
Confusion matrix visualization
Risk score prediction analysis
Requirements
The code requires the following Python libraries:

TensorFlow/Keras
Transformers (Hugging Face)
Pandas
NumPy
Matplotlib
Scikit-learn
Emoji
Results
The model achieves:

Classification accuracy: ~36%
Risk score prediction MAE: ~0.22
The system demonstrates ability to capture nuanced mental health risk signals from text, though challenges remain in accurately differentiating between closely related risk categories.

Future Work
Potential improvements include:

Fine-tuning the DistilBERT model for the specific task
Incorporating additional metadata features
Exploring ensemble methods
Adding attention mechanisms to better capture key risk signals
Citation
If you use this code for your research, please cite our paper:


Copy
@article{mentalhealth2023,
  title={Mental Health Risk Assessment from Text: A Deep Learning Approach},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2023}
}
License
This project is licensed under the MIT License - see the LICENSE file for details.
