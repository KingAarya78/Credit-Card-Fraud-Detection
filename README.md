# Credit Card Fraud Detection System

A robust machine learning system designed to detect fraudulent credit card transactions using advanced classification techniques, anomaly detection, and interactive visualizations.

## 🌟 Features

- **Machine Learning Classification**: Random Forest model optimized for imbalanced credit card fraud detection
- **Cost-Based Optimization**: Custom threshold optimization to minimize business costs of fraud
- **Hybrid Detection**: Combines supervised learning with anomaly detection for improved results
- **Explainable AI**: Uses SHAP values to provide insights into model decisions
- **Interactive Dashboard**: Beautiful visualization of results using Matplotlib and Tkinter
- **Simulation**: Includes a transaction monitor to simulate real-time fraud detection

## 📋 Requirements

- Python 3.8+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- imbalanced-learn
- SHAP (optional, for model explanations)
- PIL/Pillow
- tkinter

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/KingAarya78/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection

# Create and activate a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 📊 Dataset

The system is designed to work with the [Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud) from Kaggle. This dataset contains transactions made by credit cards in September 2013 by European cardholders, with features obtained through PCA transformation for confidentiality.

Place the `creditcard.csv` file in the root directory of the project before running.

## 💻 Usage

```bash
# Run the main detection system
python credit_card_fraud_detector.py

# Optionally, you can run with specific configuration
python credit_card_fraud_detector.py --threshold 0.8 --no-shap
```

## 🧪 How It Works

The fraud detection system follows these key steps:

1. **Data Preprocessing**: Loads and cleans transaction data, scales features, adds cyclical time features
2. **Class Imbalance Handling**: Uses SMOTE and undersampling techniques to handle the imbalanced nature of fraud data
3. **Model Training**: Trains a Random Forest model with optimized parameters
4. **Threshold Optimization**: Sets the classification threshold to minimize business costs
5. **Anomaly Detection**: Adds anomaly scores using Local Outlier Factor for hybrid detection
6. **Visualization**: Creates ROC curves, feature importance plots, and SHAP explanations
7. **Report Generation**: Saves a detailed markdown report of performance metrics
8. **Interactive Dashboard**: Displays results in an intuitive dashboard

## 📊 Performance Metrics

The system is evaluated using metrics especially relevant for fraud detection:

- **ROC AUC Score**: Measures overall classification performance
- **Business Cost**: Combines costs of false positives and false negatives
- **Recall**: Measures ability to detect actual fraud (minimizing false negatives)
- **Precision**: Measures accuracy of fraud predictions (minimizing false positives)
- **F1 Score**: Harmonic mean of precision and recall

## 📁 Output

The system generates several outputs in the `output` directory:

- `roc_curve.png`: ROC curve visualization
- `feature_importance.png`: Top features influencing fraud predictions
- `fraud_explanation.png`: SHAP explanation for a sample fraud case
- `fraud_report.md`: Complete performance report

![roc_curve](https://github.com/user-attachments/assets/964c3812-20cd-48ff-b925-cb83075465ad)


## 🔧 Configuration

You can customize the following parameters in the script:

- `RANDOM_STATE`: For reproducibility (default: 42)
- `MAX_SAMPLES`: Maximum samples for LOF training (default: 10000)
- `ENABLE_SHAP`: Enable/disable model explanations (default: True)
- `ENABLE_HYBRID`: Enable/disable hybrid detection (default: True)

## 📝 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UC Irvine Machine Learning Repository for providing datasets
- The scikit-learn and imbalanced-learn teams for their excellent libraries
- SHAP library authors for model explainability tools
