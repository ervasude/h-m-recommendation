# H&M Personalized Fashion Recommendations️

## Problem Statement

**Objective:** Predict which articles each customer will purchase in the 7-day period immediately after the training data ends.

**Business Context:**
- H&M needs to predict customer purchases for inventory planning and personalized marketing
- Customers who don't make purchases during the test period are excluded from scoring
- This is a ranking problem, not classification

**Evaluation Metric:** 
- **MAP@12** (Mean Average Precision at 12)
- We predict exactly 12 articles per customer (even if they buy fewer items)
- Higher precision at better ranks = higher score

## Dataset Overview

**Source:** Kaggle H&M Personalized Fashion Recommendations Competition

**Dataset Components:**
- `transactions_train.csv`: Customer purchase history (training data)
- `articles.csv`: Product metadata (name, type, color, department, etc.)
- `customers.csv`: Customer information (age, club membership, etc.)
- `images/`: Product images for visual feature extraction

**Data Scale:**
- ~60GB total (including images)
- Training period: Historical transactions
- Test period: 7 days after training data ends

##️ Approach

**Phase 1: Understanding & Baselines**
1. Exploratory Data Analysis (EDA)
   - Customer behavior patterns
   - Product popularity distribution
   - Seasonal trends and patterns

2. Feature Engineering
   - Customer features: purchase frequency, recency, preferences
   - Article features: popularity, repeat rate, metadata
   - Interaction features: co-purchase patterns

3. Baseline Model
   - Recommend most popular articles to all customers
   - Establishes performance benchmark

**Phase 2: Advanced Modeling**
4. Visual Feature Extraction
   - Use pre-trained ResNet-50 to extract 512-dim embeddings from product images
   - Create visual similarity index

5. Hybrid Recommendation System
   - Combine text-based signals (collaborative filtering)
   - Combine visual similarity signals
   - Weight optimization for best MAP@12 score

6. Optimization & Final Submission
   - Test different weight combinations
   - Handle edge cases (new customers, missing images)
   - Final Kaggle submission

##  Results

(Coming soon after model training and Kaggle submissions)

##  How to Reproduce

### Setup
```bash
# Create virtual environment
python3 -m venv h_m_env
source h_m_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run analysis
jupyter notebook
```

### Running Notebooks
Execute in order:
1. `01_data_loading_exploration.ipynb` - Load and explore data
2. `02_customer_analysis.ipynb` - Understand customer behavior
3. `03_product_analysis.ipynb` - Analyze products
4. `04_feature_engineering.ipynb` - Create ML features
5. `05_baseline_model.ipynb` - Establish baseline
6. `06_visual_features.ipynb` - Extract image embeddings
7. `07_hybrid_model.ipynb` - Build hybrid recommender
8. `H&M_Final_Solution.ipynb` - Final optimized model

## Project Structure
```
h-m-recommendation/
├── data/
│   ├── raw/                          # Original dataset files
│   │   ├── transactions_train.csv
│   │   ├── articles.csv
│   │   ├── customers.csv
│   │   └── images/                   # Product images
│   └── processed/                    # Engineered features & embeddings
├── notebooks/
│   ├── 01_data_loading_exploration.ipynb
│   ├── 02_customer_analysis.ipynb
│   ├── 03_product_analysis.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_baseline_model.ipynb
│   ├── 06_visual_features.ipynb
│   ├── 07_hybrid_model.ipynb
│   └── H&M_Final_Solution.ipynb
├── submissions/
│   ├── baseline_submission.csv       # Baseline results
│   ├── v1_submission.csv             # First iteration
│   └── final_submission.csv          # Best model
├── src/
│   ├── data_loader.py               # Data loading utilities
│   ├── feature_engineer.py          # Feature engineering
│   └── recommender.py               # Recommendation models
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

## 🔧 Technologies Used

- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **ML & Recommendation:** scikit-learn, faiss
- **Deep Learning:** PyTorch, torchvision (ResNet-50)
- **Notebooks:** Jupyter

##‍ Author

**Sude** - Data Science & ML Portfolio
- GitHub: [@sude77868](https://github.com/sude77868)
- Project: H&M Kaggle Competition

##  References

- [Kaggle Competition Page](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
- MAP@K Metric: [Information Retrieval Evaluation](https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Mean_average_precision)
- ResNet-50: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
