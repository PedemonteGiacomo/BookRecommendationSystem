# Amazon Top 50 Bestselling Books 2009 – 2022 Analysis and Machine Learning
Book Recommendation System with the bestelling books as analysis

## Dataset Overview
Analyzed by Ali Haider and Giacomo Pedemonte, this dataset comprises:
- **700 Books**
- **441 Unique Titles**
- **305 Authors**

### Dataset Features
- **Genres:** Fiction & Non-Fiction
- **No Empty Values:** No cleaning required

## Key Findings

### **1. Genre Analysis**
- A shift towards **Non-Fiction** may be advantageous for writers.
- Top Authors: **Jeff Kinney** (Fiction) & **Gary Chapman** (Non-Fiction).

### **2. User Reviews**
- Significant rise in user reviews, notably in **2020** (potentially due to COVID-19).

### **3. Price Analysis**
- Most bestsellers priced below **$20**.
- **Affordable books attract new and Non-Traditional readers.**
- **Non-Fiction books** consistently priced higher, targeting knowledge-seeking readers.

### **4. Machine Learning Models**

#### Genre Prediction
- **Achieved Precision:** High precision and accuracy in predicting genres based solely on book names.

#### Book Recommendation System
- Implemented **Content-Based Filtering** and **Hybrid Approaches** for precise book recommendations.
- Combined TF-IDF matrix with normalized quantitative data.

#### Price Prediction
- **Challenges:** Due to poor feature correlation, achieving precise price predictions proved difficult.
- **Best Model:** Ridge Regression after tuning hyperparameters.

## Machine Learning Insights

- **Text Features:** Models incorporating text features yielded the most accurate results.
- **Correlation:** Limited correlation between non-text features affected prediction reliability.

## Conclusion

1. **Genre Prediction:** Accurately predicted genres using book names alone.
2. **Recommendation System:** Achieved precise recommendations utilizing text features.
3. **Price Prediction:** Challenges due to feature correlation, yet achieved significant improvement using **Ridge Regression** post **Cross-Validation**.

**Important Note:** The findings suggest that utilizing text features significantly enhances prediction accuracy, while non-text features exhibited poor correlation.

**Thank you for your attention!**

