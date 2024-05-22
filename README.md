# Credit Card Default Prediction

## 1. Introduction

In the global financial landscape, banking institutions constantly face challenges in managing credit risks, especially concerning customers' inability to make credit card payments. The ability to accurately predict whether a customer will be able to meet their future financial obligations is crucial for ensuring the stability and sustainability of financial services.

In this context, predictive analytics becomes an indispensable tool, employing advanced machine learning techniques and data mining to identify patterns and trends in customer data. Particularly, binary classification emerges as an effective approach to predict default risk, where the objective is to categorize customers into two distinct classes: able or unable to meet the credit payment.

## 2. Objective

This work proposes to explore and develop classifiers to predict whether a given customer will be able to pay (or not) the credit they accrued next month, based on a dataset collected in Taiwan. Using machine learning techniques, the study aims to investigate the relationship between individual customer characteristics and the probability of default, providing valuable insights for financial institutions in strategic decision-making and risk mitigation.

## 3. Methodology

Throughout this work, fundamental aspects of building and evaluating predictive models will be addressed, including the selection and engineering of relevant features, the application of machine learning algorithms, the validation and optimization of model performance, as well as the interpretation of the obtained results.

### 3.1 Evaluated Classifiers

- **SVM Default:** Best accuracy at 82.32%.
- **Gaussian Naive Bayes:** Significantly lower performance with 56.7% accuracy.
- **KNN (K=31):** Best specificity at 97.69%.
- **Fisher LDA:** Best precision at 71.67%.
- **Adaboost:** High specificity of 95.8% but low sensitivity of 34%.

### 3.2 Performance Analysis

- **Accuracy:** SVM Default stood out with 82.32%, while Gaussian Naive Bayes had 56.7%.
- **Specificity:** KNN (K=31) with 97.69%, Gaussian Naive Bayes with 52.67%.
- **Precision:** Fisher LDA with 71.67%, Gaussian Naive Bayes with 29.61%.
- **Sensitivity:** Gaussian Naive Bayes was the best with 71.1%, KNN (K=31) with 9.87%.

## 4. Discussion

### 4.1 Classifier Performance

The SVM models, including SVM with Gridsearch and Gridsearch with Kruskal, showed results similar to the SVM Default across all evaluated metrics, suggesting that parameter tuning did not significantly impact the model's performance.

The Adaboost classifier stood out in specificity (95.8%), effective in identifying default cases, but exhibited low sensitivity (34%).

### 4.2 Dataset Balancing

During the project, it was identified that there was an imbalance in the dataset, with more cases of customers who can pay the loan next month. To address this, balancing methods could be used such as:

- **Under-sampling:** Randomly reducing the size of the larger class.
- **Over-sampling:** Increasing the sample size of the smaller class using techniques like repetition, bootstrapping, or SMOTE (Synthetic Minority Over-Sampling Technique).

## 5. Conclusion

The study demonstrated the importance of choosing and optimizing classifiers in credit default prediction problems. While SVM Default showed the best accuracy, other classifiers like KNN and Fisher LDA excelled in specific metrics. Dataset balancing is a crucial step that can significantly influence model performance.

In summary, our investigation indicates that the most effective classifier for identifying cases where individuals are unable to pay next month is KNN with a k value of 31. Its high specificity of 97% and sensitivity make it a solid choice for this specific task, ensuring reliable detection of negative cases. 

However, if the goal is to identify cases where individuals can pay next month, the Gaussian Naive Bayes classifier might be preferable. Despite lower results compared to KNN for class 0, Naive Bayes demonstrated superior capability in handling imbalanced datasets, which can be crucial when identifying positive cases. 

Thus, the choice of the most appropriate classifier will depend on the specific project needs and priorities in terms of identifying positive or negative cases. Both classifiers have their advantages and limitations.
