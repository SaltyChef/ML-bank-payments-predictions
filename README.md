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

Taking into account our initial objective of assisting the bank in identifying cases where individuals will not be able to pay the loan next month (class 0), it is crucial to place greater importance on classifiers that demonstrate higher specificity since the bank will want to guard against cases where a person cannot pay the loan but is marked as able to pay. However, other metrics associated with each classifier should not be disregarded.

In terms of accuracy, the SVM Default classifier stands out as the best, with a rate of 82.32%, while the Gaussian Naive Bayes shows significantly inferior performance, with only 56.7%. This discrepancy can be attributed to the simplistic nature of the naive Bayes model, which assumes independence between variables, which may not be the case in our dataset.

Analyzing specificity, KNN with K=31 emerges as the best classifier, achieving an impressive rate of 97.69%. On the other hand, the Gaussian Naive Bayes again shows not very optimal performance, with a specificity of only 52.67%.

Regarding precision, Fisher LDA stands out as the best classifier, with a rate of 71.67%, while Gaussian Naive Bayes continues to perform unsatisfactorily, with only 29.61%. This disparity can be attributed to the tendency of the Naive Bayes model to underestimate the probability of certain classes, resulting in low precision.

Finally, in terms of sensitivity, Gaussian Naive Bayes presents itself as the best classifier, with a rate of 71.1%, while KNN with K=31 shows a sensitivity of only 9.87%. A possible explanation for this result is the ability of Naive Bayes to better handle unbalanced datasets, as is the case here, where class 0 is predominant.

Besides the results of the classifiers mentioned earlier, it is essential to highlight the performance of other developed classifiers. The different SVM models, including SVM with Gridsearch and Gridsearch with Kruskal, showed results quite similar to SVM Default in all evaluated metrics. The reason could be that the adjustments in the SVM parameters did not have a significant impact on the model's performance.

On the other hand, the Adaboost classifier stood out with a specificity of 95.8%, making it particularly effective in identifying cases where individuals will not be able to pay the loan next month, thus minimizing false positives. However, it is important to note that Adaboost demonstrated a relatively low sensitivity of 34%, meaning it may not be as effective in capturing all positive cases, resulting in a higher rate of false negatives.

It is also of great relevance to note that throughout the project, the group failed to balance the dataset, as there are more cases where the client can pay the loan next month. 

For this balancing, under-sampling could be done by randomly reducing the size of the larger class, testing with various balanced datasets to find the best one that shows better results during the experiments. The same could be done for over-sampling, which is used when the amount of data in one of the classes is insufficient. Here the method tries to balance the dataset by increasing the sample size using repetition, bootstrapping, or SMOTE (Synthetic Minority Over-Sampling Technique).

## 5. Conclusion

The study demonstrated the importance of choosing and optimizing classifiers in credit default prediction problems. While SVM Default showed the best accuracy, other classifiers like KNN and Fisher LDA excelled in specific metrics. Dataset balancing is a crucial step that can significantly influence model performance.

In summary, our investigation indicates that the most effective classifier for identifying cases where individuals are unable to pay next month is KNN with a k value of 31. Its high specificity of 97% and sensitivity make it a solid choice for this specific task, ensuring reliable detection of negative cases. 

However, if the goal is to identify cases where individuals can pay next month, the Gaussian Naive Bayes classifier might be preferable. Despite lower results compared to KNN for class 0, Naive Bayes demonstrated superior capability in handling imbalanced datasets, which can be crucial when identifying positive cases. 

Thus, the choice of the most appropriate classifier will depend on the specific project needs and priorities in terms of identifying positive or negative cases. Both classifiers have their advantages and limitations.
