# Predicting Customer Purchases

**Task:**

We currently work within the data science team of an automotive company, who subsidises electric eco-friendly cars to companies. For one of our clients, the company has sent across employee data and whether they have put an order through. Our task is to see whether we can see any trends and produce a machine learning model that can predict whether a customer is likely to purchase the car.   

We will be completing this business task using Logistic Regression, K Nearest Neighbours, Support Vector Machine, Naive Bayes, Classification Trees and XGBoost.

### Confusion Matrix

We will be evaluating our model performance by using the confusion matrix. 

**Type I Error:**

A type I error occurs when we have predicted a positive outcome (1 or yes) but in reality it was negative (0 or no), therefore predicting an effect that did not occur. Type I errors are also referred to as "False Positives".

**Type II Error:**

A type II error occurs when we have predicted a negative outcome (0 or no) but in reality it was positive (1 or yes), therefore an unexpected effect occuring. Type II errors are also referred to as "False Negatives".

Type II errors are deemed more serious than type I errors. This is because there could be dire consequences if we have predicted an event such as an earthquake/disease not occuring, but in fact it did occur; leading to unnecessary injury and fatalities.

The diagram below shows a confusion matrix with the accuracy calculated.

<img src = 'Screen1.png' width='700'>

However, we should not rely on the confusion matrix as in some cases we can arrive at the incorrect conclusions.

**Accuracy Paradox**

Let's take a look at the confusion matrix below. Out of our 10,000 predictions only 200 were incorrect, giving us an accuracy of 98%, a very accurate model indeed. 

<img src = 'Screen2.png' width='700'>

However, if we abandon the model and set all predictions to 0, we get an accuracy of 98.5% which indicates a better accuracy compared to when using the classification model. The diagram below illustrates this. 

<img src = 'Screen3.png' width='700'>

For example, if a business sold the most luxurious products where only a small percentage of the population could afford, setting all predictions to zero would mean the business should shut down and not sell to anyone as this increases model accuracy. This case occurs when there is a very low number of observations that are positive (1 or yes) in comparison to the observations that are negative (0 or no). We should therefore always use the confusion matrix as an indicator of accuracy but apply our logic so we can interpret the confusion matrix and arrive at the correct conclusion.

## Logistic Regression

**Pros of Logistic Regression:**
- Probabilistic approach
- Gives information about statistical significance of features

**Cons of Logistic Regression:**

- Logistic Regression assumptions

**Assumptions of the Logistic Regression:**
- Appropriate outcome structure (Binary logistic regression requires a binary dependent variable, and an ordinal logistic regression requires an ordinal dependent variable)
- Observation independence (The observations are independent of each other)
- Lack of multicollinearity (The independent variables should not be highly correlated with each other)
- Linearity of independent variables and log odds (The independent variables are linearly related to the log odds)
- Large sample size

The diagram below illustrates the training set. The seperate regions (red and green) are called the predictive regions. The line that seperates the regions is called the predictive boundary, this being the logistic regressor which is a linear classifier. We will use the same regions to place the test set values to see how well the prediction performs as the model was trained on the training set (X_train, y_train). 

<img src = 'Screen5.png' width='700'>

The diagram below shows us how the logistic regression model performs for predicting the test set values.

<img src = 'Screen6.png' width='700'>

The confusion matrix below summarises the above diagram.

<img src = 'Screen4.png' width='350'>

From the confusion matrix, we conclude that our model has an accuracy of 89%.




