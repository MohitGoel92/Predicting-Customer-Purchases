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

**Pros of Logistic Regression**
- Probabilistic approach
- Gives information about statistical significance of features

**Cons of Logistic Regression**

- Logistic Regression assumptions

**Assumptions of the Logistic Regression**
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

## K Nearest Neighbours

The diagram below illustrates how the KNN algorithm works. We have two classes (A and B) and a new point depicted by the red star. In the first scenario, if we choose the number of k-neighbours to be 3 we observe 2 points from class B and only one from class A, therefore the algorithm will classify the new point as class B. In the second scenario however, if we choose the number of k-neighbours to be 6 we observe 4 points belonging to class A and only 2 points belonging to class B, therefore classifying the new point as class A.

<img src = 'Screenknn.png' width='500'>

**Steps to perform the KNN algorithm**

- Step 1: Choose the number k of neighbours (usually k = 5)
- Step 2: Take the k nearest neighbours of the new data point, according to the euclidean distance.
- Step 3: Among these k neighbours, count the number of data points in each category.
- Step 4: Assign the new data point to the category where you counted the most neighbours. 

**Pros of K Nearest Neighbours**

- Simple to understand
- Fast and efficient

**Cons of K Nearest Neighbours**

- Need to choose the number of neighbours (n_neighbors = k)

The diagram below shows the KNN algorithm being trainined on the training set.

<img src = 'Screen7.png' width='700'>

The diagram below shows us how the KNN model performs for predicting the test set values. 

<img src = 'Screen8.png' width='700'>

The confusion matrix below summarises the above diagram.

<img src = 'Screen9.png' width='350'>

From the confusion matrix, we conclude that our model has an accuracy of 93%.

## Support Vector Machine - Linear Kernel

The SVM algorithm looks at the very extreme cases, or in other words, the points that are near the boundary of the group, class or studied classification. This is used to construct the analysis which makes the SVM unique in comparison to other machine learning algorithms. The diagram below illustrates how the SVM will linearly seperate two classes.

<img src = 'Screen13.png' width='700'>

The predictive boundary (or straight line) seperating the two classes is drawn equidistant between the two extreme vectors (points); we refer to them as "Support Vectors". We call this line the "Maximum Margin Hyperplane". The margin must be maximised in order for the resulting line to be the SVC. They're called vectors as, in the higher dimensions they will be referred to as vectors and not points. The two parallel margin hyperplanes are referred to as the "Positive Hyperplane" and "Negative Hyperplane", the order does not matter.  

**Pros of SVM - Linear Kernel**
- Performant
- Not biased by outliers
- Not sensitive to overfitting

**Cons of SVM - Linear Kernel**
- Not appropriate for non-linear problems
- Not the best choice for a large number of features

The diagram below shows the SVM being trainined on the training set.

<img src = 'Screen10.png' width='700'>

The diagram below shows how the SVM model performs for predicting the test set values.

<img src = 'Screen11.png' width='700'>

The confusion matrix below summarises the above diagram.

<img src = 'Screen12.png' width='350'>

From the confusion matrix, we conclude that our model has an accuracy of 90%.

## Support Vector Machine - Gaussian RBF (Radial Basis Function) Kernel

The diagrams below show a dataset which contains two classes, the first dataset being linearly seperable but the second not being linearly seperable. For the first case we may use either use Logistic Regression or SVM with a linear kernel, but for the second case this will fail. Therefore we may use models such as the SVM with the Gaussian RBF Kernel.

<img src = 'Screen17.png' width='850'>

**Mapping to a higher-dimensional space**

Lets observe the dataset below. From inspection, we recognise that the data is not linearly seperable.

<img src = 'Screen18.png' width='600'>

Let's shift the data points to left by 5 units (X-5), and then square the function. As the data will take a parabolic shape, we now observe the data being linearly seperable with a straight line as shown below. Thus, the data being linearly seperable in a higher-dimensional space. 

<img src = 'Screen21.png' width='700'>

For a 2D plane, the data is transformed to a 3D plane and a "Maximum Margin Hyperplane" is used to linearly seperate the data. This is illustrated by the diagrams below. 

<img src = 'Screen19_1.png' width='700'>

The 3D space is then projected back to the original 2D plane, having our non-linear seperator in place. This is illustrated by the diagrams below.

<img src = 'Screen20.png' width='700'>

However, mapping to a higher dimensional space can be highly compute-intensive (demanding a lot a computation); we therefore use the kernel trick.

**The Kernel Trick**



**Pros of SVM - Gaussian RBF Kernel**

- High performance on non-linear problems
- Not biased by outliers
- Not sensitive to overfitting

**Cons of SVM - Gaussian RBF Kernel**

- Not the best choice for a large number of features
- More complex than other models

<img src = 'Screen14.png' width='700'>

<img src = 'Screen15.png' width='700'>

<img src = 'Screen16.png' width='350'>
