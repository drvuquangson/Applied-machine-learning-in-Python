
# coding: utf-8

# ---
# 
# _You are currently looking at **version 1.5** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the [Jupyter Notebook FAQ](https://www.coursera.org/learn/python-machine-learning/resources/bANLa) course resource._
# 
# ---

# # Assignment 2
# 
# In this assignment you'll explore the relationship between model complexity and generalization performance, by adjusting key parameters of various supervised learning models. Part 1 of this assignment will look at regression and Part 2 will look at classification.
# 
# ## Part 1 - Regression

# First, run the following block to set up the variables needed for later sections.

# In[1]:

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
#print(x)
#np.linspace: Return evenly spaced numbers over a specified interval
#np.random.randn: Return a sample (or samples) from the “standard normal” distribution
y = np.sin(x)+x/6 + np.random.randn(n)/10
#np.sin: Trigonometric sine
#print(x)
#print(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

# You can use this function to help you visualize the dataset by plotting a scatterplot of the data points in the training and test sets.
def part1_scatter():
    import matplotlib.pyplot as plt
    #get_ipython().magic('matplotlib notebook')
    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    plt.show()
    
    
# NOTE: Uncomment the function below to visualize the data, but be sure 
# to **re-comment it before submitting this assignment to the autograder**.   
print(part1_scatter())


# ### Summary: 
# We can't use normal linear Regression in this example

# ### Question 1
# 
# Write a function that fits a polynomial LinearRegression model on the *training data* `X_train` for degrees 1, 3, 6, and 9. (Use PolynomialFeatures in sklearn.preprocessing to create the polynomial features and then fit a linear regression model) For each model, find 100 predicted values over the interval x = 0 to 10 (e.g. `np.linspace(0,10,100)`) and store this in a numpy array. The first row of this array should correspond to the output from the model trained on degree 1, the second row degree 3, the third row degree 6, and the fourth row degree 9.
# 
# <img src="readonly/polynomialreg1.png" style="width: 1000px;"/>
# 
# The figure above shows the fitted models plotted on top of the original data (using `plot_one()`).
# 
# <br>
# *This function should return a numpy array with shape `(4, 100)`*

# ### Analysis
# 
# In this problem we have scattered classes -> We can't directly use a normal linear regression to classify classes. We have to transform it to the polynomial shape first. We have 4 degree(1,3,6,9) so our polynomial shape has 4 models. We then create X_test values for 4 of these models using np.linspace. Count only to 4 because our for loop has the maximum value of 4

# In[2]:

#Return 4 lists and inside each list has 100 items aray([[],[],[],[]])
#You need to use reshape to mach X_train(X_test) with y_train
def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures    
 
    results = np.zeros([4,100])#np.zeros: Return an array with 0 value of given shape and type. To call results from tmp_ans
        #([# of list inside the array, the # of values inside a list])
    #print(results)    
    X_test = np.linspace(0, 10, 100).reshape(-1,1) #Create 100 values for testing the classifier
    #print(X_predict)
    #.reshape(# of row(s), #values in each list). -1 means 1 row for each value

    count = 0
    
    #Step 1: Call Poly function
    for i in [1, 3, 6, 9]:
        poly = PolynomialFeatures(degree=i)
        
    #Step 2: transforms normal data to poly data than fit
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
        X_test_poly = poly.fit_transform(X_test) 
        linreg = LinearRegression().fit(X_train_poly, y_train)# Create a classifier
                                          
    #Step 3: Predict values of X-test                                  
        tmp_ans = linreg.predict(X_test_poly).reshape(1,-1) #Test with a test model
        results[count, :] = tmp_ans #[count = access to the list #0, : = takes all values]
        count = count + 1 #Count only to 4
    return results
print(answer_one())


# In[3]:

# feel free to use the function plot_one() to replicate the figure 
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    #get_ipython().magic('matplotlib notebook')
    #Step 1: Create fig
    plt.figure(figsize=(10,5)) 
    
    #Step 2: Create test and trining data points
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    
    #Step 3: Many points will form a line
    for i,degree in enumerate([0,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    plt.show()

print(plot_one(answer_one()))


# ### Summary:
# 1) Training and test data are X_train, y_train, X_test, y_test respectively
# 2) The line is the predicted results from X_test with different degree of freedom respectively
# 3) We look at the degree line, then check which line fits to both training data and test data the most

# ### Question 2
# 
# Write a function that fits a polynomial LinearRegression model on the training data `X_train` for degrees 0 through 9. For each model compute the $R^2$ (coefficient of determination) regression score on the training data as well as the the test data, and return both of these arrays in a tuple.
# 
# *This function should return one tuple of numpy arrays `(r2_train, r2_test)`. Both arrays should have shape `(10,)`*

# In[4]:

def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures
   
    results_train = np.zeros([10, 1]) #Create 10 rows and 1 column consists of 0 (1) 'results_train' varialbe needs to be referred 
        #before being used but in oder to be used inside a loop -> Use np.zeros (2) Else 'None' or '0' = error (3)
    #print(results_train)
    #Refer to ' results_train' before referring it 
    results_test = np.zeros([10, 1])
                                                  
    for i in range(0,10): 
        #Step 1: Create a loop running from 0 through 9
        poly = PolynomialFeatures(degree=i)     
        
        #Step 2: Transform original data to polynomial data
        X_train_poly = poly.fit_transform(X_train.reshape(-1, 1)) 
        X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))
        
        
        #Step 3: Compute R square
        linreg = LinearRegression().fit(X_train_poly, y_train) #Fit this model to the data set
        score_train = linreg.score(X_train_poly, y_train)  
        score_test = linreg.score(X_test_poly, y_test)    
        
        results_train[i] = score_train   
        #print(results_train[i])
        results_test[i] = score_test
        results_train = results_train.flatten() #Collapse everything into one dimension
        results_test = results_test.flatten()
       
        #print(results_train.shape)
        #print(results_train.shape)
    return (results_train, results_test)

#print(answer_two())


# ### Question 3
# 
# Based on the $R^2$ scores from question 2 (degree levels 0 through 9), what degree level corresponds to a model that is underfitting? What degree level corresponds to a model that is overfitting? What choice of degree level would provide a model with good generalization performance on this dataset? 
# 
# Hint: Try plotting the $R^2$ scores from question 2 to visualize the relationship between degree level and $R^2$. Remember to comment out the import matplotlib line before submission.
# 
# *This function should return one tuple with the degree values in this order: `(Underfitting, Overfitting, Good_Generalization)`. There might be multiple correct solutions, however, you only need to return one possible solution, for example, (1,2,3).* 

# In[5]:

def answer_three(degree_predictions):
    import matplotlib.pyplot as plt
    #get_ipython().magic('matplotlib notebook')
    results_train, results_test = answer_two() 
    
    #Step 1: Create a figure
    plt.figure(figsize=(10,5))
    
    #Step 2: Plot R-square score results from answer_two as points (colors are automatically assigned)
    plt.plot(range(0,10,1), results_train, 'o', label='training score', markersize=10)
    plt.plot(range(0,10,1), results_test, 'o', label='test score', markersize=10)
    
    #Step 3: Plot multiple points from '1,3,6,9' degree of freedom -> form a line instead of points using linewidth or lw
    for i,degree in enumerate ([1, 3, 6, 9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
        #Create a line based on 100 values * 4 columns from 'answer_one()' 
    
    plt.ylim(-1, 2.5)
    plt.xlabel('Input Value')
    plt.ylabel('Output Value')
    plt.legend(loc=9)
    plt.show() #show the figure
    return (9,1,3 )
#print(answer_three(answer_one())



# ### Summary:
# 
# Which degree line lands on both training scores and test scores which close to each other the most. Because there are no regularised -> loss function: the difference between the training scores and the test scores

# ### Question 4
# 
# Training models on high degree polynomial features can result in overly complex models that overfit, so we often use regularized versions of the model to constrain model complexity, as we saw with Ridge and Lasso linear regression.
# 
# For this question, train two models: a non-regularized LinearRegression model (default parameters) and a regularized Lasso Regression model (with parameters `alpha=0.01`, `max_iter=10000`) both on polynomial features of degree 12. Return the $R^2$ score for both the LinearRegression and Lasso model's test sets.
# 
# *This function should return one tuple `(LinearRegression_R2_test_score, Lasso_R2_test_score)`*

# In[6]:

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics.regression import r2_score
    
    #Step 1: Call Polynomyal first
    poly = PolynomialFeatures(degree=12)
    #Step 1.1: Transform original data to polynomial data
    X_train_poly = poly.fit_transform(X_train.reshape(-1, 1)) #Note 1
    X_test_poly = poly.fit_transform(X_test.reshape(-1, 1))
    
    #Step 2: Fit the data to the normal linear model + Calculate test score for normal linear model 
    linreg = LinearRegression().fit(X_train_poly, y_train)
    
    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)
    
    #Step 3: Calculate test score for Lasso linear model 
    linlasso = Lasso(alpha=0.01, max_iter=10000).fit(X_train_poly, y_train) 
    
    Lasso_R2_test_score = linlasso.score(X_test_poly, y_test)        
    return print('R-square score for normal LinearRegression: ', LinearRegression_R2_test_score,
            'R-square score for Lasso LinearRegression: ', Lasso_R2_test_score)
#print(answer_four())

#alpha = 0,01 did nothing to the coefficent of data


# ### Note 1:
# <font color='red'>X_train_poly = poly.fit_transform(X_train)</font>
# 
# X-train data has only 1 dimension -> Poly is a high dimensional model -> you can't pass 1 dimension data into it
# 
# Error will be raised when you use code like the above
# Passing 1d arrays as data is deprecated in 0.17 and will raise ValueError in 0.19. 
# Reshape your data either using X.reshape(-1, 1) if your data has a single feature or X.reshape(1, -1) 
# if it contains a single sample

# ## Part 2 - Classification
# 
# Here's an application of machine learning that could save your life! For this section of the assignment we will be working with the [UCI Mushroom Data Set](http://archive.ics.uci.edu/ml/datasets/Mushroom?ref=datanews.io) stored in `readonly/mushrooms.csv`. The data will be used to train a model to predict whether or not a mushroom is poisonous. The following attributes are provided:
# 
# *Attribute Information:*
# 
# (shape, color, ordor of mushrooms)
# 1. cap-shape: bell=b, conical=c, convex=x, flat=f, knobbed=k, sunken=s 
# 2. cap-surface: fibrous=f, grooves=g, scaly=y, smooth=s 
# 3. cap-color: brown=n, buff=b, cinnamon=c, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y 
# 4. bruises?: bruises=t, no=f 
# 5. odor: almond=a, anise=l, creosote=c, fishy=y, foul=f, musty=m, none=n, pungent=p, spicy=s 
#     (almond: hanh nhan, anise: hoi, creosote: oil distilled from coal tar )
#     
# (grill = the part belows the veil)
# 6. gill-attachment: attached=a, descending=d, free=f, notched=n 
# 7. gill-spacing: close=c, crowded=w, distant=d 
# 8. gill-size: broad=b, narrow=n 
# 9. gill-color: black=k, brown=n, buff=b, chocolate=h, gray=g, green=r, orange=o, pink=p, purple=u, red=e, white=w, yellow=y 
# 
# (stalk = the main steam of a plant)
# 10. stalk-shape: enlarging=e, tapering=t 
# 11. stalk-root: bulbous=b, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r, missing=? 
# 12. stalk-surface-above-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 13. stalk-surface-below-ring: fibrous=f, scaly=y, silky=k, smooth=s 
# 14. stalk-color-above-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 15. stalk-color-below-ring: brown=n, buff=b, cinnamon=c, gray=g, orange=o, pink=p, red=e, white=w, yellow=y 
# 
# (veil = mushroom hat)
# 16. veil-type: partial=p, universal=u 
# 17. veil-color: brown=n, orange=o, white=w, yellow=y 
# 
# (the part cover the upper stem)
# 18. ring-number: none=n, one=o, two=t 
# 19. ring-type: cobwebby=c, evanescent=e, flaring=f, large=l, none=n, pendant=p, sheathing=s, zone=z 
# 
# (demographics &  enviroments)
# 20. spore-print-color: black=k, brown=n, buff=b, chocolate=h, green=r, orange=o, purple=u, white=w, yellow=y 
# 21. population: abundant=a, clustered=c, numerous=n, scattered=s, several=v, solitary=y 
# 22. habitat: grasses=g, leaves=l, meadows=m, paths=p, urban=u, waste=w, woods=d
# 
# <br>
# 
# The data in the mushrooms dataset is currently encoded with strings. These values will need to be encoded to numeric to work with sklearn. We'll use pd.get_dummies to convert the categorical variables into indicator variables (dummy variables).  

# In[13]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv('mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)

X_mush = mush_df2.iloc[:,2:] #features
y_mush = mush_df2.iloc[:,1] #label

# use the variables X_train2, y_train2 for Question 5
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

# For performance reasons in Questions 6 and 7, we will create a smaller version of the
# entire mushroom dataset for use in those questions.  For simplicity we'll just re-use
# the 25% test split created above as the representative subset.
#
# Use the variables X_subset, y_subset for Questions 6 and 7.
X_subset = X_test2
y_subset = y_test2


# ### Question 5
# 
# Using `X_train2` and `y_train2` from the preceeding cell, train a DecisionTreeClassifier with default parameters and random_state=0. What are the 5 most important features found by the decision tree?
# 
# As a reminder, the feature names are available in the `X_train2.columns` property, and the order of the features in `X_train2.columns` matches the order of the feature importance values in the classifier's `feature_importances_` property. 
# 
# *This function should return a list of length 5 containing the feature names in descending order of importance.*
# 
# *Note: remember that you also need to set random_state in the DecisionTreeClassifier.*

# In[14]:

def answer_five():
    from sklearn.tree import DecisionTreeClassifier
    
    #Step 1: Call decision tree
    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    
    #Step 2: Calculate feature importance + set column labels as index
    Series = pd.Series(clf.feature_importances_, index = X_train2.columns)
    
    #Step 3: Sort in descending order -> transform index to list
    Sorted_data = Series.sort_values(ascending = False, inplace = False)
    #print(Sorted_data)
    
    Answer = Sorted_data[:5].index.tolist()
    #.index: take index only
    #.tolist(): transfer a numpy array to list
    return Answer
#print(answer_five())

#feature importances: are features of any column that have the most weight and the weight of all features
#when sum up must equal 1


# ### Question 6
# 
# For this question, we're going to use the `validation_curve` function in `sklearn.model_selection` to determine training and test scores for a Support Vector Classifier (`SVC`) with varying parameter values.  Recall that the validation_curve function, in addition to taking **<font color='red'>an initialized unfitted classifier object</font>**, takes a **<font color='green'>dataset</font>** as input and does its own internal train-test splits to compute results.
# 
# **Because creating a validation curve requires fitting multiple models, for performance reasons this question will use just a subset of the original mushroom dataset: please use the variables X_subset and y_subset as input to the validation curve function (instead of X_mush and y_mush) to reduce computation time.**
# 
# **<font color='red'>The initialized unfitted classifier object</font>** we'll be using is a Support Vector Classifier with **<font color='blue'>radial basis kernel</font>**. 
# 
# So your first step is to create an `SVC` object with default parameters (i.e. `kernel='rbf', C=1`) and `random_state=0`. Recall that the **kernel width** of the RBF kernel is controlled using the `gamma` parameter.  
# 
# With this classifier, and the dataset in X_subset, y_subset, explore the effect of `gamma` on classifier accuracy by using the `validation_curve` function to find the training and test scores for 6 values of `gamma` from `0.0001` to `10` (i.e. `np.logspace(-4,1,6)`). Recall that you can specify what **scoring metric** you want validation_curve to use by setting the **"scoring"** parameter.  In this case, we want to use **<font color='red'>"accuracy"</font>** as the scoring metric.
# 
# For each level of `gamma`, `validation_curve` will fit 3 models on different subsets of the data, returning two 6x3 (6 levels of gamma x 3 fits per level) arrays of the scores for the training and test sets.
# 
# Find the mean score across the three models for each level of `gamma` for both arrays, creating two arrays of length 6, and return a tuple with the two arrays.
# 
# e.g.
# 
# if one of your array of scores is
# 
#     array([[ 0.5,  0.4,  0.6],
#            [ 0.7,  0.8,  0.7],
#            [ 0.9,  0.8,  0.8],
#            [ 0.8,  0.7,  0.8],
#            [ 0.7,  0.6,  0.6],
#            [ 0.4,  0.6,  0.5]])
#        
# it should then become
# 
#     array([ 0.5,  0.73333333,  0.83333333,  0.76666667,  0.63333333, 0.5])
# 
# *This function should return one tuple of numpy arrays `(training_scores, test_scores)` where each array in the tuple has shape `(6,)`.*

# In[15]:

def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    
    
    #Step 1: Create an SVC object
    clf = SVC(kernel = 'rbf', C = 1).fit(X_train2, y_train2) #clf = classifier linear function
    
    #Step 2: Find the training and test score:
    log_scale = np.logspace(-4,1,6)
    training_scores, test_scores = validation_curve(clf, X_subset, y_subset, param_name='gamma', 
                                                    param_range = log_scale, cv = 3, scoring = 'accuracy') 
    
    
    #Step 3: Compute the mean of 3 models for each level of gamma for both arrays:
      #the default of np.mean calculates mean of a flattened array -> Change axis = None to axis = 1 to calculate mean of 6 rows 
    training_scores_mean = np.mean(training_scores, axis = 1) 
    test_scores_mean = np.mean(test_scores, axis = 1)
    return (training_scores_mean, test_scores_mean)
#print(answer_six())


# **<font color='red'> Summary Q6:</font>**
# 
# Higher gamma means
# the higher accuracy for training data, and also higher accuracy for test data **but** only up to a certain point

# ### Question 7
# 
# Based on the scores from question 6, what gamma value corresponds to a model that is underfitting (and has the worst test set accuracy)? What gamma value corresponds to a model that is overfitting (and has the worst test set accuracy)? What choice of gamma would be the best choice for a model with good generalization performance on this dataset (high accuracy on both training and test set)? 
# 
# Hint: Try plotting the scores from question 6 to visualize the relationship between gamma and accuracy. Remember to comment out the import matplotlib line before submission.
# 
# *This function should return one tuple with the degree values in this order: `(Underfitting, Overfitting, Good_Generalization)` Please note there is only one correct solution.*
# 
# smaller gamma = **<font color='red'>overfitting.</font>**
# 
# larger gamma = **<font color='blue'>underfitting.</font>**

# In[18]:

def answer_seven():
    import matplotlib.pyplot as plt
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    #get_ipython().magic('matplotlib notebook')
    
#Step 0: Recreate training&test scores to calculate standard devidation. Because you can't call variables inside other functions
    clf = SVC(kernel = 'rbf', C = 1).fit(X_train2, y_train2)
    log_scale = np.logspace(-4,1,6)
    train_scores, test_scores = validation_curve(clf, X_subset, y_subset, param_name='gamma', 
                                                    param_range = log_scale, cv = 3, scoring = 'accuracy') 
    
    #Step 1: Prepare gamma and std
    (train_scores_mean, test_scores_mean) = answer_six()
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_std = np.std(test_scores, axis =1)
   
    #Step 3: Fine-tuned the figure
    plt.figure  
    plt.title('Validation Curve with SVM')
    plt.xlabel('$\gamma$ (gamma)')
    plt.ylabel('Accuracy score')
    plt.ylim(0.0, 1.1)
     
    
    #Step 2: Plot the log scaling on the figure
    log_scale = np.logspace(-4,1,6)
    
    plt.semilogx(log_scale, train_scores_mean, label = 'Training score', color = 'darkorange', lw = 2)
    
    plt.fill_between(log_scale,train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
                 alpha = 0.2, color = 'darkorange', lw = 2) #Show variation around the mean
    
    plt.semilogx(log_scale, test_scores_mean, label = 'Test sore', color = 'darkblue', lw = 2)
    
    plt.fill_between(log_scale, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, 
                     alpha = 0.2,color = 'darkorange', lw =2)
    
    #fine-tuned the figure
    plt.legend(loc='best')
    plt.show()
    
    Underfitting = 'Underfitting when gamma = 0.0001'
    Overfitting = 'Overfitting when gamma = 10'
    Fitting = 'Good generalisation when gamma = 0.1'
 
    return (Underfitting, Overfitting, Fitting)
#print(answer_seven())


# **<font color='red'> Summary Q7:</font>**
# 
# 1) The standard deviation of this model is really low even when gamma is low -> The subpoints of training points focus mostly
# around the center  
# 
# 2) The accuracy score of test data will reach its highest when it hits a certain level of gamma, but after that it's gonna start to decline

# 
