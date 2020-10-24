---
title: "Dimensionality Reduction Code"
author_profile: false
toc: false

toc_sticky: true
tags:
  - mathjax
date: 2016-11-14 08:15:00 +0530
categories: courses
header:
  teaser: "/assets/images/ai1.jpg"
excerpt: "Dimensionality Reduction Code"
description: "Dimensionality Reduction Code"
og_image: "/assets/images/proj.jpg"
use_math: true
mathjax: true
---

Python code block:

```python
    import numpy as np

    def test_function(x, y):
      z = np.sum(x,y)
      return z
```

```python
  import numpy as np

  def test_function(x, y):
    z = np.sum(x,y)
    return z
```

R code block:

```r
library(tidyverse)
df <- read_csv("some_file.csv")
head(df)
```

Here's some inline code `x+y`.

Here's an image:
<img src="{{ site.url }}{{ site.baseurl }}/assets/images/perceptron/linsep.jpg" alt="linearly separable data">

Here's another image using Kramdown:
![alt]({{ site.url }}{{ site.baseurl }}/assets/images/perceptron/linsep.jpg)

Here's some math:

$$z=x+y$$

You can also put it inline $$z=x+y$$

# **DSDA_2020 (Data Science and Data Analytics: Opportunities and Challenges)**

# **High Dimensionality Dataset Reduction Methodologies in Applied Machine Learning**

Farhan Hai Khan<sup>a</sup>,Tannistha Pal<sup>b</sup>

a. Department of Electrical Engineering, Institute of Engineering & Management, Kolkata, India, khanfarhanpro@gmail.com<br>
b. Department of Electronics and Communication Engineering, Institute of Engineering & Management, Kolkata, India, paltannistha@gmail.com

## **Abstract**

A common problem faced while handling multi-featured datasets is the high amount of dimensionality that they often consist of, leading to barriers in generalized hands-on Machine Learning. These datasets also give a drastic impact on the performance of Machine Learning algorithms, being memory inefficient and frequently leading to model overfitting. It often becomes difficult to visualize or gain insightful knowledge on the data features such as presence of outliers.

This chapter will help data analysts reduce data dimensionality using various methodologies such as:<br>

1. Feature Selection using Covariance Matrix
2. Principal Component Analysis (PCA)
3. t-distributed Stochastic Neighbour Embedding (t-SNE)

Under applications of Dimensionality Reduction Algorithms with Visualizations, firstly, we introduce the Boston Housing Dataset and use the Correlation Matrix to apply Feature Selection on the strongly positive correlated data and perform Simple Linear Regression over the new features.Then we use UCI Breast Cancer Dataset to perform PCA Analysis with Support Vector Machine Classification (SVM). Lastly, we apply t-SNE to MNIST Handwritten Digits Dataset and use k-Nearest Neighbours (kNNs) clustering for classification.

Finally, we explore the benefits of using Dimensionality Reduction Methods and provide a comprehensive overview of reduction in storage space, efficient models,feature selection guidelines ,redundant data removal and outlier analysis.

## **Keywords** : Dimensionality Reduction, Feature Selection, Covariance Matrix, PCA , t-SNE

## **Table of Contents**

1. Problems faced with Multi-Dimensional Datasets
   1. Data Intuition
   2. Data Visualization Constraints
   3. Outlier Detection
2. Dimensionality Reduction Algorithms with Visualizations
   1. Feature Selection using Covariance Matrix
   2. Principal Component Analysis (PCA)
   3. t-distributed Stochastic Neighbour Embedding (t-SNE)
3. Benefits of Dimensionality Reduction
   1. Storage Space Reduction
   2. Computation Time Optimization
   3. Redundant Feature Removal
   4. Incorrect Data Removal

## Problems faced with Multi-Dimensional Datasets

TODO

## Initial Seeds

np.seed(num)

## Dimensionality Reduction Algorithms with Visualizations

### Feature Selection using Covariance Matrix

**Objectives :** Introduce Boston Housing Dataset and use the Correlation Matrix to apply Feature Selection on the strongly positive correlated data and perform Simple Linear Regression over the new features.

https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155

https://www.geeksforgeeks.org/ml-boston-housing-kaggle-challenge-with-linear-regression/

https://towardsdatascience.com/polynomial-regression-bbe8b9d97491

Ridge & Lasso

```python
  # We will need 3 datasets for this chapter, each of which have been documented on our github repository.
  # So let us create a local copy (clone) of that repo here :)
  !git clone www.github.com/khanfarhan10/DIMENSIONALITY_REDUCTION.git
```

```python
  import numpy as np

  def test_function(x, y):
    z = np.sum(x,y)
    return z
```

```python

  # Firstly we will import all the necessary libraries that we will be requiring for Dataset Reductions.

  import numpy as np              # Mathematical Functions , Linear Algebra, Matrix Operations
  import pandas as pd             # Data Manipulations,  Data Analysis/Storing/Preparation


  import matplotlib.pyplot as plt # Simple Data Visualization , Basic Plotting Utilities
  import seaborn as sns           # Advanced Data Visualization, High Level Figures Interfacing

  %matplotlib inline
  # Interactive Jupyter Notebook Plotting
  #%matplotlib inline             #This can be used as an alternative but the plots obtained will be non interactive in nature.
```

```python

  # Importing Data
  from sklearn.datasets import load_boston
  boston_dataset = load_boston()
  df = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
  df['MEDV'] = boston_dataset.target
  df.head(10)
  #NON EXISTENT, df.to_excel("Boston_Data.xlsx",index=False)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.00632</td>
      <td>18.0</td>
      <td>2.31</td>
      <td>0.0</td>
      <td>0.538</td>
      <td>6.575</td>
      <td>65.2</td>
      <td>4.0900</td>
      <td>1.0</td>
      <td>296.0</td>
      <td>15.3</td>
      <td>396.90</td>
      <td>4.98</td>
      <td>24.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.02731</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>6.421</td>
      <td>78.9</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>396.90</td>
      <td>9.14</td>
      <td>21.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.02729</td>
      <td>0.0</td>
      <td>7.07</td>
      <td>0.0</td>
      <td>0.469</td>
      <td>7.185</td>
      <td>61.1</td>
      <td>4.9671</td>
      <td>2.0</td>
      <td>242.0</td>
      <td>17.8</td>
      <td>392.83</td>
      <td>4.03</td>
      <td>34.7</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.03237</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.998</td>
      <td>45.8</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.63</td>
      <td>2.94</td>
      <td>33.4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.06905</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>7.147</td>
      <td>54.2</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>396.90</td>
      <td>5.33</td>
      <td>36.2</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.02985</td>
      <td>0.0</td>
      <td>2.18</td>
      <td>0.0</td>
      <td>0.458</td>
      <td>6.430</td>
      <td>58.7</td>
      <td>6.0622</td>
      <td>3.0</td>
      <td>222.0</td>
      <td>18.7</td>
      <td>394.12</td>
      <td>5.21</td>
      <td>28.7</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.08829</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.012</td>
      <td>66.6</td>
      <td>5.5605</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>395.60</td>
      <td>12.43</td>
      <td>22.9</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.14455</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.172</td>
      <td>96.1</td>
      <td>5.9505</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>396.90</td>
      <td>19.15</td>
      <td>27.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.21124</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>5.631</td>
      <td>100.0</td>
      <td>6.0821</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>386.63</td>
      <td>29.93</td>
      <td>16.5</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.17004</td>
      <td>12.5</td>
      <td>7.87</td>
      <td>0.0</td>
      <td>0.524</td>
      <td>6.004</td>
      <td>85.9</td>
      <td>6.5921</td>
      <td>5.0</td>
      <td>311.0</td>
      <td>15.2</td>
      <td>386.71</td>
      <td>17.10</td>
      <td>18.9</td>
    </tr>
  </tbody>
</table>
</div>

```python
    boston_dataset.keys()
```

    dict_keys(['data', 'target', 'feature_names', 'DESCR', 'filename'])

```python
    boston_dataset.feature_names
```

    array(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
           'TAX', 'PTRATIO', 'B', 'LSTAT'], dtype='<U7')

```python
    boston_dataset.DESCR
```

    ".. _boston_dataset:\n\nBoston house prices dataset\n---------------------------\n\n**Data Set Characteristics:**  \n\n    :Number of Instances: 506 \n\n    :Number of Attributes: 13 numeric/categorical predictive. Median Value (attribute 14) is usually the target.\n\n    :Attribute Information (in order):\n        - CRIM     per capita crime rate by town\n        - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.\n        - INDUS    proportion of non-retail business acres per town\n        - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)\n        - NOX      nitric oxides concentration (parts per 10 million)\n        - RM       average number of rooms per dwelling\n        - AGE      proportion of owner-occupied units built prior to 1940\n        - DIS      weighted distances to five Boston employment centres\n        - RAD      index of accessibility to radial highways\n        - TAX      full-value property-tax rate per $10,000\n        - PTRATIO  pupil-teacher ratio by town\n        - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town\n        - LSTAT    % lower status of the population\n        - MEDV     Median value of owner-occupied homes in $1000's\n\n    :Missing Attribute Values: None\n\n    :Creator: Harrison, D. and Rubinfeld, D.L.\n\nThis is a copy of UCI ML housing dataset.\nhttps://archive.ics.uci.edu/ml/machine-learning-databases/housing/\n\n\nThis dataset was taken from the StatLib library which is maintained at Carnegie Mellon University.\n\nThe Boston house-price data of Harrison, D. and Rubinfeld, D.L. 'Hedonic\nprices and the demand for clean air', J. Environ. Economics & Management,\nvol.5, 81-102, 1978.   Used in Belsley, Kuh & Welsch, 'Regression diagnostics\n...', Wiley, 1980.   N.B. Various transformations are used in the table on\npages 244-261 of the latter.\n\nThe Boston house-price data has been used in many machine learning papers that address regression\nproblems.   \n     \n.. topic:: References\n\n   - Belsley, Kuh & Welsch, 'Regression diagnostics: Identifying Influential Data and Sources of Collinearity', Wiley, 1980. 244-261.\n   - Quinlan,R. (1993). Combining Instance-Based and Model-Based Learning. In Proceedings on the Tenth International Conference of Machine Learning, 236-243, University of Massachusetts, Amherst. Morgan Kaufmann.\n"

```python
    boston_dataset.filename
```

    '/usr/local/lib/python3.6/dist-packages/sklearn/datasets/data/boston_house_prices.csv'

```python
  #ALTERNATIVELY,
  df=pd.read_excel("DIMENSIONALITY_REDUCTION/data/Boston_Data.xlsx")
```

```python
  df.isnull().sum()
```

    CRIM       0
    ZN         0
    INDUS      0
    CHAS       0
    NOX        0
    RM         0
    AGE        0
    DIS        0
    RAD        0
    TAX        0
    PTRATIO    0
    B          0
    LSTAT      0
    MEDV       0
    dtype: int64

```python
    df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 506 entries, 0 to 505
    Data columns (total 14 columns):
     #   Column   Non-Null Count  Dtype
    ---  ------   --------------  -----
     0   CRIM     506 non-null    float64
     1   ZN       506 non-null    float64
     2   INDUS    506 non-null    float64
     3   CHAS     506 non-null    float64
     4   NOX      506 non-null    float64
     5   RM       506 non-null    float64
     6   AGE      506 non-null    float64
     7   DIS      506 non-null    float64
     8   RAD      506 non-null    float64
     9   TAX      506 non-null    float64
     10  PTRATIO  506 non-null    float64
     11  B        506 non-null    float64
     12  LSTAT    506 non-null    float64
     13  MEDV     506 non-null    float64
    dtypes: float64(14)
    memory usage: 55.5 KB

```python
    df.describe()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677083</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>

```python
  plt.style.use("dark_background")
```

```python
  #gibberish
  COLORS=["lime"]*13
  COLORS+=["red"]
```

```python
df.hist(bins=30,figsize=(20,10),grid=False,color="crimson"); #cyan , magenta , crimson,  #column="MEDV",,color='lime'
plt.title("Boston Dataset : Frequency Distribution of Numerical Data")
plt.savefig("Boston Dataset Frequency Distribution of Numerical Data.png",dpi=600)
```

![png](Dimensionality_Reduction_Algorithms_Boston_Corr_Matrix_Regression_files/Dimensionality_Reduction_Algorithms_Boston_Corr_Matrix_Regression_18_0.png)

```python
correlation_matrix = df.corr().round(1)
# annot = True to print the values inside the square
plt.figure(figsize=(20,10))
sns.heatmap(data=correlation_matrix,cmap="inferno", annot=True)
plt.savefig("Correlation_Data.png",dpi=600)
```

![png](Dimensionality_Reduction_Algorithms_Boston_Corr_Matrix_Regression_files/Dimensionality_Reduction_Algorithms_Boston_Corr_Matrix_Regression_19_0.png)

```python
df.columns
```

    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'MEDV'],
          dtype='object')

```python
X = df.drop(columns=['MEDV'])
y = df['MEDV']

print('Shape of X : {} , Shape of y : {}'.format(X.shape,y.shape))
```

    Shape of X : (506, 13) , Shape of y : (506,)

```python
# Normalization of the Data
from sklearn.preprocessing import StandardScaler

scaler_X = StandardScaler()
X_norm= scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_norm= scaler_y.fit_transform(X)


```

```python
"abc {} xyz {}".format(1,2)
```

    'abc 1 xyz 2'

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=42)
print('Shape of X_train : {} , Shape of X_test : {}'.format(X_train.shape,X_test.shape))
print('Shape of y_train : {} , Shape of y_test : {}'.format(y_train.shape,X_test.shape))
```

    Shape of X_train : (354, 13) , Shape of X_test : (152, 13)
    Shape of y_train : (354,) , Shape of y_test : (152, 13)

```python
from sklearn.linear_model import LinearRegression

lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
```

    LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)

```
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

accLR=lin_model.score(X_test,y_test)
print(accLR)
```

    0.7112260057484874

```python
lin_model.coef_ #theta 1-13
```

    array([-1.33470103e-01,  3.58089136e-02,  4.95226452e-02,  3.11983512e+00,
           -1.54170609e+01,  4.05719923e+00, -1.08208352e-02, -1.38599824e+00,
            2.42727340e-01, -8.70223437e-03, -9.10685208e-01,  1.17941159e-02,
           -5.47113313e-01])

```python
lin_model.coef_.shape
```

    (13,)

```python
lin_model.intercept_ #theta 0
```

    31.631084035691632

```python
df.columns
```

    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'MEDV'],
          dtype='object')

```python
# Coefficient & Intercept :
def merge_list_to_dict(test_keys,test_values):
  """Uses dictionary comprehension to create a dict from 2 lists"""
  merged_dict = {test_keys[i]: test_values[i] for i in range(len(test_keys))}
  return merged_dict

def get_params(model_intercept,model_coefficients,data_cols):
  """Returns a dataframe of organised values for model parameters output with colums of the original dataframe"""
  res_theta=[model_intercept ]+ list(model_coefficients)
  res_y=['INTERCEPT']+list(data_cols)
  dict_res=merge_list_to_dict(res_y,res_theta)
  data=[dict_res]
  results=pd.DataFrame(data)
  return results

res=get_params(lin_model.intercept_,lin_model.coef_,X.columns)
res.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>INTERCEPT</th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>31.631084</td>
      <td>-0.13347</td>
      <td>0.035809</td>
      <td>0.049523</td>
      <td>3.119835</td>
      <td>-15.417061</td>
      <td>4.057199</td>
      <td>-0.010821</td>
      <td>-1.385998</td>
      <td>0.242727</td>
      <td>-0.008702</td>
      <td>-0.910685</td>
      <td>0.011794</td>
      <td>-0.547113</td>
    </tr>
  </tbody>
</table>
</div>

```python
help(get_params)
```

    Help on function get_params in module __main__:

    get_params(model_intercept, model_coefficients, data_cols)
        Returns a dataframe of organised values for model parameters output with colums of the original dataframe

```python
#demonstrating the purpose of the docstring
def tanni():
  """Meow meow meow"""
  return 1
help(tanni)
```

    Help on function tanni in module __main__:

    tanni()
        Meow meow meow

```


```

```

```

```

```

```

```

```

```

```python
preds=lin_model.predict(X_test)

preds.mean()
```

    21.33015845218848

```python
y_test.mean()
```

    21.407894736842103

```python
temp=[]
for a,b in zip(preds,y_test):
  temp.append(abs(a-b))
temp=np.array(temp)/len(temp)
1-temp.mean()
```

    0.9791926982140957

```

```

```

```

```python
df.columns
```

    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'MEDV'],
          dtype='object')

```

```

```

```

```

```

```python
round(2.7346374,2)
```

    2.73

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def getevaluation(model, X_subset,y_subset,subset_type="Train",round_scores=None):
  y_subset_predict = model.predict(X_subset)
  rmse = (np.sqrt(mean_squared_error(y_subset, y_subset_predict)))
  r2 = r2_score(y_subset, y_subset_predict)
  mae=mean_absolute_error(y_subset, y_subset_predict)
  if round_scores!=None:
    rmse=round(rmse,round_scores)
    r2=round(r2,round_scores)
    mae=round(mae,round_scores)

  print("Model Performance for {} subset :: RMSE: {} | R2 score: {} | MAE: {}".format(subset_type,rmse,r2,mae))

getevaluation(model=lin_model,X_subset=X_train,y_subset=y_train,subset_type="Train",round_scores=2)
getevaluation(model=lin_model,X_subset=X_test,y_subset=y_test,subset_type="Test ",round_scores=2)
```

    Model Performance for Train subset :: RMSE: 4.75 | R2 score: 0.74 | MAE: 3.36
    Model Performance for Test  subset :: RMSE: 4.64 | R2 score: 0.71 | MAE: 3.16

```

```

```

```

```

```

```

```

```

```

```

```

```

```

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# model evaluation for training set
y_train_predict = lin_model.predict(X_train)



rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

```

    The model performance for training set
    --------------------------------------
    RMSE is 4.748208239685937
    R2 score is 0.7434997532004697


    The model performance for testing set
    --------------------------------------
    RMSE is 4.638689926172867
    R2 score is 0.7112260057484874

## Metrics Report :

table

# ML Model with reduced dataset

The correlation coefficient ranges from -1 to 1. If the value is close to 1, it means that there is a strong positive correlation between the two variables. When it is close to -1, the variables have a strong negative correlation.

## Observations:

To fit a linear regression model, we select those features which have a high correlation with our target variable MEDV. By looking at the correlation matrix we can see that RM has a strong positive correlation with MEDV (0.7) where as LSTAT has a high negative correlation with MEDV(-0.74).
An important point in selecting features for a linear regression model is to check for multi-co-linearity. The features RAD, TAX have a correlation of 0.91. These feature pairs are strongly correlated to each other. We should not select both these features together for training the model. Check this for an explanation. Same goes for the features DIS and AGE which have a correlation of -0.75.
Based on the above observations we will use RM and LSTAT as our features. Using a scatter plot let’s see how these features vary with MEDV.

```python
df.columns
```

    Index(['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX',
           'PTRATIO', 'B', 'LSTAT', 'MEDV'],
          dtype='object')

```python
correlation_matrix = df.corr().round(2)

correlation_matrix
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>CRIM</th>
      <th>ZN</th>
      <th>INDUS</th>
      <th>CHAS</th>
      <th>NOX</th>
      <th>RM</th>
      <th>AGE</th>
      <th>DIS</th>
      <th>RAD</th>
      <th>TAX</th>
      <th>PTRATIO</th>
      <th>B</th>
      <th>LSTAT</th>
      <th>MEDV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CRIM</th>
      <td>1.00</td>
      <td>-0.20</td>
      <td>0.41</td>
      <td>-0.06</td>
      <td>0.42</td>
      <td>-0.22</td>
      <td>0.35</td>
      <td>-0.38</td>
      <td>0.63</td>
      <td>0.58</td>
      <td>0.29</td>
      <td>-0.39</td>
      <td>0.46</td>
      <td>-0.39</td>
    </tr>
    <tr>
      <th>ZN</th>
      <td>-0.20</td>
      <td>1.00</td>
      <td>-0.53</td>
      <td>-0.04</td>
      <td>-0.52</td>
      <td>0.31</td>
      <td>-0.57</td>
      <td>0.66</td>
      <td>-0.31</td>
      <td>-0.31</td>
      <td>-0.39</td>
      <td>0.18</td>
      <td>-0.41</td>
      <td>0.36</td>
    </tr>
    <tr>
      <th>INDUS</th>
      <td>0.41</td>
      <td>-0.53</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>0.76</td>
      <td>-0.39</td>
      <td>0.64</td>
      <td>-0.71</td>
      <td>0.60</td>
      <td>0.72</td>
      <td>0.38</td>
      <td>-0.36</td>
      <td>0.60</td>
      <td>-0.48</td>
    </tr>
    <tr>
      <th>CHAS</th>
      <td>-0.06</td>
      <td>-0.04</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>0.09</td>
      <td>-0.10</td>
      <td>-0.01</td>
      <td>-0.04</td>
      <td>-0.12</td>
      <td>0.05</td>
      <td>-0.05</td>
      <td>0.18</td>
    </tr>
    <tr>
      <th>NOX</th>
      <td>0.42</td>
      <td>-0.52</td>
      <td>0.76</td>
      <td>0.09</td>
      <td>1.00</td>
      <td>-0.30</td>
      <td>0.73</td>
      <td>-0.77</td>
      <td>0.61</td>
      <td>0.67</td>
      <td>0.19</td>
      <td>-0.38</td>
      <td>0.59</td>
      <td>-0.43</td>
    </tr>
    <tr>
      <th>RM</th>
      <td>-0.22</td>
      <td>0.31</td>
      <td>-0.39</td>
      <td>0.09</td>
      <td>-0.30</td>
      <td>1.00</td>
      <td>-0.24</td>
      <td>0.21</td>
      <td>-0.21</td>
      <td>-0.29</td>
      <td>-0.36</td>
      <td>0.13</td>
      <td>-0.61</td>
      <td>0.70</td>
    </tr>
    <tr>
      <th>AGE</th>
      <td>0.35</td>
      <td>-0.57</td>
      <td>0.64</td>
      <td>0.09</td>
      <td>0.73</td>
      <td>-0.24</td>
      <td>1.00</td>
      <td>-0.75</td>
      <td>0.46</td>
      <td>0.51</td>
      <td>0.26</td>
      <td>-0.27</td>
      <td>0.60</td>
      <td>-0.38</td>
    </tr>
    <tr>
      <th>DIS</th>
      <td>-0.38</td>
      <td>0.66</td>
      <td>-0.71</td>
      <td>-0.10</td>
      <td>-0.77</td>
      <td>0.21</td>
      <td>-0.75</td>
      <td>1.00</td>
      <td>-0.49</td>
      <td>-0.53</td>
      <td>-0.23</td>
      <td>0.29</td>
      <td>-0.50</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>RAD</th>
      <td>0.63</td>
      <td>-0.31</td>
      <td>0.60</td>
      <td>-0.01</td>
      <td>0.61</td>
      <td>-0.21</td>
      <td>0.46</td>
      <td>-0.49</td>
      <td>1.00</td>
      <td>0.91</td>
      <td>0.46</td>
      <td>-0.44</td>
      <td>0.49</td>
      <td>-0.38</td>
    </tr>
    <tr>
      <th>TAX</th>
      <td>0.58</td>
      <td>-0.31</td>
      <td>0.72</td>
      <td>-0.04</td>
      <td>0.67</td>
      <td>-0.29</td>
      <td>0.51</td>
      <td>-0.53</td>
      <td>0.91</td>
      <td>1.00</td>
      <td>0.46</td>
      <td>-0.44</td>
      <td>0.54</td>
      <td>-0.47</td>
    </tr>
    <tr>
      <th>PTRATIO</th>
      <td>0.29</td>
      <td>-0.39</td>
      <td>0.38</td>
      <td>-0.12</td>
      <td>0.19</td>
      <td>-0.36</td>
      <td>0.26</td>
      <td>-0.23</td>
      <td>0.46</td>
      <td>0.46</td>
      <td>1.00</td>
      <td>-0.18</td>
      <td>0.37</td>
      <td>-0.51</td>
    </tr>
    <tr>
      <th>B</th>
      <td>-0.39</td>
      <td>0.18</td>
      <td>-0.36</td>
      <td>0.05</td>
      <td>-0.38</td>
      <td>0.13</td>
      <td>-0.27</td>
      <td>0.29</td>
      <td>-0.44</td>
      <td>-0.44</td>
      <td>-0.18</td>
      <td>1.00</td>
      <td>-0.37</td>
      <td>0.33</td>
    </tr>
    <tr>
      <th>LSTAT</th>
      <td>0.46</td>
      <td>-0.41</td>
      <td>0.60</td>
      <td>-0.05</td>
      <td>0.59</td>
      <td>-0.61</td>
      <td>0.60</td>
      <td>-0.50</td>
      <td>0.49</td>
      <td>0.54</td>
      <td>0.37</td>
      <td>-0.37</td>
      <td>1.00</td>
      <td>-0.74</td>
    </tr>
    <tr>
      <th>MEDV</th>
      <td>-0.39</td>
      <td>0.36</td>
      <td>-0.48</td>
      <td>0.18</td>
      <td>-0.43</td>
      <td>0.70</td>
      <td>-0.38</td>
      <td>0.25</td>
      <td>-0.38</td>
      <td>-0.47</td>
      <td>-0.51</td>
      <td>0.33</td>
      <td>-0.74</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>

```

```

```
to_keep=[""]
```

```

```

```

```

```

```

```

```

```

```

```

```

```
# Lasso/Ridge Regression

```

```

```

```

```

```

```

```

```

```

```

```

```

```
sns.pairplot(df);
```

    <seaborn.axisgrid.PairGrid at 0x7f7d53b7c828>

![png](Dimensionality_Reduction_Algorithms_Boston_Corr_Matrix_Regression_files/Dimensionality_Reduction_Algorithms_Boston_Corr_Matrix_Regression_76_1.png)

```

```

```

```

```

```

In this story, we applied the concepts of linear regression on the Boston housing dataset. I would recommend to try out other datasets as well.
Here are a few places you can look for data
https://www.kaggle.com/datasets
https://toolbox.google.com/datasetsearch
https://archive.ics.uci.edu/ml/datasets.html

## References

1. abc
2. bhshsh
3. jhgde
