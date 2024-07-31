# Titanic Machine Learning

## Importing the Dependencies

  ```python
  #Importing the Dependencies

  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  import seaborn as sns

  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression
  from sklearn.metrics import accuracy_score

  ```
  ## Reading the data

  We will first use `pd.read_csv` to load the data from a CSV file into a Pandas DataFrame:

  ```python
  data = pd.read_csv('/content/train.csv')
  ```

After loading the data, we will review it using the `head` command to ensure it has been read correctly.

  ```python
  #this will print first 5 rows in the dataset

  data.head()
  ```

  <p align="center">
  <img src="https://github.com/user-attachments/assets/44a47a61-b1b6-42fc-824b-82a14938b2c0">
</p>

  ## Data Preprocessing
  Next, we will use the `info` command to learn more about the data, including the number of rows and columns, data types, and the number of missing values.

  ```python
  data.info()
  ```
  <p align="center">
  <img src="https://github.com/user-attachments/assets/78fb8c65-278a-4999-833e-4630394581da">
</p>

  ## Dealing with Missing Data

  ```python
  # to view the Missing values in each column:

  data.isnull().sum()
  ```
  <p align="center">
    <img src="https://github.com/user-attachments/assets/99b9821f-ad3d-4f3f-bd17-df3dc1498f6b">
  </p>

  You have three options to address this:

- Delete rows that contain missing values
- Delete the entire column that contains missing values
- Replace missing values with a specific value (Mean, Median, Mode, constant)

There are three columns with missing values: Age, Cabin, and Embarked. For the Age column, we will fill the missing values with the mean, as it is a simple and quick method to handle missing data and helps maintain the overall distribution of the dataset.


  ```python
  data['Age'] = data['Age'].fillna(data['Age'].mean())
  data['Age'].isnull().sum() #Output = 0
  ```
  

  The Cabin column has many missing values, so we will drop it from the dataset.
  
  ```python
  data = data.drop(['Cabin'], axis=1)
  data.head()
  ```

<p align="center">
    <img src="https://github.com/user-attachments/assets/7c079533-089c-4301-ad9d-db1ed2e9090d">
  </p>
  
  In the Embarked column, there are only two missing values. Let’s check the categories in this column.
  
  ```python
data['Embarked'].value_counts()
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/31da1a67-5a4e-4fde-afb3-bf5153b8e93a">
  </p>
  
  ```python
  data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])
  data['Embarked'].isnull().sum() #Output = 0
  ```


  ## Drop useless columns
  
  As you know, the PassengerId and Name of the passenger do not affect the probability of survival. Additionally, the Ticket column does not have a clear relationship to the survival of passengers, so these columns will be dropped.

```python
# Drop the PassengerId and Name Columns from the dataset:

data = data.drop(['PassengerId', 'Name', 'Ticket'], axis=1)
data.head()
```

  <p align="center">
      <img src="https://github.com/user-attachments/assets/b6ae3c83-83f8-4f60-9200-8a1b68865c0b">
    </p>
  
  ## Encode Categorical Columns

  The values in the Sex and Embarked columns are text, which cannot be directly used in a machine learning model. Therefore, we need to convert these text values into meaningful numerical values.
  
  For the Sex column, we will replace all male values with 0 and all female values with 1. Similarly, for the Embarked column: S => 0, C => 1, Q => 2.

  ```python
  data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
  data.head()
  ```

<p align="center">
    <img src="https://github.com/user-attachments/assets/75480f3e-752f-41b0-b566-0aede8b59098">
  </p>

  ## Dealing with Duplicates

  Check for duplicates in the dataset:
  
  ```python
  data.duplicated().sum() #Output = 111
  ```

  ```python
  data.drop_duplicates(inplace=True)
  ```

## Data Analysis

  In this section, we will explore the data and the relationships between features using statistical analysis and visualization techniques. This will help us understand the underlying patterns and correlations in the dataset, providing valuable insights for model building.
  
  The `describe()` function provides summary statistics for numerical columns, including count, mean, standard deviation, min, max, and quartiles. This function helps us understand the distribution and central tendencies of the data. However, in our Titanic dataset, while useful, it may not be the primary focus since many insights come from categorical features and their relationships with survival, which are better explored through other means.

  ```python
  data.describe()
  ```

  <p align="center">
      <img src="https://github.com/user-attachments/assets/6e8a256f-370c-46a9-ac51-3a2baf200742">
    </p>
  
  ## Look for Correlations

  To understand the relationships between features, we can use a correlation matrix, which shows the correlation coefficients between different features in a dataset. Each cell in the matrix represents the correlation between two features. The correlation coefficient ranges from -1 to 1, where:
  
  - **1** indicates a perfect positive correlation: as one feature increases, the other feature increases proportionally.
  - **-1** indicates a perfect negative correlation: as one feature increases, the other feature decreases proportionally.
  - **0** indicates no correlation: the features do not show any linear relationship.

  ```python
  data.corr()['Survived']
  ```

<p align="center">
    <img src="https://github.com/user-attachments/assets/a0814017-1487-4276-ad31-3b88d47fda8c">
  </p>
  
  The correlation values provide insights into how different features relate to the survival outcome in the Titanic dataset:
  
  - **Pclass**: Negative correlation (-0.338). Higher classes (lower number) are more likely to survive.
  - **Sex**: Positive correlation (0.543). Females are more likely to survive.
  - **Age**: Slight negative correlation (-0.070). Older passengers have a marginally lower chance of survival.
  - **SibSp**: Slight negative correlation (-0.035). Having more siblings/spouses aboard slightly decreases survival chances.
  - **Parch**: Slight positive correlation (0.082). Having more parents/children aboard slightly increases survival chances.
  - **Fare**: Positive correlation (0.257). Passengers who paid higher fares are more likely to survive.
  - **Embarked**: Slight positive correlation (0.107). The port of embarkation has a minor effect on survival.
  
  These correlations help identify which features may be important for predicting survival.
  
  ```python
  # to understand more about data lets find the number of people survived and not survived
  
  data['Survived'].value_counts()
  ```

<p align="center">
    <img src="https://github.com/user-attachments/assets/a560ccf8-20c5-426b-baac-ac3cb748058f">
  </p>

  ```python
  # making a count plot for 'Survived' column
  
  sns.countplot(x='Survived', data=data)
  ```

  <p align="center">
      <img src="https://github.com/user-attachments/assets/4ab6ed09-025e-404c-a626-b9aa9c470256">
    </p>
  
  ```python
  # making a count plot for 'Sex' column
  
  sns.countplot(x='Sex', data=data)
  ```

  <p align="center">
      <img src="https://github.com/user-attachments/assets/b126c28c-955c-4703-b3b3-ab30aa90101e">
    </p>

```python
# now lets compare the number of survived beasd on the gender

sns.countplot(x='Sex', hue='Survived', data=data)
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/4719e1e8-ce4e-46ee-a9f3-85dbd6883637">
  </p>

  As we can see, even though there are more males in our dataset, the number of females who survived is higher. This is one of the very important insights we can get from this data.


```python
# now lets compare the number of survived beasd on the Pclass

sns.countplot(x='Pclass', hue='Survived', data=data)
```
  You can do the same for the other columns to get more insights about the dataset.

## Model Building

**Separating features & Target**

  Separating features and target so that we can prepare the data for training machine learning models. In the Titanic dataset, the Survived column is the target variable, and the other columns are the features.

``` python
x = data.drop(columns = ['Survived'], axis=1)
y = data['Survived']
```
**Splitting the data into training data & Testing data**

  To effectively build and assess a machine learning model, it’s crucial to divide the dataset into training and testing sets. The training set is utilized to teach the model, enabling it to recognize patterns and relationships within the data. Conversely, the testing set is employed to evaluate the model’s performance on new, unseen data, ensuring it can generalize well to new instances. This division helps prevent overfitting and offers a reliable estimate of the model’s predictive accuracy.

  ```python
  from sklearn.model_selection import train_test_split
  
  # Split the data into training data & Testing data using train_test_split function :

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
  ```

## Model Training

 Model training is a crucial step in machine learning where the algorithm learns from the training data to make predictions. Logistic Regression is a commonly used algorithm for binary classification tasks, such as predicting whether a passenger survived in the Titanic dataset. By training the model on our training data, we aim to find the best-fit parameters that minimize prediction errors. Once trained, this model can be used to predict outcomes on new, unseen data.

  ```python
  from sklearn.linear_model import LogisticRegression
  
  # Create a Logistic Regression model and Train it on the training data:
  
  model = LogisticRegression(max_iter=1000)
  model.fit(x_train, y_train)
  ```

<p align="center">
    <img src="https://github.com/user-attachments/assets/2068786b-b657-42f4-8536-bd78888fbd0d">
  </p>
  
  ## Model Evaluation
  
  Evaluating a model is essential in machine learning to determine how well a trained model performs on testing data. The accuracy score, a widely used evaluation metric, indicates the proportion of correct predictions out of all predictions. This metric helps assess the model's effectiveness, ensures it generalizes well to new data, and guides further improvements.

  ```python
  from sklearn.metrics import accuracy_score
  
  # First, let the model predict the outcomes for x_test
  # Then, use the accuracy score to evaluate the model's performance
  # Finally, print the accuracy score
  
  x_test_prediction = model.predict(x_test)
  training_data_accuracy = accuracy_score(y_test, x_test_prediction)
  print('Accuracy:', training_data_accuracy)
  
  ```



