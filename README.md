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
  <img src="https://github.com/user-attachments/assets/9620ef52-ad5e-4c27-9ca1-702a8056d605">
</p>

  ## Data Preprocessing
  Next, we will use the `info` command to learn more about the data, including the number of rows and columns, data types, and the number of missing values.

  ```python
  data.info()
  ```
  <p align="center">
  <img src="https://github.com/user-attachments/assets/732e2560-587e-4734-be19-6f5c039cee22">
</p>

  ## Dealing with Missing Data

  ```python
  # to view the Missing values in each column:

  data.isnull().sum()
  ```
  <p align="center">
    <img src="https://github.com/user-attachments/assets/326a517c-1221-4e8d-9e1a-0d3882b36109">
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
    <img src="https://github.com/user-attachments/assets/90ae004b-c72f-4a9e-878e-c2b0192c5f23">
  </p>
  
  In the Embarked column, there are only two missing values. Let’s check the categories in this column.
  
  ```python
data['Embarked'].value_counts()
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/6555e06c-2191-4808-9698-3d92877a2e25">
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
      <img src="https://github.com/user-attachments/assets/e194ad29-f8cf-4df1-bf2f-80401e654d20">
    </p>
  
  ## Encode Categorical Columns

  The values in the Sex and Embarked columns are text, which cannot be directly used in a machine learning model. Therefore, we need to convert these text values into meaningful numerical values.
  
  For the Sex column, we will replace all male values with 0 and all female values with 1. Similarly, for the Embarked column: S => 0, C => 1, Q => 2.

  ```python
  data.replace({'Sex':{'male':0,'female':1},'Embarked':{'S':0,'C':1,'Q':2}}, inplace=True)
  data.head()
  ```

<p align="center">
    <img src="https://github.com/user-attachments/assets/08470892-204f-4e4d-942b-961d256a6f85">
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
      <img src="https://github.com/user-attachments/assets/9948ae30-1ec1-4db6-b225-82eb4b4927ad">
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
    <img src="https://github.com/user-attachments/assets/8cb32371-b8e8-4474-b8e3-e773b95d0f7c">
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
    <img src="https://github.com/user-attachments/assets/a508ed81-703b-40db-8130-ec904600ee68">
  </p>

  ```python
  # making a count plot for 'Survived' column
  
  sns.countplot(x='Survived', data=data)
  ```

  <p align="center">
      <img src="https://github.com/user-attachments/assets/bb40121f-0662-4ca6-87e3-3c029901a107">
    </p>
  
  ```python
  # making a count plot for 'Sex' column
  
  sns.countplot(x='Sex', data=data)
  ```

  <p align="center">
      <img src="https://github.com/user-attachments/assets/01918199-f52e-4a88-ad04-b8336319a174">
    </p>

```python
# now lets compare the number of survived beasd on the gender

sns.countplot(x='Sex', hue='Survived', data=data)
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/00b5cad6-9b9c-4956-85ce-09f21a23153c">
  </p>

  As we can see, even though there are more males in our dataset, the number of females who survived is higher. This is one of the very important insights we can get from this data.


```python
# now lets compare the number of survived beasd on the Pclass

sns.countplot(x='Pclass', hue='Survived', data=data)
```

<p align="center">
    <img src="https://github.com/user-attachments/assets/81329114-9e93-49a3-8fa2-8f9a0de123de">
  </p>
  
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
    <img src="https://github.com/user-attachments/assets/0b97af7e-6142-42af-9aba-023366ce694f">
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



