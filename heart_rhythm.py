# Importing the datasets from the catalog (you must first run the project_final script in order to import the data)
# Change the path to the place where you saved the dataframes
import os
import pandas as pd
os.chdir(r"C:\\Users\macie\\Desktop\\Semestr 5\\Sieci neuronowe\\heart_rhythm\\data")

df1 = pd.read_csv("df_mean_range_final")
df2 = pd.read_csv("df_std_range_final")
#%% Describing the DataFrame
print(df1.describe())
#%% Checking the types
print(df1.dtypes)
#%% Visualising the correlations
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df1.iloc[:,1:12].corr(),cmap='coolwarm',annot=True,lw=1)
plt.show()
#%% Scaling the columns so that their mean = 0 and std = 1 using the StandardScaler()
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

scaled_features = df1.copy()

col_names = ['Most common value', 'Minimum', 'Maximum', 'Mean', 'STD', 'SDNN', 'RMSSD', 'pNN50', 'pNN20']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

scaled_features[col_names] = features
print(scaled_features.head())
#%% Assigning the data to X and y
X = scaled_features.iloc[:, 3:53]
y = scaled_features.iloc[:, 1]
#%% Splitting the data to train and test using stratify split
from sklearn.model_selection import train_test_split
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)
print('X train', X_train.shape, 'X test', X_test.shape)
print('y train', y_train.shape, 'y test', y_test.shape)

#%% PCA with 2 components 
from sklearn.decomposition import PCA

# We cannot use more than 2 components because the number of components must be 
# between 0 and min(n_samples, n_features)=2 with svd_solver = 'full'

pca = PCA(n_components = 2)
  
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
  
explained_variance = pca.explained_variance_ratio_
print('This is the explained variance: \n', explained_variance)
#%% Checking the shape before and after the PCA
print('Shape before PCA: ', X.shape)
print('Shape after PCA: ', X_train.shape)
 
# Creating a pca DataFrame
pca_df = pd.DataFrame(
    data=X_train, 
    columns=['PC1', 'PC2'])
print('This is the pca DataFrame: \n', pca_df)

#%%% Fitting using SGDRegressor
from sklearn.linear_model import SGDRegressor
import numpy as np

sgdr = SGDRegressor()
print(sgdr)

sgdr.fit(X_train, y_train)

score = sgdr.score(X_train, y_train)
print("R-squared:", score)

print('The coefficients of the linear regression: ')
print(sgdr.intercept_)
print("The intercept (often labeled the constant) is the expected mean value of Y when all X=0. \nIn a purely mathematical sense, this definition is correct. \nUnfortunately, it’s frequently impossible to set all variables to zero because this combination can be an impossible or irrational arrangement.")
#%% The coeeff parameters
coeff_parameter = pd.DataFrame(sgdr.coef_,columns=['Coefficient'])
print('A negative sign indicates that as the predictor variable increases, the Target variable decreases: ', coeff_parameter)
#%% Searching for the best hyperparameters

from sklearn.model_selection import GridSearchCV

moje_cv = 5

params = {
    'loss'  :['squared_error', 'huber'],
    'alpha' : [ 0.01, 0.1],
    'penalty' :['l2','l1']#,'elasticnet','none']
        }


grid = GridSearchCV(sgdr, param_grid =params, cv=moje_cv, scoring ='explained_variance',
                     return_train_score=True)

grid.fit(X_train, y_train)

cv_res = grid.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(mean_score, params)

#%% Getting the final model params

final_model = grid.best_estimator_
final_model_param = final_model.get_params()
final_training = final_model.fit(X_train,y_train)
#%% The validation process

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(final_model, X_train, y_train, cv = moje_cv, scoring="explained_variance")
sign = 'The accuracy of good predictions: '
print(sign, cv_score)
#%% Our early summary
print("CV mean score and STD values: %.3f"% cv_score.mean(), "+/- %.3f"% cv_score.std())
#%% Predicting our X_test
final_predictions = final_training.predict(X_test)
print("These are our predictions: \n", final_predictions)
#%% Rounding our predictions up to the nearest tens
for i in range(len(final_predictions)):
    final_predictions[i] = round(final_predictions[i], -1)

#%% Printing our final predictions without decimal points and ones
print("These are our predictions: \n", final_predictions)
#%% Calculating the mean squared error (MSE)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, final_predictions)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0)) 

#%% Plotting the y_test and predicted values

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, final_predictions, linewidth=1.1, label="predicted")
plt.title("y_test and final_predictions data")
plt.xlabel('X_test')
plt.ylabel('Age')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()
#%% Checking how our model looks like

sns.regplot(x = y_test, y = final_predictions, color = "r", marker = '*')
plt.xlabel('True age')
plt.ylabel("Age predicted")
plt.title("True age vs age predicted")
plt.grid(True)
plt.show()
#%% The error matrix

from sklearn.model_selection import cross_val_predict

y_train_predict = cross_val_predict(final_model, X_train, y_train, cv= moje_cv)
for i in range(len(y_train_predict)):
    y_train_predict[i] = round(y_train_predict[i], -1)
    
print("The prediction created by cross validation: ", y_train_predict)
accuracy_score = sum(y_train == y_train_predict)/len(y_train)
print('The accuracy score of train data: ', round(accuracy_score,2),'%')
#%% The confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, final_predictions)
print(f'The confusion matrix: \n{cm}')
#%% Displaying the confusion matrix in a nicer way
from sklearn.metrics import ConfusionMatrixDisplay

cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
#%%# Bar plot of explained_variance

plt.bar(
    range(1,len(explained_variance)+1),
    pca.explained_variance_
    )

plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.show()
#%% Bar plot of cumulative explained_variance
plt.bar(
    range(1,len(explained_variance)+1),
    pca.explained_variance_
    )

plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Cumulative Explained Variance')
 
plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')
 
plt.show()

#%% 
print("Doing the same for df2(df_std_range_final)")
#%% Describing the DataFrame
print(df2.describe())
#%% Checking the types
print(df2.dtypes)
#%% Visualising the correlations
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df2.iloc[:,1:12].corr(),cmap='coolwarm',annot=True,lw=1)
plt.show()
#%% Scaling the columns so that their mean = 0 and std = 1 using the StandardScaler()
#from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

scaled_features = df2.copy()

col_names = ['Most common value', 'Minimum', 'Maximum', 'Mean', 'STD', 'SDNN', 'RMSSD', 'pNN50', 'pNN20']
features = scaled_features[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)

scaled_features[col_names] = features
print(scaled_features.head())
#%% Assigning the data to X and y
X = scaled_features.iloc[:, 3:53]
y = scaled_features.iloc[:, 1]
#%% Splitting the data to train and test using stratify split
from sklearn.model_selection import train_test_split
  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1, stratify=y)
print('X train', X_train.shape, 'X test', X_test.shape)
print('y train', y_train.shape, 'y test', y_test.shape)

#%% PCA with 2 components 
from sklearn.decomposition import PCA

# We cannot use more than 2 components because the number of components must be 
# between 0 and min(n_samples, n_features)=2 with svd_solver = 'full'

pca = PCA(n_components = 2)
  
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
  
explained_variance = pca.explained_variance_ratio_
print('This is the explained variance: \n', explained_variance)
#%% Checking the shape before and after the PCA
print('Shape before PCA: ', X.shape)
print('Shape after PCA: ', X_train.shape)
 
# Creating a pca DataFrame
pca_df = pd.DataFrame(
    data=X_train, 
    columns=['PC1', 'PC2'])
print('This is the pca DataFrame: \n', pca_df)


#%%% Fitting using SGDRegressor
from sklearn.linear_model import SGDRegressor
import numpy as np

sgdr = SGDRegressor()
print(sgdr)

sgdr.fit(X_train, y_train)

score = sgdr.score(X_train, y_train)
print("R-squared:", score)

print('The coefficients of the linear regression: ')
print(sgdr.intercept_)
print("The intercept (often labeled the constant) is the expected mean value of Y when all X=0. \nIn a purely mathematical sense, this definition is correct. \nUnfortunately, it’s frequently impossible to set all variables to zero because this combination can be an impossible or irrational arrangement.")
#%% The coeeff parameters
coeff_parameter = pd.DataFrame(sgdr.coef_,columns=['Coefficient'])
print('A negative sign indicates that as the predictor variable increases, the Target variable decreases: ', coeff_parameter)
#%% Searching for the best hyperparameters

from sklearn.model_selection import GridSearchCV

moje_cv = 5

params = {
    'loss'  :['squared_error', 'huber'],
    'alpha' : [ 0.01, 0.1],
    'penalty' :['l2','l1']#,'elasticnet','none']
        }


grid = GridSearchCV(sgdr, param_grid =params, cv=moje_cv, scoring ='explained_variance',
                     return_train_score=True)

grid.fit(X_train, y_train)

cv_res = grid.cv_results_
for mean_score, params in zip(cv_res["mean_test_score"], cv_res["params"]):
    print(mean_score, params)

#%% Getting the final model params

final_model = grid.best_estimator_
final_model_param = final_model.get_params()
final_training = final_model.fit(X_train,y_train)
#%% The validation process

from sklearn.model_selection import cross_val_score

cv_score = cross_val_score(final_model, X_train, y_train, cv = moje_cv, scoring="explained_variance")
sign = 'The accuracy of good predictions: '
print(sign, cv_score)
#%% Our early summary
print("CV mean score and STD values: %.3f"% cv_score.mean(), "+/- %.3f"% cv_score.std())
#%% Predicting our X_test
final_predictions = final_training.predict(X_test)
print("These are our predictions: \n", final_predictions)
#%% Rounding our predictions up to the nearest tens
for i in range(len(final_predictions)):
    final_predictions[i] = round(final_predictions[i], -1)

#%% Printing our final predictions without decimal points and ones
print("These are our predictions: \n", final_predictions)
#%% Calculating the mean squared error (MSE)
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, final_predictions)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0)) 

#%% Plotting the y_test and predicted values

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, linewidth=1, label="original")
plt.plot(x_ax, final_predictions, linewidth=1.1, label="predicted")
plt.title("y_test and final_predictions data")
plt.xlabel('X_test')
plt.ylabel('Age')
plt.legend(loc='best',fancybox=True, shadow=True)
plt.grid(True)
plt.show()
#%% Checking how our model looks like

sns.regplot(x = y_test, y = final_predictions, color = "r", marker = '*')
plt.xlabel('True age')
plt.ylabel("Age predicted")
plt.title("True age vs age predicted")
plt.grid(True)
plt.show()
#%% The error matrix

from sklearn.model_selection import cross_val_predict

y_train_predict = cross_val_predict(final_model, X_train, y_train, cv= moje_cv)
for i in range(len(y_train_predict)):
    y_train_predict[i] = round(y_train_predict[i], -1)
    
print("The prediction created by cross validation: ", y_train_predict)
accuracy_score = sum(y_train == y_train_predict)/len(y_train)
print('The accuracy score of train data: ', round(accuracy_score,2),'%')
#%% The confusion matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, final_predictions)
print(f'The confusion matrix: \n{cm}')
#%% Displaying the confusion matrix in a nicer way
from sklearn.metrics import ConfusionMatrixDisplay

cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()
#%%# Bar plot of explained_variance

plt.bar(
    range(1,len(explained_variance)+1),
    pca.explained_variance_
    )

plt.xlabel('PCA Feature')
plt.ylabel('Explained variance')
plt.title('Feature Explained Variance')
plt.show()
#%% Bar plot of cumulative explained_variance
plt.bar(
    range(1,len(explained_variance)+1),
    pca.explained_variance_
    )

plt.plot(
    range(1,len(pca.explained_variance_ )+1),
    np.cumsum(pca.explained_variance_),
    c='red',
    label='Cumulative Explained Variance')
 
plt.legend(loc='upper left')
plt.xlabel('Number of components')
plt.ylabel('Explained variance (eignenvalues)')
plt.title('Scree plot')
 
plt.show()
#%%
print("The end of project 2 :-)")
print("Author: Maciej Ossowski")