import pandas as pd
import numpy as np
import re
import datetime

from sklearn.model_selection import train_test_split, cross_val_score, KFold
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import ppscore as pps
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer    
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn import linear_model, decomposition, set_config
set_config(display='diagram')
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Read the data
dataset = pd.read_excel('output_all_students_Train_v10.xlsx', header=0)

def prepare_data(dataframe):
    # We will convert the price to a numerical value
    dataframe['price'] = dataframe['price'].apply(lambda x: re.findall(r'\d+\.?\d*', str(x)))
    dataframe['price'] = dataframe['price'].apply(lambda x: float(''.join(x)) if len(x) > 0 else None)
    
    # We will discard all rows without values
    dataframe.dropna(subset=['price'], inplace=True)
    
    # We will convert the Area to a numerical value
    dataframe['Area'] = dataframe['Area'].apply(lambda x: re.findall(r'\d+\.?\d*', str(x)))
    dataframe['Area'] = dataframe['Area'].apply(lambda x: float(''.join(x)) if len(x) > 0 else None)
    outlier = (dataframe['Area'] > 950.0) & (dataframe['price'] < 5000000)
    dataframe.loc[~outlier, :] = dataframe.loc[~outlier, :]

    # Cleaning up unnecessary punctuation
    dataframe['City'] = dataframe['City'].apply(lambda x: re.sub(r'[^\w\s"]', '', str(x)).lstrip().rstrip())
    dataframe['City'] = dataframe['City'].replace('נהרייה', 'נהריה')
    dataframe['type'] = dataframe['type'].apply(lambda x: re.sub(r'[^\w\s"]', '', str(x)).lstrip().rstrip())
    dataframe['type'] = dataframe['type'].replace('דירת נופש', 'דירה')
    dataframe['type'] = dataframe['type'].replace('קוטג', 'בית פרטי')
    dataframe['Street'] = dataframe['Street'].apply(lambda x: re.sub(r'[^\w\s"]', '', str(x)).lstrip().rstrip())
    fill_position_error = (dataframe['Street'].str.contains('שכונה', case=False, na=False) |
        dataframe['Street'].str.contains('בשכונת', case=False, na=False) |
        dataframe['Street'].str.contains('השכונה', case=False, na=False)) & \
       dataframe['city_area'].isna()
    dataframe.loc[fill_position_error, 'city_area'] = dataframe.loc[fill_position_error, 'Street']
    dataframe['city_area'] = dataframe['city_area'].apply(lambda x: re.sub(r'[^\w\s"]', '', str(x)).lstrip().rstrip())
    dataframe['description '] = dataframe['description '].apply(lambda x: re.sub(r'[^\w\s"]', '', str(x)).lstrip().rstrip())
    dataframe['room_number'] = dataframe['room_number'].apply(lambda x: re.findall(r'\d+\.?\d*', str(x)))
    dataframe['room_number'] = dataframe['room_number'].apply(lambda x: float(''.join(x)) if len(x) > 0 else None)
    # We will assume that villas with more than 25 rooms are not sold in Israel,
    # therefore any number beyond this number is a typing error in which they forgot to put a period
    dataframe['room_number'] = dataframe['room_number'].apply(lambda x: x / 10 if x > 25 else x)
    dataframe['number_in_street'] = dataframe['number_in_street'].apply(lambda x: re.findall(r'\d+\.?\d*', str(x)))
    dataframe['number_in_street'] = dataframe['number_in_street'].apply(lambda x: int(''.join(x)) if len(x) > 0 else None)
    dataframe['number_in_street'] = dataframe['number_in_street'].astype(float).astype('Int64')
    
    # column split floor_out_of to floor and total_floors
    dataframe['floor'] = dataframe['floor_out_of'].apply(lambda x: re.findall(r'(?:קומה|קומת) (\w+)', str(x))[0] if re.search(r'(?:קומה|קומת) \w+', str(x)) else 'none')
    dataframe['total_floors'] = dataframe['floor_out_of'].apply(lambda x: re.findall(r'מתוך (\w+)', str(x))[0] if re.search(r'מתוך \w+', str(x)) else 'none')  
    dataframe.drop('floor_out_of', axis=1, inplace=True)
    dataframe['floor'] = dataframe['floor'].apply(lambda x: '0' if x == 'קרקע' else x)
    dataframe['floor'] = dataframe['floor'].apply(lambda x: '-1' if x == 'מרתף' else x)
    dataframe['floor'] = dataframe['floor'].apply(lambda x: float(x) if x != 'none' else None)
    dataframe['floor'] = dataframe['floor'].astype(float).astype('Int64')
    dataframe['total_floors'] = dataframe['total_floors'].apply(lambda x: float(x) if x != 'none' else None)
    dataframe['total_floors'] = dataframe['total_floors'].astype(float).astype('Int64')
    
    # Updating the entranceDate column to the categorical values
    def categorize_entrance_date(date):
        if isinstance(date, str):
            date = re.sub(r'[^\w\s]', '', date.lstrip().rstrip())
            if date == 'מיידי':
                return 'less_than_6 months'
            elif date == 'גמיש':
                return 'flexible'
            elif date == 'לא צויין':
                return 'not_defined'
        elif isinstance(date, pd.Timestamp) or isinstance(date, datetime.datetime):
            current_date = pd.Timestamp.now()
            if date < current_date + pd.DateOffset(months=6):
                return 'less_than_6 months'
            elif date < current_date + pd.DateOffset(months=12):
                return 'months_6_12'
            elif date > current_date + pd.DateOffset(months=12):
                return 'above_year'
        else:
            return np.nan

    dataframe['entranceDate '] = dataframe['entranceDate '].apply(categorize_entrance_date)

    # Define the mapping dictionary
    mapping = {
        True: 1,
        False: 0,
        'כן': 1,
        'לא': 0,
        np.nan: np.nan,
        'יש מעלית': 1,
        'אין מעלית': 0,
        'יש': 1,
        'אין': 0,
        'no': 0,
        'yes': 1,
        'יש חניה': 1,
        'אין חניה': 0,
        'יש חנייה': 1,
        'אין חנייה': 0,
        'אין סורגים': 0,
        'יש סורגים': 1,
        'אין מחסן': 0,
        'יש מחסן': 1,
        'יש מיזוג אוויר': 1,
        'אין מיזוג אוויר': 0,
        'יש מיזוג אויר': 1,
        'אין מיזוג אויר': 0,
        'אין מרפסת': 0,
        'יש מרפסת': 1,
        'אין ממ"ד': 0,
        'יש ממ"ד': 1,
        'אין ממ״ד': 0,
        'יש ממ״ד': 1,
        'לא נגיש לנכים': 0,
        'נגיש לנכים': 1,
        'נגיש': 1,
        'לא נגיש': 0
    }

    # Apply the mapping to the desired columns
    columns_to_convert = ['hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ', 'handicapFriendly ']
    dataframe[columns_to_convert] = dataframe[columns_to_convert].replace(mapping)
    dataframe[columns_to_convert] = dataframe[columns_to_convert].applymap(lambda x: int(x) if x != 'none' and pd.notnull(x) else None)
    dataframe[columns_to_convert] = dataframe[columns_to_convert].astype(float).astype('Int64')
    
    # Handling the description column and converting it to binary
    dataframe['description '] = dataframe['description '].fillna(0).apply(lambda x: 1 if x != 0 else 0)
    dataframe['description '] = dataframe['description '].astype('Int64')
    
    # Handling the condition column There should be no duplicate entries
    dataframe['condition '] = dataframe['condition '].replace(['None', False], np.nan)
    
    # Handling the publishedDays column and converting it to int
    dataframe['publishedDays '] = dataframe['publishedDays '].apply(lambda x: re.sub(r'[^\w\s"]', '', str(x)).lstrip().rstrip())
    dataframe['publishedDays '] = dataframe['publishedDays '].replace(['None', 'nan', '', 'Nan'], np.nan)
    dataframe['publishedDays '] = dataframe['publishedDays '].replace(['חדש'], '0')
    dataframe['publishedDays '] = dataframe['publishedDays '].astype(float).astype('Int64')
    
    # Handling the num_of_images column and converting it to int
    dataframe['num_of_images'] = dataframe['num_of_images'].astype(float).astype('Int64')

    # The arrangement of the columns, and the predicted column shift to the right
    order_columns = ['City','type', 'room_number','Area', 'Street', 'number_in_street', 'city_area', 'num_of_images', 'floor','total_floors',
                 'hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'condition ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ',
                 'handicapFriendly ','entranceDate ', 'furniture ','publishedDays ', 'description ', 'price']
    dataframe = dataframe[order_columns]
    
    return dataframe

# Data cleansing
dataset = prepare_data(dataset)

# Dependent & independent variables
X = dataset.iloc[:, :-1] 
y = dataset.iloc[:, -1] 

########## Tests for feature engineering start ##########
# Train and Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# We will check the correlation in the columns
spearman_X_train = X_train.corr(method = 'spearman')
sns.heatmap(spearman_X_train, cmap='coolwarm', center=0)
plt.figure(figsize=(35,35))

# There is a high correlation between the number of rooms and the area of the apartment (fairly logical) 
# therefore in order to optimize the model we decided to drop the column number of rooms
X_train = X_train.drop('room_number', axis=1)
# There is also a high correlation between the floor and the number of floors, we will drop out the total floors column
X_train = X_train.drop('total_floors', axis=1)

# We will make a copy for information in order to check on it the VIF
# Convert numeric columns to float
X_train_copy = X_train
num_cols_copy = num_cols = [col for col in X_train.columns if (X_train[col].dtypes!='O')]
X_train_copy[num_cols_copy] = X_train_copy[num_cols_copy].astype(float)
# Check if num_cols_copy is empty
if len(num_cols_copy) > 0:
    simp = SimpleImputer(strategy='most_frequent', add_indicator=False)
    simp.fit(X_train_copy[num_cols_copy])
    X_train_copy[num_cols_copy] = simp.transform(X_train_copy[num_cols_copy])
X_train_copy.isnull().sum().sort_values()
# Calculate VIF
vif_data = pd.DataFrame()
vif_data["Features"] = num_cols_copy
vif_data["VIF"] = [variance_inflation_factor(X_train_copy[num_cols_copy].values, i) for i in range(len(num_cols_copy))]
vif_data = vif_data.sort_values(by='VIF', ascending=False)
vif_data

# VIF score above 5 we drop
X_train = X_train.drop('description ', axis=1)

# We will check using PPS that can be used as an alternative to the correlation (matrix).
predictors_X_train = pps.predictors(X_train, y='floor')
pps_matrix_X_train = pps.matrix(X_train)[['x', 'y', 'ppscore']].pivot(columns='x', index='y', values='ppscore')
sns.heatmap(pps_matrix_X_train, vmin=0, vmax=1, cmap="Blues", linewidths=0.5)
# We didn't find anything significant, so we will continue

########## Tests for feature engineering end ##########

# Feature_Engineering According to the tests done above
X_Feature_Engineering = X.drop(['room_number', 'total_floors', 'description '], axis=1)

# Numeric columns
num_cols = [col for col in X_Feature_Engineering.columns if X_Feature_Engineering[col].dtypes != 'O']
                               
# Categorical columns
cat_cols = [col for col in X_Feature_Engineering.columns if (X_Feature_Engineering[col].dtypes=='O')]
                               
# Preprocessing for the numerical columns
numerical_pipeline = Pipeline([
    ('numerical_imputation', SimpleImputer(strategy='median', add_indicator=False)),
    ('scaling', MinMaxScaler())
])
                        
# Preprocessing for the numerical columns
categorical_pipeline = Pipeline([
    ('categorical_imputation', SimpleImputer(strategy='most_frequent', add_indicator=False)),
    ('one_hot_encoding', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
])

# Transformation to information
column_transformer = ColumnTransformer([
     ('numerical_preprocessing', numerical_pipeline, num_cols),
     ('categorical_preprocessing', categorical_pipeline, cat_cols)
], remainder='passthrough')
                               
# Dimension reduction by using PCA
# Principal Component Analysis(PCA) will reduce the dimension of features
# by creating new features which have most of the varience of the original data
pca = decomposition.PCA(n_components = 3)
                               
# Pipe to training the model
ElasticNet_pipe_preprocessing_model = Pipeline([
    ('preprocessing_step', column_transformer),
    ('Dimension_reduction', pca),
    ('ElasticNet_model', linear_model.ElasticNet(alpha=0.9, l1_ratio=0.8, selection='random'))
])

# We arrived at the parameters in PCA and EN by running on the GridSearchCV, the problem is that it takes a lot of time to run,
# so if we had a computer with greater computational power we would run it a few more times and with more options.
                               
# Perform cross-validation to examining the performance of the model
cv = KFold(n_splits=10, shuffle = True, random_state = 11)
cv_result = cross_val_score(ElasticNet_pipe_preprocessing_model, X_Feature_Engineering, y, cv=cv, scoring='neg_mean_squared_error',error_score='raise')
rmse_scores = np.sqrt(-cv_result)

# Print the results
print("Cross-validation scores:", cv_result)
print("Average MSE:", np.mean(-cv_result))
print("Average STD:", np.std(-cv_result))
print("Average RMSE:", np.mean(rmse_scores))

# trained model
ElasticNet_pipe_preprocessing_model.fit(X_train, y_train)

# pickle dump
pickle.dump(ElasticNet_pipe_preprocessing_model, open("trained_model.pkl","wb"))

# Examining the performance on the test data set
X_test = X_test.drop(['room_number', 'total_floors', 'description '], axis=1)
y_pred = ElasticNet_pipe_preprocessing_model.predict(X_test)

def score_mse_model(y_test, y_pred, model_name):
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)

    #R_squared = r2_score(y_test, y_pred)

    print(f"Model test results: {model_name} ,MSE: {np.round(MSE, 2)} ,RMSE: {np.round(RMSE, 2)}")
    
# Model performance evaluation on test sample
score_mse_model(y_test, y_pred, "linear_model.ElasticNet")