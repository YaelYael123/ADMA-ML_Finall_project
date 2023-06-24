import pandas as pd
import numpy as np
import re
import datetime

# Read the data
dataframe = pd.read_excel('C:/Users/YaelD/Desktop/output_all_students_Train_v10.xlsx', header=0)

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
    order_columns = ['City','type', 'room_number','Area', 'Street', 'number_in_street', 'city_area', 'floor','total_floors', 'num_of_images',
                 'hasElevator ', 'hasParking ', 'hasBars ', 'hasStorage ', 'condition ', 'hasAirCondition ', 'hasBalcony ', 'hasMamad ',
                 'handicapFriendly ','entranceDate ', 'furniture ','publishedDays ', 'description ', 'price']
    dataframe = dataframe[order_columns]
    
    return dataframe

# Data cleansing
prepare_data(dataframe)