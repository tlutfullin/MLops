# Import libraries
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer 

# Define column types
target_col = 'binary_target'
categorical_cols = ['частота_пополнения']
continuous_cols = ['сумма', 'секретный_скор']
drop_col = ['client_id', 'mrg_',
            'регион', 'использование', 'частота', 'on_net',
            'зона_1', 'зона_2', 'pack', 'pack_freq',
            'доход', 'сегмент_arpu', 'объем_данных', 'продукт_1', 'продукт_2']

def import_data(path_to_file):

    # Get input dataframe
    input_df = pd.read_csv(path_to_file).drop(columns=drop_col)

    return input_df

# Function creating categories
def cat_create(train_df, test_df, col, n_cats):
    
    new_col = col + '_cat'
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Get tablee of categories
    temp_df = train_df.groupby(col, dropna=False)[[target_col]].count().sort_values(target_col, ascending=False)\
                      .reset_index().set_axis([col, 'count'], axis=1)\
                      .reset_index()
    temp_df['index'] = temp_df.apply(lambda x: np.nan if pd.isna(x[col]) else x['index'], axis=1)
    temp_df[new_col] = ['cat_NAN' if pd.isna(x) else 'cat_' + str(x) if x < n_cats else f'cat_{n_cats}+' for x in temp_df['index']]

    # Merge to initial dataset
    train_df = train_df.merge(temp_df[[col, new_col]], how='left', on=col).drop(columns=col)
    test_df = test_df.merge(temp_df[[col, new_col]], how='left', on=col).drop(columns=col)
    
    return train_df, test_df  


# Main preprocessing function
def run_preproc(input_df):

    # # Import Train dataset
    import os

    # # Путь к директории проекта
    # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # # Путь к файлу train.csv
    # train_file_path = os.path.join(BASE_DIR, 'app', 'train_data', 'train.csv')

    # # Использование этого пути для загрузки данных
    # train = pd.read_csv(train_file_path)

    # train_file_path = 'a/train_data/train.csv'

    # train = pd.read_csv(train_file_path)


    #train = pd.read_csv('./train_data/train.csv')

    # Создаем абсолютный путь к файлу
    #base_path = os.path.dirname(__file__)  # Это даст вам путь до файла, в котором выполняется этот код
    train_file_path = os.path.join('train_data', 'train.csv')

    # Проверяем, существует ли файл
    if os.path.exists(train_file_path):
        train = pd.read_csv(train_file_path)
    else:
        raise FileNotFoundError(f"Файл не найден по пути {train_file_path}")



    print('Train data imported...')

    # Convert columns to categorical
    for col in categorical_cols:
        train, input_df = cat_create(train, input_df, col, 15)
    print('Categorical features created...')

    # Run mean ecoding for continuous variables
    for col in categorical_cols:

        # Create table of means
        means_tb = train.groupby(col + '_cat')[[target_col]].mean()\
                        .reset_index().rename(columns={target_col:f'{col}_mean_enc'})
        
        # Fill non-matched values of categorical columns with default category
        input_df[col + '_cat'] = input_df[col + '_cat'].fillna('cat_NAN')
        
        # Join to datasets
        train = train.merge(means_tb, how='left', on=col + '_cat')
        input_df = input_df.merge(means_tb, how='left', on=col + '_cat')
    print('Mean encoding complete...')

    # Impute empty values with mean value
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
    imputer = imputer.fit(train[continuous_cols])
    print('Imputing complete...')

    # Create dataframe 
    output_df = pd.concat([
        input_df.drop(columns=continuous_cols),
        pd.DataFrame(imputer.transform(input_df.copy()[continuous_cols]), columns=continuous_cols)
    ], axis=1)

    # Return resulting dataset
    return output_df