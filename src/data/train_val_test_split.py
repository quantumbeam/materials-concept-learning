import os
import pandas as pd
from sklearn.model_selection import train_test_split

print('loading data...')
ROOT_DIR = 'less_than_quinary20210610/'
ORIGINAL_FILE_NAME = 'less_than_quinary_asof2021_06_10.pkl'
mp_data = pd.read_pickle(ROOT_DIR + ORIGINAL_FILE_NAME)
mp_data_train_and_val, mp_data_test = train_test_split(mp_data, test_size=0.2, random_state=42)
mp_data_train, mp_data_val = train_test_split(mp_data_train_and_val, test_size=0.2, random_state=42)
print(f'loaded data: {ROOT_DIR}{ORIGINAL_FILE_NAME}')

print('train data: ', len(mp_data_train))
print('val data: ', len(mp_data_val))
print('test data: ', len(mp_data_test))

for save_dir in ['train_and_val', 'train', 'val', 'test']:
    if not os.path.exists(ROOT_DIR + save_dir):
        os.makedirs(ROOT_DIR + save_dir)
print('exporting...')
pd.to_pickle(mp_data_train, f'{ROOT_DIR}train/{ORIGINAL_FILE_NAME}')
pd.to_pickle(mp_data_val, f'{ROOT_DIR}val/{ORIGINAL_FILE_NAME}')
pd.to_pickle(mp_data_test, f'{ROOT_DIR}test/{ORIGINAL_FILE_NAME}')
print('finish!')