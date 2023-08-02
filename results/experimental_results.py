import os
import pandas as pd
from utils import transform_address
# dirs = ['AIDS', 'MUTAG', 'Mutagenicity', 'COX2_MD']
dirs = ['Mutagenicity']
expls = ['PT']

for fols in dirs:
    for ex_tp in expls:
        a = transform_address(os.getcwd() + f'\\{fols}' + f'\\{ex_tp}')
        csvs = [x for x in os.listdir(a) if x.__contains__('.csv')]
        d = []
        for c in csvs:
            tmp = pd.read_csv(transform_address(os.getcwd() + f'\\{fols}' + f'\\{ex_tp}' + f'\\{c}'))
            temp_df = tmp[:-2].copy()
            temp_df['file'] = c
            temp_df['pgexpl'] = tmp['accuracy'].iloc[-1]
            d.append(temp_df)
        df = pd.concat(d)
        if ex_tp == 'PT':
            sorted_df = df.sort_values(['accuracy', 'pgexpl'], ascending=[False, True])
        if ex_tp == 'CF':
            sorted_df = df.sort_values(['sparsity', 'accuracy'], ascending=[True, False])
        if ex_tp == 'EXE':
            sorted_df = df.sort_values(['sparsity', 'accuracy'], ascending=[False, False])

