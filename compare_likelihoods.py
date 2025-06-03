# Likelihood ratio test for different plots.
import os
import pandas as pd
import numpy as np

directory_custom_pot = 'results_new_data_2'
file_name_custom_pot = 'mle_max_european_left.csv'

directory_evt_pot = 'results_EVT_POT'
file_name_evt_pot = 'EVT_results_mle_european_left.csv'

directory_evt_bm = 'results_BM_2'
file_name_evt_bm = 'EVT_results_mle_european_right.csv'

directory_compare_custom_evt_pot = 'compare_custom_evt_pot'
os.makedirs(directory_compare_custom_evt_pot, exist_ok=True)


seasons = ['spring', 'summer', 'autumn', 'winter']
grid_names = ['european', 'nordic', 'uk']
directions = ['right', 'left']
for season in seasons:
    for grid_name in grid_names:
        for direction in directions:
            save_dir = os.path.join(directory_compare_custom_evt_pot, season, grid_name, direction)
            os.makedirs(save_dir, exist_ok=True)
            df_custom_pot = pd.read_csv(os.path.join(directory_custom_pot, season, grid_name, direction, f'mle_max_{grid_name}_{direction}.csv'))
            df_custom_pot['threshold'] = round(np.abs(50 - df_custom_pot['threshold']), 3)
            df_evt_pot = pd.read_csv(os.path.join(directory_evt_pot, season, grid_name, direction, f'EVT_results_mle_{grid_name}_{direction}.csv'))
            df_evt_pot['threshold'] = round(df_evt_pot['threshold'], 3)
            df_compare = (pd.merge(left=df_custom_pot, right=df_evt_pot, on='threshold', how='inner'))
            df_compare = df_compare.rename(columns={'log_likelihood_x': 'log_likelihood_custom', 'log_likelihood_y': 'log_likelihood_evt'})
            df_compare = df_compare[['threshold', 'log_likelihood_custom', 'log_likelihood_evt']]
            df_compare.to_csv(os.path.join(save_dir, f'compare_pot_{grid_name}_{direction}.csv'), index=False)
