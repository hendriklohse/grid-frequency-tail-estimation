import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from pyextremes import EVA

def read_df(grid_name, season):

	if season == 'spring':
		months = ['03', '04', '05']
	elif season == 'summer':
		months = ['06', '07', '08']
	elif season == 'autumn':
		months = ['09', '10', '11']
	elif season == 'winter':
		months = ['12', '01', '02']
	else:
		raise ValueError("Invalid season. Choose from 'spring', 'summer', 'autumn', or 'winter'.")

	if grid_name == 'european':
		df_frequency_merged = pd.DataFrame()
		for month_str in months:
			folder_path = os.path.join(os.getcwd(), "data", "european", "netztransparenz")
			csv_files = [file for file in os.listdir(folder_path) if file.endswith('csv')]
			for file in csv_files:
				if file.split('_')[1][4:6] == month_str:
					file_path = os.path.join(folder_path, file)
					df_frequency = pd.read_csv(file_path, sep=';')
					df_frequency.columns = ['Date', 'Time', 'Value']
					df_frequency['Datetime'] = pd.to_datetime(df_frequency['Date'] + ' ' + df_frequency['Time'], dayfirst=True)
					df_frequency['Value'] = df_frequency['Value'].astype(str).str.replace(',', '.').astype(float)
					df_frequency.set_index('Datetime', inplace=True)
					df_frequency.drop(columns=['Date', 'Time'], inplace=True)
					df_frequency_merged = pd.concat([df_frequency_merged, df_frequency])

		# Sort the merged DataFrame by index
		df_frequency_merged.sort_index(inplace=True)
		df = df_frequency_merged
	elif grid_name == 'nordic':
		df_frequency_merged = pd.DataFrame()

		for month_str in months:
			folder_path = os.path.join(os.getcwd(), "data", "nordic", f"2024-{month_str}")
			csv_files = [file for file in os.listdir(folder_path) if file.endswith('csv')]
			for file in csv_files:
				file_path = os.path.join(folder_path, file)
				df_frequency = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
				if len(df_frequency[df_frequency['Value'] < 40.0]) > 0:
					print(f"File {file} contains values below 40.0")
					df_frequency = df_frequency[df_frequency['Value'] >= 40.0]
				# df_frequency['Value'] = df_frequency['Value'].astype(float)
				df_frequency_merged = pd.concat([df_frequency_merged, df_frequency])

		# Sort the merged DataFrame by index
		df_frequency_merged.sort_index(inplace=True)

		# Subsample to have second instead of millisecond, not use mean
		samples = [df_frequency_merged] + [df_frequency_merged.iloc[i::10] for i in range(10)]
		df = samples[1]
	elif grid_name == 'uk':
		df_frequency_merged = pd.DataFrame()
		for month_str in months:
			folder_path = os.path.join(os.getcwd(), "data", "uk")
			csv_files = [f"fnew-2024-{int(month_str)}.csv"]

			for file in csv_files:
				file_path = os.path.join(folder_path, file)
				df_frequency = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
				if len(df_frequency[df_frequency['f'] < 40.0]) > 0:
					print(f"File {file} contains values below 40.0")
					df_frequency = df_frequency[df_frequency['Value'] >= 40.0]
				# df_frequency['Value'] = df_frequency['Value'].astype(float)
				df_frequency_merged = pd.concat([df_frequency_merged, df_frequency])

		# Rename
		df_frequency_merged.rename(columns={"f": "Value"}, inplace=True)
		df_frequency_merged.index.name = "Time"

		# Sort the merged DataFrame by index
		df_frequency_merged.sort_index(inplace=True)
		df = df_frequency_merged
	else:
		raise ValueError("Invalid grid name. Choose from 'european', 'nordic', or 'uk'.")
	return df

grid_name = 'nordic'
season = 'spring'
df = read_df(grid_name, season)
series = df['Value']

model = EVA(series)

model.get_extremes(method="BM", block_size="365.2425D")
model.plot_extremes()
