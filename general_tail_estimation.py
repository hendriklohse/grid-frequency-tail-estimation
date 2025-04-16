import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import seaborn as sns

def read_df(grid_name):
	if grid_name == 'european':
		folder_path = os.path.join(os.getcwd(), "data", "european", "netztransparenz")
		file_path = os.path.join(folder_path, "Frequenz_20240801_20240831.csv")
		df_frequency = pd.read_csv(file_path, sep=';')

		df_frequency.columns = ['Date', 'Time', 'Value']
		df_frequency['Datetime'] = pd.to_datetime(df_frequency['Date'] + ' ' + df_frequency['Time'], dayfirst=True)
		df_frequency['Value'] = df_frequency['Value'].astype(str).str.replace(',', '.').astype(float)
		df_frequency.set_index('Datetime', inplace=True)
		df_frequency.drop(columns=['Date', 'Time'], inplace=True)
		df = df_frequency
	elif grid_name == 'nordic':
		folder_path = os.path.join(os.getcwd(), "data", "nordic", "2024-08")
		csv_files = [file for file in os.listdir(folder_path) if file.endswith('csv')]

		df_frequency_merged = pd.DataFrame()

		for file in csv_files:
			file_path = os.path.join(folder_path, file)
			df_frequency = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
			if len(df_frequency[df_frequency['Value'] < 40.0]) > 0:
				print(f"File {file} contains values below 40.0")
			# df_frequency['Value'] = df_frequency['Value'].astype(float)
			df_frequency_merged = pd.concat([df_frequency_merged, df_frequency])

		# Sort the merged DataFrame by index
		df_frequency_merged.sort_index(inplace=True)

		# Subsample to have second instead of millisecond, not use mean
		samples = [df_frequency_merged] + [df_frequency_merged.iloc[i::10] for i in range(10)]
		df = samples[1]
	elif grid_name == 'uk':
		folder_path = os.path.join(os.getcwd(), "data", "uk")
		csv_files = ["fnew-2024-8.csv"]

		df_frequency_merged = pd.DataFrame()

		for file in csv_files:
			file_path = os.path.join(folder_path, file)
			df_frequency = pd.read_csv(file_path, header=0, index_col=0, parse_dates=True)
			if len(df_frequency[df_frequency['f'] < 40.0]) > 0:
				print(f"File {file} contains values below 40.0")
			# df_frequency['Value'] = df_frequency['Value'].astype(float)
			df_frequency_merged = pd.concat([df_frequency_merged, df_frequency])

		# Rename
		df_frequency_merged.rename(columns={"f": "Value"}, inplace=True)
		df_frequency_merged.index.name = "Time"

		# Sort the merged DataFrame by index
		df_frequency_merged.sort_index(inplace=True)
		df = df_frequency_merged

	return df

def censored_log_likelihood(beta, alpha, xi):
	if beta <= 0:
		return np.inf
	k = len(xi)
	term1 = k / beta
	term2 = alpha * k * xi[k-1] ** beta * np.log(xi[k-1])
	term3 = np.sum(np.log(xi))
	term4 = alpha * np.sum(xi ** beta * np.log(xi))
	return term1 + term2 + term3 - term4

def optimize_thresholds_heatmap(df, thresholds, alpha_range, grid_name, direction='right'):
	beta_bounds = (1e-6, 10.0)
	results = []

	for threshold in thresholds:
		if direction == 'right':
			tail_data = df[df['Value'] >= threshold]['Value']
			x_data = 50.00 + (tail_data - 50.00).sort_values(ascending=True).values
		elif direction == 'left':
			tail_data = df[df['Value'] <= threshold]['Value']
			x_data = 50.00 - (50.00 - tail_data).sort_values(ascending=True).values
		else:
			raise ValueError("Direction must be 'right' or 'left'.")

		# x_data = tail_data.sort_values(ascending=True).values
		print(f"Threshold {threshold:.2f} → {len(x_data)} tail points")
		if len(x_data) == 0 or np.any(x_data <= 0):
			continue

		for alpha in alpha_range:
			result = minimize_scalar(
				censored_log_likelihood,
				bounds=beta_bounds,
				args=(alpha, x_data),
				method='bounded'
			)
			if result.success:
				beta = result.x
				nll = result.fun
				results.append({
					'threshold': round(threshold, 2),
					'alpha': round(alpha, 2),
					'beta': beta,
					'neg_log_likelihood': round(nll, 4)
				})

	if results:
		df_results = pd.DataFrame(results)

		# Heatmap: Negative Log-Likelihood
		pivot_ll = df_results.pivot(index='alpha', columns='threshold', values='neg_log_likelihood')
		plt.figure(figsize=(12, 6))
		sns.heatmap(pivot_ll, annot=True, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Negative Log-Likelihood'})
		plt.title(f'Negative Log-Likelihood Heatmap ({direction.title()} Tail) — {grid_name.title()}')
		plt.xlabel('Threshold')
		plt.ylabel('Alpha')
		plt.tight_layout()
		plt.show()

		# Heatmap: Estimated Beta
		pivot_beta = df_results.pivot(index='alpha', columns='threshold', values='beta')
		plt.figure(figsize=(12, 6))
		sns.heatmap(pivot_beta, annot=True, fmt=".4f", cmap="magma", cbar_kws={'label': 'Estimated Beta'})
		plt.title(f'Estimated Beta Heatmap ({direction.title()} Tail) — {grid_name.title()}')
		plt.xlabel('Threshold')
		plt.ylabel('Alpha')
		plt.tight_layout()
		plt.show()

	return pd.DataFrame(results)

def grid_search_thresholds(df, thresholds, alpha_range, beta_range, grid_name, direction='right'):
	results = []

	for threshold in thresholds:
		if direction == 'right':
			tail_data = df[df['Value'] >= threshold]['Value']
			x_data = 50.00 + (tail_data - 50.00).sort_values(ascending=True).values
		elif direction == 'left':
			tail_data = df[df['Value'] <= threshold]['Value']
			x_data = 50.00 - (50.00 - tail_data).sort_values(ascending=True).values
		else:
			raise ValueError("Direction must be 'right' or 'left'.")

		print(f"Threshold {threshold:.2f} → {len(x_data)} tail points")
		if len(x_data) == 0 or np.any(x_data <= 0):
			continue

		for alpha in alpha_range:
			for beta in beta_range:
				nll = censored_log_likelihood(beta, alpha, x_data)
				results.append({
					'threshold': round(threshold, 2),
					'alpha': round(alpha, 2),
					'beta': round(beta, 4),
					'neg_log_likelihood': round(nll, 4)
				})

	if results:
		df_results = pd.DataFrame(results)

		for threshold in thresholds:
			df_sub = df_results[df_results['threshold'] == round(threshold, 2)]
			if len(df_sub) == 0:
				continue
			pivot_ll = df_sub.pivot(index='alpha', columns='beta', values='neg_log_likelihood')
			plt.figure(figsize=(10, 6))
			ax = sns.heatmap(pivot_ll, annot=False, fmt=".4f", cmap="viridis", cbar_kws={'label': 'Negative Log-Likelihood'})
			plt.title(f'LL Heatmap (Alpha vs Beta) for Threshold={threshold:.2f} ({direction.title()} Tail) — {grid_name.title()}')
			plt.xlabel('Beta')
			plt.ylabel('Alpha')
			# Draw box around minimum
			min_idx = np.unravel_index(np.nanargmin(pivot_ll.values), pivot_ll.shape)
			alpha_val = pivot_ll.index[min_idx[0]]
			beta_val = pivot_ll.columns[min_idx[1]]
			ax.add_patch(plt.Rectangle((min_idx[1], min_idx[0]), 1, 1, fill=False, edgecolor='red', linewidth=2))

			plt.tight_layout()
			plt.show()

	return pd.DataFrame(results)


# Example usage:
grid_name = 'nordic'  # 'european', 'nordic', or 'uk'
direction = 'right'
df = read_df(grid_name)
alpha_range = np.linspace(0.01, 4.00, 50)
if grid_name == 'european':
	thresholds = np.linspace(50.05, 50.15, 10)
else:
	thresholds = np.linspace(50.10, 50.20, 10)
print('Optimizing Thresholds:')
optimize_thresholds_heatmap(df, thresholds=thresholds, alpha_range=alpha_range, grid_name=grid_name, direction=direction)

# beta_range = np.linspace(1.0, 3.0, 30)
# print('\nGrid Search for Thresholds:')
# grid_search_thresholds(df, thresholds=thresholds, alpha_range=alpha_range, beta_range=beta_range, direction=direction)
