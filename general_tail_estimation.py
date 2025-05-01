import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import scipy.stats as stats
import seaborn as sns

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

def plot_histogram(df, season, grid_name, save_path):

    plt.figure(figsize=(10, 6))
    plt.hist(
        df["Value"],
        bins=1000,
        color="blue",
        edgecolor="black",
    )
    plt.title(f"Histogram ({season} - {grid_name})")
    plt.xlabel("Frequency")
    plt.ylabel("Frequency Count")
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "histogram.png"))
    plt.show()

def plot_QQ_plot(df, season, grid_name, save_path):

    stats.probplot(df["Value"], dist="norm", plot=plt)
    plt.title(f"Q-Q Plot ({season} - {grid_name})")
    plt.savefig(os.path.join(save_path, "qq_plot.png"))
    plt.show()

def plot_countour(df, thresholds, direction, season, grid_name, save_path):
    alphas = np.arange(0.0001, 10, 0.005)
    betas = np.linspace(0.1, 3, 40)

    # Prepare a meshgrid for plotting
    A, B = np.meshgrid(alphas, betas)
    LL = np.zeros_like(A)

    def _log_likelihood(alpha, beta, data, x_k):
        k = len(data)
        term1 = k * (np.log(alpha) + np.log(beta) + alpha * x_k ** beta)
        term2 = (beta - 1) * np.sum(np.log(data))
        term3 = alpha * np.sum(data ** beta)
        return term1 + term2 - term3

    for threshold in thresholds:
        if direction == 'right':  # from threshold to right [x_th, inf)
            tail_data = df[df['Value'] >= threshold]['Value']
            x_data = 50 + (tail_data - 50.00).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = threshold
        elif direction == 'left':  # from left to threshold (-inf, x_th]
            tail_data = df[df['Value'] <= threshold]['Value']
            tail_data_mirrored = 50 + (50 - tail_data)
            x_data = 50 + (tail_data_mirrored - 50).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = 50 + (50 - threshold)
        else:
            raise ValueError("Direction must be 'right' or 'left'.")

        # Calculate log-likelihood over the grid
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                LL[i, j] = _log_likelihood(A[i, j], B[i, j], x_data, x_k)

        # Print maximum log-likelihood
        max_ll = np.max(LL)
        max_alpha = A[np.unravel_index(np.argmax(LL), LL.shape)]
        max_beta = B[np.unravel_index(np.argmax(LL), LL.shape)]
        print(f"Threshold: {threshold:.2f}, Max LL: {max_ll}, Alpha: {max_alpha}, Beta: {max_beta}")

        # Plot the surface as a contour plot
        fig = go.Figure(
            data=[go.Surface(z=LL, x=alphas, y=betas, colorscale="Viridis", opacity=0.5)])
        fig.update_layout(
            title=f'{direction} tail Log-Likelihood Surface ({season} - {grid_name})',
            scene=dict(xaxis_title='Alpha',
                       yaxis_title='Beta',
                       zaxis_title='LL(alpha, beta)'),
            autosize=True
            # width=500,
            # height=500,
            # margin=dict(l=65, r=50, b=65, t=90),
            # scene_aspectmode='cube'
        )
        # save interactive figure
        fig.write_html(os.path.join(save_path, f"contour_{threshold:.2f}.html"))
        fig.show()



def plot_likelihood_fixed_beta(df, thresholds, beta, direction, season, grid_name, save_path):
    # Log-likelihood function
    def _log_likelihood(alpha, beta, data, x_k):
        k = len(data)
        term1 = k * (np.log(alpha) + np.log(beta) + alpha * x_k ** beta)
        term2 = (beta - 1) * np.sum(np.log(data))
        term3 = alpha * np.sum(data ** beta)
        return term1 + term2 - term3

    for threshold in thresholds:
        if direction == 'right':  # from threshold to right [x_th, inf)
            tail_data = df[df['Value'] >= threshold]['Value']
            x_data = 50 + (tail_data - 50.00).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = threshold
        elif direction == 'left':  # from left to threshold (-inf, x_th]
            tail_data = df[df['Value'] <= threshold]['Value']
            tail_data_mirrored = 50 + (50 - tail_data)
            x_data = 50 + (tail_data_mirrored - 50).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = 50 + (50 - threshold)
        else:
            raise ValueError("Direction must be 'right' or 'left'.")

        beta_variants = [beta, beta - 0.2, beta - 0.1, beta + 0.1, beta + 0.2]

        # Given beta, plot for different alpha
        alpha_values = np.linspace(0.1, 10, 200)
        alpha_values = np.arange(0.0001, 0.1, 0.0001)
        # Plot for each beta variant
        plt.figure(figsize=(10, 6))
        for b in sorted(beta_variants):
            ll_values = [_log_likelihood(alpha, b, x_data, x_k) for alpha in alpha_values]
            plt.plot(alpha_values, ll_values, label=f'Beta={b:.2f}')

        plt.title(f'[base beta={beta}, x_th={threshold}] {direction} tail Log-Likelihood vs Alpha ({season} - {grid_name})')
        plt.xlabel('Alpha')
        plt.xscale('log')
        # plt.ylim(0, 1e6)
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, f"fixed_beta_{beta}_x_th_{threshold}_likelihood.png"))
        plt.show()

def plot_likelihood_fixed_alpha(df, thresholds, alpha, direction, season, grid_name, save_path):
    # Log-likelihood function
    def _log_likelihood(alpha, beta, data, x_k):
        k = len(data)
        term1 = k * (np.log(alpha) + np.log(beta) + alpha * x_k ** beta)
        term2 = (beta - 1) * np.sum(np.log(data))
        term3 = alpha * np.sum(data ** beta)
        return term1 + term2 - term3

    for threshold in thresholds:
        if direction == 'right':  # from threshold to right [x_th, inf)
            tail_data = df[df['Value'] >= threshold]['Value']
            x_data = 50 + (tail_data - 50.00).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = threshold
        elif direction == 'left':  # from left to threshold (-inf, x_th]
            tail_data = df[df['Value'] <= threshold]['Value']
            tail_data_mirrored = 50 + (50 - tail_data)
            x_data = 50 + (tail_data_mirrored - 50).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = 50 + (50 - threshold)
        else:
            raise ValueError("Direction must be 'right' or 'left'.")

        alpha_variants = [alpha, alpha - 0.2, alpha - 0.1, alpha + 0.1, alpha + 0.2]

        # Given beta, plot for different alpha
        beta_values = np.linspace(0.1, 3, 40)
        # Plot for each beta variant
        plt.figure(figsize=(10, 6))
        for a in sorted(alpha_variants):
            ll_values = [_log_likelihood(a, beta, x_data, x_k) for beta in beta_values]
            plt.plot(beta_values, ll_values, label=f'Alpha={a:.2f}')

        plt.title(f'[base alpha={alpha}, x_th={threshold}] {direction} tail Log-Likelihood vs Beta ({season} - {grid_name})')
        plt.xlabel('Beta')
        # plt.xlim(1, 2.5)
        # plt.ylim(0, 1e6)
        plt.ylabel('Log-Likelihood')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(save_path, f"fixed_alpha_{alpha}_x_th_{threshold}_likelihood.png"))
        plt.show()


def print_stats(df):
    def calculate_and_print_percentage(df_, lower_bound, upper_bound, description):
        percentage = ((df_["Value"] > lower_bound) & (df_["Value"] <= upper_bound)).mean() * 100
        print(f"% of entries {description}: {percentage}%")

    frequency_ranges = [
        (float("-inf"), 49.9, "below 49.9"),
        (50.1, float("inf"), "above 50.1"),
        (float("-inf"), 49.8, "below 49.8"),
        (50.2, float("inf"), "above 50.2"),
        (float("-inf"), 49.7, "below 49.7"),
        (50.3, float("inf"), "above 50.3"),
    ]

    for lower_bound, upper_bound, description in frequency_ranges:
        calculate_and_print_percentage(
            df, lower_bound, upper_bound, description
        )

# def censored_log_likelihood_old(beta, alpha, xi):
# 	if beta <= 0:
# 		return np.inf
# 	k = len(xi)
# 	term1 = k / beta
# 	term2 = alpha * k * xi[k-1] ** beta * np.log(xi[k-1])
# 	term3 = np.sum(np.log(xi))
# 	term4 = alpha * np.sum(xi ** beta * np.log(xi))
# 	return term1 + term2 + term3 - term4

# def censored_log_likelihood(beta, alpha, xi, xk):
# 	k = len(xi)
# 	term1 = k*(np.log(alpha) + np.log(beta) + alpha*xi[k-1]**beta)
# 	term2 = (beta - 1)*np.sum(np.log(xi))
# 	term3 = alpha*np.sum(xi**beta)
# 	return term1 + term2 - term3

def log_likelihood(beta, alpha, data, x_k):
    k = len(data)
    term1 = k * (np.log(alpha) + np.log(beta) + alpha * x_k ** beta)
    term2 = (beta - 1) * np.sum(np.log(data))
    term3 = alpha * np.sum(data ** beta)
    return term1 + term2 - term3

def negative_log_likelihood(beta, alpha, data, x_k):
    return -log_likelihood(beta, alpha, data, x_k)

def optimize_thresholds_heatmap(df, thresholds, alpha_range, direction, grid_name, save_path):
    beta_bounds = (1e-6, 3.0)
    results = []

    for threshold in thresholds:
        if direction == 'right': # from threshold to right [x_th, inf)
            tail_data = df[df['Value'] >= threshold]['Value']
            x_data = 50 + (tail_data - 50.00).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = threshold
        elif direction == 'left': # from left to threshold (-inf, x_th]
            tail_data = df[df['Value'] <= threshold]['Value']
            tail_data_mirrored = 50 + (50 - tail_data)
            x_data = 50 + (tail_data_mirrored - 50).sort_values(ascending=True).values
            if len(x_data) > 0:
                x_k = min(x_data)
            else:
                x_k = 50 + (50 - threshold)
        else:
            raise ValueError("Direction must be 'right' or 'left'.")

        # x_data = tail_data.sort_values(ascending=True).values
        # print(f"Threshold {threshold:.2f} → {len(x_data)} tail points")
        if len(x_data) < 20 or np.any(x_data <= 0):
            continue

        for alpha in alpha_range:
            # beta_values = np.linspace(1e-6, 3, 300)
            # ll_values = [log_likelihood(beta, alpha, x_data, x_k) for beta in beta_values]
            # plt.plot(beta_values, ll_values)
            # plt.title(f"Log-Likelihood vs Beta for alpha={alpha}")
            result = minimize_scalar(
                negative_log_likelihood,
                bounds=beta_bounds,
                args=(alpha, x_data, x_k),
                method='bounded'
            )
            if result.success:
                beta = result.x
                nll = result.fun
                results.append({
                    'threshold': threshold,
                    'alpha': alpha,
                    'beta': beta,
                    'log_likelihood': round(-nll, 4)
                })

    if results:
        df_results = pd.DataFrame(results)
        # Save results to CSV
        df_results.to_csv(os.path.join(save_path, f'mle_{grid_name}_{direction}.csv'), index=False)

        # For each threshold, save the row with alpha and beta that maximizes the log-likelihood
        df_max = df_results.loc[df_results.groupby('threshold')['log_likelihood'].idxmax()]
        df_max.to_csv(os.path.join(save_path, f'mle_max_{grid_name}_{direction}.csv'), index=False)

        # Heatmap: Negative Log-Likelihood
        pivot_ll = df_results.pivot(index='alpha', columns='threshold', values='log_likelihood')

        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(pivot_ll, annot=False, fmt=".3e", cmap="viridis", cbar_kws={'label': 'Log-Likelihood'})
        plt.title(f'Log-Likelihood Heatmap ({direction.title()} Tail) — {grid_name.title()}')
        plt.xlabel('Threshold')
        plt.ylabel('Alpha')
        ax.set_yticklabels([f"{float(label.get_text()):.5f}" for label in ax.get_yticklabels()])
        ax.set_xticklabels([f"{float(label.get_text()):.5f}" for label in ax.get_xticklabels()])

        # Draw boxes around column-wise maxima
        for col_idx, col in enumerate(pivot_ll.columns):
            max_row_idx = pivot_ll[col].idxmax()
            row_idx = pivot_ll.index.get_loc(max_row_idx)
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, fill=False, edgecolor='red', linewidth=2))

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'heatmap_LL_{grid_name}_{direction}.png'))
        # plt.show()

        # Heatmap: Estimated Beta
        pivot_beta = df_results.pivot(index='alpha', columns='threshold', values='beta')

        plt.figure(figsize=(12, 6))
        ax = sns.heatmap(pivot_beta, annot=False, fmt=".3f", cmap="magma", cbar_kws={'label': 'Estimated Beta'})
        plt.title(f'Estimated Beta Heatmap ({direction.title()} Tail) — {grid_name.title()}')
        plt.xlabel('Threshold')
        ax.set_yticklabels([f"{float(label.get_text()):.5f}" for label in ax.get_yticklabels()])
        ax.set_xticklabels([f"{float(label.get_text()):.5f}" for label in ax.get_xticklabels()])
        plt.ylabel('Alpha')
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'heatmap_beta_{grid_name}_{direction}.png'))
        # plt.show()

    return pd.DataFrame(results)


def grid_search_thresholds(df, thresholds, alpha_range, beta_range, direction, grid_name, save_path):
    results = []

    for threshold in thresholds:
        if direction == 'right':  # from threshold to right [x_th, inf)
            tail_data = df[df['Value'] >= threshold]['Value']
            x_data = 50 + (tail_data - 50.00).sort_values(ascending=True).values
            x_k = threshold
        elif direction == 'left':  # from left to threshold (-inf, x_th]
            tail_data = df[df['Value'] <= threshold]['Value']
            x_data = 50.00 - (50.00 - tail_data).sort_values(ascending=False).values
            x_k = threshold
        else:
            raise ValueError("Direction must be 'right' or 'left'.")

        # print(f"Threshold {threshold:.2f} → {len(x_data)} tail points")
        if len(x_data) < 20 or np.any(x_data <= 0):
            continue

        for alpha in alpha_range:
            for beta in beta_range:
                nll = negative_log_likelihood(beta, alpha, x_data, x_k)
                results.append({
                    'threshold': threshold,
                    'alpha': alpha,
                    'beta': round(beta, 4),
                    'log_likelihood': round(-nll, 4)
                })

    if results:
        df_results = pd.DataFrame(results)
        # Save results to CSV
        df_results.to_csv(os.path.join(save_path, f'grid_search_{grid_name}_{direction}.csv'), index=False)

        for threshold in thresholds:
            df_sub = df_results[df_results['threshold'] == round(threshold, 2)]
            if len(df_sub) == 0:
                continue
            pivot_ll = df_sub.pivot(index='alpha', columns='beta', values='log_likelihood')
            plt.figure(figsize=(10, 6))
            ax = sns.heatmap(pivot_ll, annot=False, fmt=".2e", cmap="viridis", cbar_kws={'label': 'Log-Likelihood'})
            plt.title(f'LL Heatmap (Alpha vs Beta) for Threshold={threshold:.2f} ({direction.title()} Tail) — {grid_name.title()}')
            plt.xlabel('Beta')
            plt.ylabel('Alpha')
            ax.set_yticklabels([f"{float(label.get_text()):.5f}" for label in ax.get_yticklabels()])
            ax.set_xticklabels([f"{float(label.get_text()):.5f}" for label in ax.get_xticklabels()])
            # Draw box around maximum
            max_idx = np.unravel_index(np.nanargmax(pivot_ll.values), pivot_ll.shape)
            alpha_val = pivot_ll.index[max_idx[0]]
            beta_val = pivot_ll.columns[max_idx[1]]
            ax.add_patch(plt.Rectangle((max_idx[1], max_idx[0]), 1, 1, fill=False, edgecolor='red', linewidth=2))

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f'heatmap_grid_LL_{grid_name}_{direction}_{threshold:.2f}.png'))
            plt.show()

    return pd.DataFrame(results)



seasons = ['spring', 'summer', 'autumn', 'winter']
grid_names = ['european', 'nordic', 'uk']
directions = ['right', 'left']
for season in seasons:
    for grid_name in grid_names:
        for direction in directions:
            save_path = os.path.join(os.getcwd(), "results_new_decision", season, grid_name, direction)
            os.makedirs(save_path, exist_ok=True)

            print(f"Season: {season}, Grid: {grid_name}, Direction: {direction}")
            df = read_df(grid_name, season)
            alpha_range = np.arange(0.001, 10, 0.001) #[10**n for n in range(-10, 11)]
            # alpha_range = [(1/100)*(2**n) for n in range(-15, 5+1)] #np.linspace(0.01, 10, 20)
            if direction == 'right':
                if grid_name == 'european':
                    thresholds = np.linspace(50.05, 50.15, 5)
                else:
                    thresholds = np.linspace(50.10, 50.20, 5)
            elif direction == 'left':
                if grid_name == 'european':
                    thresholds = reversed(np.linspace(49.85, 49.95, 5))
                else:
                    thresholds = reversed(np.linspace(49.80, 49.90, 5))
            else:
                raise ValueError("Direction must be 'right' or 'left'.")

            # plot_countour(df, thresholds, direction, season, grid_name, save_path)
            # plot_likelihood_fixed_beta(df, thresholds, beta=1.5, season=season, grid_name=grid_name, save_path=save_path)
            # plot_likelihood_fixed_beta(df, thresholds, beta=0.71, direction=direction, season=season, grid_name=grid_name, save_path=save_path)
            # plot_likelihood_fixed_beta(df, thresholds, beta=2, direction=direction, season=season, grid_name=grid_name, save_path=save_path)
            # plot_likelihood_fixed_beta(df, thresholds, beta=2.5, direction=direction, season=season, grid_name=grid_name, save_path=save_path)
            # plot_likelihood_fixed_alpha(df, thresholds, alpha=999, direction=direction, season=season, grid_name=grid_name, save_path=save_path)
            # plot_likelihood_fixed_alpha(df, thresholds, alpha=0.0833, direction=direction, season=season, grid_name=grid_name, save_path=save_path)
            # plot_likelihood_fixed_alpha(df, thresholds, alpha=5.35, season=season, grid_name=grid_name, save_path=save_path)

            optimize_thresholds_heatmap(df, thresholds=thresholds, alpha_range=alpha_range, direction=direction, grid_name=grid_name, save_path=save_path)
            # beta_range = np.linspace(1.0, 3.0, 500)
            # print('\nGrid Search for Thresholds:')
            # grid_search_thresholds(df, thresholds=thresholds, alpha_range=alpha_range, beta_range=beta_range, direction=direction, grid_name=grid_name, save_path=save_path)

            # fit_levy_stable(df, thresholds, direction, grid_name, save_path)
            # fit_weibull(df, thresholds, direction, grid_name, save_path)

# for season in seasons:
# 	for grid_name in grid_names:
# 		save_path_eda = os.path.join(os.getcwd(), "results_new2", season, grid_name, "eda")
# 		os.makedirs(save_path_eda, exist_ok=True)
# 		df = read_df(grid_name, season)
# 		plot_histogram(df, season, grid_name, save_path=save_path_eda)
# 		plot_QQ_plot(df, season, grid_name, save_path=save_path_eda)
