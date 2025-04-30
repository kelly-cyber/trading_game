import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
import datetime
from scipy.stats import yeojohnson

# Load combined_data.parquet
combined_data = pd.read_parquet("combined_data.parquet")
print(combined_data.shape)
# Print cols
print(combined_data.columns)

# Filter for only COFFEE C - ICE FUTURES U.S. data
coffee_data = combined_data[combined_data['Market_and_Exchange_Names'] == 'COFFEE C - ICE FUTURES U.S.']
# GOLD - COMMODITY EXCHANGE INC.
# coffee_data = combined_data[combined_data['Market_and_Exchange_Names'] == 'GOLD - COMMODITY EXCHANGE INC.']
# LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE
# coffee_data = combined_data[combined_data['Market_and_Exchange_Names'] == 'LIVE CATTLE - CHICAGO MERCANTILE EXCHANGE']
print("\nCOFFEE C - ICE FUTURES U.S. data shape:")
print(coffee_data.shape)
print("\nHead of COFFEE C - ICE FUTURES U.S. data:")
print(coffee_data.head())

# Convert 'Report_Date_as_MM_DD_YYYY' to datetime and sort by date
coffee_data['Report_Date'] = pd.to_datetime(coffee_data['Report_Date_as_MM_DD_YYYY'])
coffee_data = coffee_data.sort_values('Report_Date')
print("\nDate range in the data:")
print(f"Start: {coffee_data['Report_Date'].min()}, End: {coffee_data['Report_Date'].max()}")

# Download KC=F data from Yahoo Finance
print("\nDownloading coffee futures data from Yahoo Finance...")
kc_data = yf.download("KC=F", start=coffee_data['Report_Date'].min() - datetime.timedelta(days=30), 
                      end=coffee_data['Report_Date'].max() + datetime.timedelta(days=60))

# kc_data = yf.download("GC=F", start=coffee_data['Report_Date'].min() - datetime.timedelta(days=30), 
#                         end=coffee_data['Report_Date'].max() + datetime.timedelta(days=60))

# Download live cattle futures data from Yahoo Finance
# kc_data = yf.download("LE=F", start=coffee_data['Report_Date'].min() - datetime.timedelta(days=30), 
#                          end=coffee_data['Report_Date'].max() + datetime.timedelta(days=60))

# Flatten the MultiIndex columns
if isinstance(kc_data.columns, pd.MultiIndex):
    # Convert MultiIndex columns to single-level
    kc_data.columns = [col[0] for col in kc_data.columns]
    print("\nFlattened kc_data columns:")
    print(kc_data.columns)



# Calculate weekly returns instead of monthly
kc_data['weekly_return'] = kc_data['Close'].pct_change(periods=5)  # Approx 5 trading days in a week

# Create forward-looking weekly returns (next week's return)
kc_data['next_week_return'] = kc_data['weekly_return'].shift(-5)  # Next week's return

#plot the distribution of next_week_return
plt.figure(figsize=(10, 6))
plt.hist(kc_data['next_week_return'], bins=50, edgecolor='black')
plt.title('Distribution of Next Week Returns')
plt.xlabel('Next Week Return')
plt.ylabel('Frequency')
# plt.show()


coffee_data['Report_Date'] = pd.to_datetime(coffee_data['Report_Date_as_MM_DD_YYYY'])


kc_data.reset_index(inplace=True)
# Merge CFTC coffee data with price data
merged_data = pd.merge(
    coffee_data, 
    kc_data[['Date', 'next_week_return']],
    left_on='Report_Date',
    right_on='Date',
)

print("\nMerged data shape:")
print(merged_data.shape)
print("\nSample of merged data:")
print(merged_data[['Report_Date', 'Date', 'next_week_return']].head())


# Get all columns except metadata columns
excluded_prefixes = ['Market_and', 'As_of_Date', 'Report_Date', 'CFTC_Contr', 'CFTC_Marke', 'CFTC_Regio', 'CFTC_Comm', 'Contract_Units', 'CFTC_SubGroup_Code', 'FutOnly_or_Combined', 'source_file']
all_columns = merged_data.columns.tolist()
selected_features = []

for col in all_columns:
    if not any(col.startswith(prefix) for prefix in excluded_prefixes) and col != 'next_week_return' and col != 'Date':
        selected_features.append(col)


# Check which features are actually available in the dataset
available_features = [col for col in selected_features if col in merged_data.columns]
print(f"\nFound {len(available_features)} of the selected features in the dataset")
print(f"First 10 features: {available_features[:10]}...")

# Remove features that have nan values
features_with_nan = [col for col in available_features if merged_data[col].isna().any()]
clean_features = [col for col in available_features if col not in features_with_nan]
print(f"\nRemoved {len(features_with_nan)} features that contain NaN values")
print(f"Remaining {len(clean_features)} features after removing those with NaNs")
if len(features_with_nan) > 0:
    print(f"First few removed features: {features_with_nan[:5]}...")

# Convert all features to percent changes
feature_df = merged_data[clean_features].copy()
feature_df_pct = feature_df.pct_change()
print("\nConverting all features to percentage changes")
print(f"Shape before: {feature_df.shape}, Shape after: {feature_df_pct.shape}")

feature_df_pct = feature_df_pct.iloc[1:]

# Replace infinities with large but finite values
max_value = 9999
feature_df_pct = feature_df_pct.replace([np.inf], max_value)
feature_df_pct = feature_df_pct.replace([-np.inf], -max_value)

# Check which features contain NaN values
features_with_nan = [col for col in feature_df_pct.columns if feature_df_pct[col].isna().any()]

# Print information about the identified features
print(f"Found {len(features_with_nan)} features with NaN values")
if features_with_nan:
    print(f"First few features with NaNs: {features_with_nan[:5]}")

# Remove those features from the DataFrame
feature_df_pct = feature_df_pct.drop(columns=features_with_nan)
print(f"After removing features with NaNs, shape: {feature_df_pct.shape}")

# feature_df_pct = pd.DataFrame(StandardScaler().fit_transform(feature_df_pct), index=feature_df_pct.index)
# clip the data to -5 and 5
# feature_df_pct = feature_df_pct.clip(lower=-5, upper=5)

# Apply Yeo-Johnson transformation column by column
transformed_data = {}
print(f"Applying Yeo-Johnson transformation to {len(feature_df_pct.columns)} columns...")

for col in feature_df_pct.columns:
    try:
        # Transform each column individually
        transformed_data[col], _ = yeojohnson(feature_df_pct[col].values)
    except Exception as e:
        print(f"Error transforming column {col}: {e}")
        # Skip this column

# Convert the dictionary to a DataFrame with the same index as the original
feature_df_pct_transformed = pd.DataFrame(transformed_data, index=feature_df_pct.index)
print(f"Shape after transformation: {feature_df_pct_transformed.shape}")

# Replace original with transformed data
feature_df_pct = feature_df_pct_transformed

# Save the transformed data
feature_df_pct.to_csv('feature_df_pct.csv')


# # Apply RobustScaler after pct_change - RobustScaler scales each feature (column) independently
# scaler = RobustScaler()

# # Store column names before scaling
# feature_columns = feature_df_pct.columns

# # Apply scaling (this returns a numpy array) - scikit-learn scalers work column-wise by default
# feature_df_scaled = scaler.fit_transform(feature_df_pct)

# # Convert back to DataFrame with original column names
# feature_df_pct = pd.DataFrame(feature_df_scaled, columns=feature_columns, index=feature_df_pct.index)

# feature_df_pct.to_csv('feature_df_pct.csv')
# # Drop the first row which will have NaNs after pct_change()
# feature_df_pct = feature_df_pct.iloc[1:]

# Get corresponding target values
y = merged_data['next_week_return'].iloc[1:]

# Now check for NaN values in target (y)
nan_in_target = y.isna()
if nan_in_target.any():
    print(f"\nFound {nan_in_target.sum()} rows with NaNs in target")
    
    # Drop rows where target is NaN
    valid_rows = ~nan_in_target
    feature_df_pct = feature_df_pct.loc[valid_rows]
    y = y.loc[valid_rows]
    print(f"After removing rows with NaN targets: X shape: {feature_df_pct.shape}, y shape: {y.shape}")

# # Single-line alternative
# feature_df_pct = feature_df_pct.clip(lower=-5, upper=5)  # Constrain all values to [-5, 5]
# Now X and y should be clean and aligned
X = feature_df_pct
print(f"Final X shape: {X.shape}, y shape: {y.shape}")
feature_df_pct.to_csv('feature_df_pct.csv')

# Plot histograms for each feature in feature_df_pct
def plot_feature_histograms(df, cols_per_figure=16, save_path='feature_histograms/'):
    import os
    import math
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    total_cols = len(df.columns)
    total_figures = math.ceil(total_cols / cols_per_figure)
    
    print(f"Creating {total_figures} figures with {cols_per_figure} histograms each...")
    
    # Loop through columns in batches
    for fig_num in range(total_figures):
        # Calculate start and end indices for this figure
        start_idx = fig_num * cols_per_figure
        end_idx = min((fig_num + 1) * cols_per_figure, total_cols)
        
        # Get columns for this figure
        cols_to_plot = df.columns[start_idx:end_idx]
        
        # Calculate grid dimensions (approximately square)
        grid_size = math.ceil(math.sqrt(len(cols_to_plot)))
        rows = grid_size
        cols = math.ceil(len(cols_to_plot) / rows)
        
        # Create figure and subplots
        plt.figure(figsize=(cols * 4, rows * 3))
        
        # Plot each feature
        for i, col in enumerate(cols_to_plot):
            plt.subplot(rows, cols, i + 1)
            plt.hist(df[col].dropna(), bins=30, alpha=0.7)
            plt.title(col)
            plt.grid(True, alpha=0.3)
            
            # Add some statistics
            mean = df[col].mean()
            median = df[col].median()
            plt.axvline(mean, color='r', linestyle='--', alpha=0.5)
            plt.axvline(median, color='g', linestyle='-', alpha=0.5)
            
            # Add text with stats (limit precision to keep it readable)
            stats_text = f"mean: {mean:.2f}\nmedian: {median:.2f}"
            plt.annotate(stats_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                         va='top', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}features_hist_{fig_num+1}.png", dpi=150)
        print(f"Saved figure {fig_num+1}/{total_figures}")

# Run the plotting function
print("\nPlotting histograms for all features...")
plot_feature_histograms(feature_df_pct)
print("Feature histogram plotting complete!")

# To visualize the target variable distribution as well
plt.figure(figsize=(10, 6))
plt.hist(y, bins=30, alpha=0.7, color='blue')
plt.title('Distribution of Target Variable (next_week_return)')
plt.grid(True, alpha=0.3)
plt.axvline(y.mean(), color='r', linestyle='--', label=f'Mean: {y.mean():.4f}')
plt.axvline(y.median(), color='g', linestyle='-', label=f'Median: {y.median():.4f}')
plt.legend()
plt.savefig('target_distribution.png')
# plt.show()


# exit() 


# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Try many alpha values for Lasso regression
print("\nTesting multiple alpha values for Lasso regression...")
alphas = np.logspace(-5, 2, 100)  # 100 values from 0.00001 to 100
results = []


for alpha in alphas:
    # Train Lasso model with this alpha
    model = Lasso(alpha=alpha, max_iter=100000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(np.max(y_pred))
    test_mse = mean_squared_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    # Count non-zero coefficients (features used)
    n_features = np.sum(model.coef_ != 0)
    
    # Store results
    results.append({
        'alpha': alpha,
        'test_mse': test_mse,
        'test_r2': test_r2,
        'n_features': n_features
    })
    
    # Print progress for every 10th alpha
    if (list(alphas).index(alpha) + 1) % 10 == 0:
        print(f"Tested {list(alphas).index(alpha) + 1} alphas...")

# Convert results to DataFrame for analysis
results_df = pd.DataFrame(results)

# Find best alpha based on test R²
best_r2_idx = results_df['test_r2'].idxmax()
best_r2_alpha = results_df.loc[best_r2_idx, 'alpha']
best_r2 = results_df.loc[best_r2_idx, 'test_r2']
best_n_features = results_df.loc[best_r2_idx, 'n_features']

print(f"\nBest alpha based on test R²: {best_r2_alpha:.6f}")
print(f"Best R² score: {best_r2:.4f}")
print(f"Number of features with best R² alpha: {best_n_features}")

# Find alpha with reasonable R² and fewer features
threshold = best_r2 * 0.95  # 95% of best R²
parsimonious_model = results_df[(results_df['test_r2'] >= threshold) & 
                               (results_df['n_features'] < best_n_features)]

if not parsimonious_model.empty:
    parsimonious_alpha = parsimonious_model.loc[parsimonious_model['n_features'].idxmin(), 'alpha']
    parsimonious_n_features = parsimonious_model.loc[parsimonious_model['n_features'].idxmin(), 'n_features']
    parsimonious_r2 = parsimonious_model.loc[parsimonious_model['n_features'].idxmin(), 'test_r2']
    print(f"\nMore parsimonious model with alpha: {parsimonious_alpha:.6f}")
    print(f"R² score: {parsimonious_r2:.4f} ({parsimonious_r2/best_r2:.1%} of best)")
    print(f"Number of features: {parsimonious_n_features}")
    
    # Use this alpha for final model
    alpha = parsimonious_alpha
else:
    # Use the best R² alpha
    alpha = best_r2_alpha

# Plot alpha vs number of features and R²
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

ax1.semilogx(results_df['alpha'], results_df['n_features'])
ax1.set_xlabel('Alpha')
ax1.set_ylabel('Number of Features')
ax1.set_title('Alpha vs Number of Features')
ax1.axvline(x=alpha, color='r', linestyle='--')

ax2.semilogx(results_df['alpha'], results_df['test_r2'])
ax2.set_xlabel('Alpha')
ax2.set_ylabel('R² Score')
ax2.set_title('Alpha vs R² Score')
ax2.axvline(x=alpha, color='r', linestyle='--')

plt.tight_layout()
plt.savefig('alpha_selection.png')
plt.show()

# Train final model with selected alpha
print(f"\nTraining final Lasso model with alpha = {alpha:.6f}")
model = Lasso(alpha=alpha, max_iter=10000, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nFinal Model Performance:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R-squared: {r2:.4f}")

# Count non-zero coefficients
non_zero_coefs = np.sum(model.coef_ != 0)
print(f"Number of features used by the model: {non_zero_coefs} out of {len(clean_features)}")

# Feature importance for final model
feature_importance = pd.DataFrame({
    'Feature': X.columns,  # Use X.columns instead of clean_features
    'Coefficient': model.coef_
})
# Filter for non-zero coefficients and sort by absolute value
non_zero_features = feature_importance[feature_importance['Coefficient'] != 0]
non_zero_features = non_zero_features.sort_values(by='Coefficient', key=abs, ascending=False)
print("\nFeatures with non-zero coefficients (most important first):")
print(non_zero_features)

# Plot actual vs predicted returns
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.xlabel('Actual Returns')
plt.ylabel('Predicted Returns')
plt.title('Actual vs Predicted Next Week Returns')
plt.savefig('actual_vs_predicted.png')
plt.show()

exit()
