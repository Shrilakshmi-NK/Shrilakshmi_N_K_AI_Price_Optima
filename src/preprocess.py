import pandas as pd
import os

# Paths (adjust if needed)
data_dir = 'data/'
output_dir = 'data/processed/'
os.makedirs(output_dir, exist_ok=True)  # Create output folder if not exists

# Step 1: Load all CSVs
print("Loading data...")
train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
stores = pd.read_csv(os.path.join(data_dir, 'stores.csv'))
oil = pd.read_csv(os.path.join(data_dir, 'oil.csv'))
holidays = pd.read_csv(os.path.join(data_dir, 'holidays_events.csv'))
test = pd.read_csv(os.path.join(data_dir, 'test.csv'))  # Optional, for later validation

# Step 2: Clean individual datasets
# Convert dates to datetime
train['date'] = pd.to_datetime(train['date'])
oil['date'] = pd.to_datetime(oil['date'])
holidays['date'] = pd.to_datetime(holidays['date'])
test['date'] = pd.to_datetime(test['date'])

# Handle missing values
# Oil: Forward-fill missing prices (simple imputation)
oil['dcoilwtico'].fillna(method='ffill', inplace=True)
oil['dcoilwtico'].fillna(method='bfill', inplace=True)  # Backup backward fill

# Holidays: Keep only national/regional (filter if too many; simple keep all)
holidays = holidays[holidays['transferred'] == False]  # Exclude transferred holidays

# Stores: No major cleaning needed

# Train: Handle zero/negative sales (clip to 0)
train['sales'] = train['sales'].clip(lower=0)

# Step 3: Merge datasets
# Merge train with stores (on store_nbr)
merged = pd.merge(train, stores, on='store_nbr', how='left')

# Merge with oil (on date)
merged = pd.merge(merged, oil, on='date', how='left')

# Merge with holidays (on date; add holiday flag)
holidays['is_holiday'] = 1
merged = pd.merge(merged, holidays[['date', 'is_holiday']], on='date', how='left')
merged['is_holiday'].fillna(0, inplace=True)  # 0 if no holiday

# For test data (optional: preprocess similarly for backtesting)
test_merged = pd.merge(test, stores, on='store_nbr', how='left')
test_merged = pd.merge(test_merged, oil, on='date', how='left')
test_merged = pd.merge(test_merged, holidays[['date', 'is_holiday']], on='date', how='left')
test_merged['is_holiday'].fillna(0, inplace=True)

# Step 4: Feature Engineering (simple calculations)
# Seasonality: Extract day, month, year, weekday
merged['year'] = merged['date'].dt.year
merged['month'] = merged['date'].dt.month
merged['day'] = merged['date'].dt.day
merged['weekday'] = merged['date'].dt.weekday  # 0=Monday, 6=Sunday

# Demand elasticity proxy: Sales sensitivity to promotion (avoid divide by zero)
merged['elasticity_proxy'] = merged['sales'] / merged['onpromotion'].replace(0, 1)  # Simple ratio

# Competitor price index proxy: Use oil price as economic proxy (or avg sales across stores if needed)
# For simplicity, use oil price directly; add a group avg if wanted
merged['comp_price_proxy'] = merged['dcoilwtico']

# Inventory health proxy: Cumulative sales per store/family (higher = healthier turnover)
merged['cum_sales'] = merged.groupby(['store_nbr', 'family'])['sales'].cumsum()

# Aggregate: Example - monthly sales sum (for EDA later)
monthly_agg = merged.groupby(['year', 'month', 'store_nbr', 'family'])['sales'].sum().reset_index()

# Step 5: Save processed data
merged.to_csv(os.path.join(output_dir, 'train_processed.csv'), index=False)
test_merged.to_csv(os.path.join(output_dir, 'test_processed.csv'), index=False)
monthly_agg.to_csv(os.path.join(output_dir, 'monthly_agg.csv'), index=False)

print("Preprocessing complete! Files saved in data/processed/")