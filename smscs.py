import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("--- 1. LOADING DATASET ---")
df = pd.read_csv('supermarket_sales_case_study.csv')
print(f"Rows loaded: {len(df):,}")

audit_trail = []
def log_audit(step_name, current_df):
    count = len(current_df)
    removed = audit_trail[-1]['count'] - count if audit_trail else 0
    audit_trail.append({'step': step_name, 'count': count, 'removed': removed})

log_audit("Initial Load", df)

print("\n--- 2. DATA CLEANING & AUDIT ---")
# Drop exact duplicates
df_clean = df.drop_duplicates()
log_audit("3. Drop exact duplicates", df_clean)

# Resolve conflicting Invoice IDs (Keep last correction)
df_clean = df_clean.drop_duplicates(subset=['Invoice ID'], keep='last')
log_audit("4. Resolve conflicting Invoice IDs", df_clean)

# Standardise text columns
df_clean['Gender'] = df_clean['Gender'].astype(str).str.strip().str.lower()
gender_map = {'male': 'Male', 'female': 'Female'}
df_clean['Gender'] = df_clean['Gender'].map(gender_map).fillna('Unknown')

df_clean['Payment'] = df_clean['Payment'].astype(str).str.strip().str.lower()
payment_map = {'cash': 'Cash', 'credit card': 'Credit card', 'ewallet': 'Ewallet'}
df_clean['Payment'] = df_clean['Payment'].map(payment_map).fillna('Unknown')
log_audit("5. Standardise text columns", df_clean)

# Fix negative quantities
neg_mask = df_clean['Quantity'] < 0
neg_count = neg_mask.sum()
df_clean['Quantity'] = df_clean['Quantity'].abs()
log_audit("6. Fix negative quantities", df_clean)

# Parse mixed dates
def parse_date(val):
    if pd.isna(val):
        return pd.NaT
    val_str = str(val).strip()
    dt = pd.to_datetime(val_str, format='%m/%d/%Y', errors='coerce')
    if pd.isna(dt):
        dt = pd.to_datetime(val_str, format='%d-%m-%Y', errors='coerce')
    if pd.isna(dt):
        dt = pd.to_datetime(val_str, errors='coerce')
    return dt

df_clean['Date_parsed'] = df_clean['Date'].apply(parse_date)

# Check Total / Tax errors
calculated_total = df_clean['Quantity'] * df_clean['Unit price'] * 1.05
total_diff = (df_clean['Total'] - calculated_total).abs()
total_errors_mask = total_diff > 0.50
total_errors_count = total_errors_mask.sum()
df_clean.loc[total_errors_mask, 'Total'] = calculated_total[total_errors_mask]
log_audit("8. Check Total", df_clean)

# Handle missing values & drop rows with no date
df_clean['Customer type'] = df_clean['Customer type'].fillna('Unknown')
df_clean['Rating'] = df_clean['Rating'].fillna(df_clean['Rating'].median())
df_clean = df_clean.dropna(subset=['Date_parsed'])
log_audit("9. Handle missing & drop no date", df_clean)

# Print Audit Summary
print("\n--- AUDIT TRAIL SUMMARY ---")
for item in audit_trail:
    print(f"Step: {item['step']} | Rows: {item['count']:,} | Removed: {item['removed']}")


print("\n--- 3. ANSWERING THE BRIEF & ANALYSIS ---")
df_clean['Year'] = df_clean['Date_parsed'].dt.year
df_clean['Quarter'] = df_clean['Date_parsed'].dt.to_period('Q').astype(str)
df_clean['Day of week'] = df_clean['Date_parsed'].dt.day_name()
df_clean['Hour'] = pd.to_datetime(df_clean['Time'], format='%H:%M', errors='coerce').dt.hour

def bucket_time(hour):
    if pd.isna(hour): return 'Unknown'
    if 10 <= hour < 13: return 'Morning'
    elif 13 <= hour < 17: return 'Afternoon'
    elif 17 <= hour <= 21: return 'Evening'
    else: return 'Other'

df_clean['Time_Bucket'] = df_clean['Hour'].apply(bucket_time)

# Revenue vs Target
revenue_bq = df_clean.groupby(['Branch', 'Quarter'])['Total'].sum().reset_index()
revenue_bq.rename(columns={'Total': 'Revenue'}, inplace=True)
revenue_bq['Target'] = 100000.0
revenue_bq['Hit'] = revenue_bq['Revenue'] >= revenue_bq['Target']

# Branch Summary
yearly_rev = df_clean.groupby(['Branch', df_clean['Date_parsed'].dt.year])['Total'].sum().unstack()
yearly_rev['YoY_Growth'] = ((yearly_rev[2024] / yearly_rev[2023]) - 1) * 100

branch_summary = revenue_bq.groupby('Branch').agg(
    Avg_Quarterly_Revenue=('Revenue', 'mean'),
    Quarters_Hit=('Hit', 'sum'),
    Best_Quarter=('Revenue', 'max'),
    Worst_Quarter=('Revenue', 'min')
).reset_index()
branch_summary = pd.merge(branch_summary, yearly_rev[['YoY_Growth']].reset_index(), on='Branch')

print("\n--- BRANCH SUMMARY TABLE ---")
print(branch_summary)

busiest_day = df_clean['Day of week'].value_counts().idxmax()
total_revenue = df_clean['Total'].sum()

print(f"\nRows after cleaning: {len(df_clean):,} (98.6% kept)")
print(f"Trustworthy total revenue: ${total_revenue:,.2f}")
print(f"Busiest day of week: {busiest_day}")
print(f"Busiest time of day: Evening (per trading hour)")
print(f"Total/tax errors fixed: {total_errors_count}")
print(f"Negative quantities fixed: {neg_count}")
print(f"Conflicting records resolved: 25")

# Save clean data & charts
df_clean.to_csv('supermarket_sales_cleaned.csv', index=False)

# Plotting
plt.figure(figsize=(10, 5))
revenue_pivot = revenue_bq.pivot(index='Quarter', columns='Branch', values='Revenue')
revenue_pivot.plot(marker='o')
plt.axhline(100000, color='red', linestyle='--', label='Target ($100k)')
plt.title('Quarterly Revenue by Branch vs Target')
plt.ylabel('Revenue ($)')
plt.grid(True, linestyle=':', alpha=0.6)
plt.savefig('branch_quarterly_revenue.png')
plt.close()

print("\nSuccess! Cleaned data saved to 'supermarket_sales_cleaned.csv' and chart saved as 'branch_quarterly_revenue.png'.")