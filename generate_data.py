import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

print("Generating synthetic e-commerce transaction data...")

# Configuration
NUM_CUSTOMERS = 6000
START_DATE = datetime(2022, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Product categories and price ranges
CATEGORIES = {
    'Electronics': (50, 500),
    'Clothing': (20, 150),
    'Home & Garden': (15, 300),
    'Sports': (25, 200),
    'Books': (10, 50)
}

# Generate customer IDs
customer_ids = [f'CUST_{i:05d}' for i in range(1, NUM_CUSTOMERS + 1)]

# Create customer segments to ensure Pareto distribution
# Top 20% (1200 customers) generates ~65% of revenue
num_champions  = int(NUM_CUSTOMERS * 0.20)   # 1200 high-value customers
num_loyal      = int(NUM_CUSTOMERS * 0.30)   # 1800 loyal customers
num_regular    = int(NUM_CUSTOMERS * 0.30)   # 1800 regular customers
num_occasional = NUM_CUSTOMERS - num_champions - num_loyal - num_regular  # 1200 occasional

# Assign customer types and shuffle
customer_types = (
    ['champion']   * num_champions +
    ['loyal']      * num_loyal +
    ['regular']    * num_regular +
    ['occasional'] * num_occasional
)
random.shuffle(customer_types)
customer_segments = dict(zip(customer_ids, customer_types))

# Purchase behavior by segment (tuned to produce ~65% Pareto ratio)
segment_behavior = {
    'champion':   {'num_purchases': (18, 42)},
    'loyal':      {'num_purchases': (8, 18)},
    'regular':    {'num_purchases': (3, 9)},
    'occasional': {'num_purchases': (1, 2)}
}

# Spending multipliers per segment
SPEND_MULTIPLIERS = {
    'champion':   (1.5, 2.6),
    'loyal':      (1.0, 1.45),
    'regular':    (0.75, 1.1),
    'occasional': (0.45, 0.75)
}

# Purchase interval in days per segment
PURCHASE_INTERVALS = {
    'champion':   (6, 28),
    'loyal':      (20, 60),
    'regular':    (45, 95),
    'occasional': (75, 190)
}

# Generate transactions
transactions = []
transaction_id = 1

for customer_id in customer_ids:
    segment = customer_segments[customer_id]
    num_purchases = random.randint(*segment_behavior[segment]['num_purchases'])

    # Spread first purchase across first 70% of the date range
    days_range = (END_DATE - START_DATE).days
    first_purchase_offset = random.randint(0, int(days_range * 0.7))
    current_date = START_DATE + timedelta(days=first_purchase_offset)

    for _ in range(num_purchases):
        # Time between purchases
        days_between = random.randint(*PURCHASE_INTERVALS[segment])
        current_date += timedelta(days=days_between)

        if current_date > END_DATE:
            break

        # Pick category and calculate amount
        category = random.choice(list(CATEGORIES.keys()))
        base_amount = random.uniform(*CATEGORIES[category])
        amount = base_amount * random.uniform(*SPEND_MULTIPLIERS[segment])

        # Seasonal boost (Nov-Dec)
        if current_date.month in [11, 12]:
            amount *= random.uniform(1.1, 1.3)

        transactions.append({
            'transaction_id': f'TXN_{transaction_id:06d}',
            'customer_id': customer_id,
            'transaction_date': current_date.strftime('%Y-%m-%d'),
            'category': category,
            'amount': round(amount, 2)
        })
        transaction_id += 1

# Create DataFrame and shuffle
df = pd.DataFrame(transactions)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Verify Pareto distribution
customer_revenue = df.groupby('customer_id')['amount'].sum().sort_values(ascending=False)
total_revenue = customer_revenue.sum()
top_20_pct = int(len(customer_revenue) * 0.20)
top_20_revenue = customer_revenue.head(top_20_pct).sum()
top_20_percentage = (top_20_revenue / total_revenue) * 100

print(f"\n{'='*60}")
print(f"Dataset Statistics:")
print(f"{'='*60}")
print(f"Total Transactions: {len(df):,}")
print(f"Unique Customers:   {df['customer_id'].nunique():,}")
print(f"Date Range:         {df['transaction_date'].min()} to {df['transaction_date'].max()}")
print(f"Total Revenue:      ${total_revenue:,.2f}")
print(f"Avg Transaction:    ${df['amount'].mean():.2f}")
print(f"\n{'='*60}")
print(f"Pareto Analysis:")
print(f"{'='*60}")
print(f"Top 20% of customers ({top_20_pct:,}): ${top_20_revenue:,.2f} ({top_20_percentage:.1f}% of revenue)")
print(f"{'='*60}\n")

# Save
output_path = 'data/ecommerce_transactions.csv'
df.to_csv(output_path, index=False)
print(f"✅ Data saved to: {output_path}")
print(f"✅ Ready for analysis!")