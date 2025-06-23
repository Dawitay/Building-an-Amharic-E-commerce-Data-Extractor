import pandas as pd

# Load the Excel file
excel_file = '../data/raw_data/telegram_data.xlsx'  
df = pd.read_excel(excel_file)

# Save as CSV
csv_file = '../data/raw_data/telegram_data.csv'  
df.to_csv(csv_file, index=False)

print(f"Excel file has been successfully converted to {csv_file}")
