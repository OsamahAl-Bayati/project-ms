import pandas as pd

data = pd.read_csv('ftp://ftp.csiro.au/legresy/gmsl_files/CSIRO_Alt_seas_inc.csv')

# Convert the 'Time' column to datetime format
data['Time'] = pd.to_datetime((data['Time'] - 1970) * 365.25, unit='D', origin='unix')

# Extract the year and month from the 'Time' column and add them as new columns
data['year'] = data['Time'].dt.year
data['month'] = data['Time'].dt.month

# Combine the 'year' and 'month' columns into a single 'year-month' column
data['date'] = data['year'].astype(str) + '-' + data['month'].astype(str).str.zfill(2)

data['GMSL'] = data['GMSL (monthly)']

# Normalize the 'GMSL' column by subtracting the minimum value
data['GMSL'] = data['GMSL'] - data['GMSL'].min()

# Round down the 'GMSL' column to 1 decimal place
data['GMSL'] = data['GMSL'].round(decimals=1)

# Select the columns 'year-month' and 'GMSL' and save to a new dataframe
new_data = data[['date', 'GMSL']]

# Save the new dataframe to a CSV file without the index column
new_data.to_csv('GMSL.csv', index=False)