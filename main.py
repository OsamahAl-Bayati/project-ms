# for numerical operations
import numpy as np
# for data manipulation and analysis
import pandas as pd
# from scikit-learn to perform linear regression
from sklearn.linear_model import LinearRegression
# for creating plots
import matplotlib.pyplot as plt
# for generating random numbers
import random
# for showing progress bars in loops
from tqdm import tqdm
# for formatting tick labels on the y-axis
from matplotlib.ticker import FuncFormatter

# Define function 'read_data' to read the CSV file and return the data as a pandas DataFrame.
def read_data(file_name):
    data = pd.read_csv(file_name)
    return data

# Define function 'fit_linear_regression' that takes input features 'X' and target variable 'y', creates a LinearRegression model, fits the model to the data, and returns the fitted model.
def fit_linear_regression(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

# Define function 'monte_carlo_projection' that performs Monte Carlo simulation to project sea level rise for a target year. The function takes a fitted model, target year, input data, and an optional number of simulations. It returns a list of predicted sea level rise values for the target year.
def monte_carlo_projection(model, target_year, data, n_simulations=1000):
    last_year = pd.to_datetime(data['date'].iloc[-1]).year
    n_years = target_year - last_year
    preds = []

    X = np.array(data.index).reshape(-1, 1)
    y = data['GMSL']

    for _ in range(n_simulations):
        noise = np.random.normal(0, y.std(), len(data))
        sim_data = y + noise
        sim_model = fit_linear_regression(X, sim_data)
        pred = sim_model.predict([[len(data) + n_years * 12]])
        preds.append(pred[0])

    return preds

# Define function 'damage_function' that estimates the damage caused by sea level rise.
def damage_function(sea_level_rise, coastal_population, infrastructure_value, flood_protection_level, exponential_factor=1.5, max_population_damage_fraction=0.8, max_infrastructure_damage_fraction=0.8):
    # Parameters
    exposure_factor = 0.1  # Fraction of coastal population exposed to sea level rise
    infrastructure_factor = 0.05  # Fraction of infrastructure value exposed to sea level rise
    
    # Calculate the damages
    population_exposure = coastal_population * exposure_factor
    infrastructure_exposure = infrastructure_value * infrastructure_factor
    
    # Sigmoid function
    def sigmoid(x, k=0.1):
        return 1 / (1 + np.exp(-k * x))
    
    # Calculate the inundation factor considering the flood protection level (both in mm)
    inundation_factor = sigmoid(sea_level_rise - flood_protection_level)
    
    # Calculate the damages with exponential factor
    raw_population_damage = population_exposure * (inundation_factor ** exponential_factor)
    raw_infrastructure_damage = infrastructure_exposure * (inundation_factor ** exponential_factor)
    
    # Limit damages based on the maximum allowed damage fractions
    max_population_damage = max_population_damage_fraction * coastal_population
    max_infrastructure_damage = max_infrastructure_damage_fraction * infrastructure_value
    
    population_damage = min(raw_population_damage, max_population_damage)
    infrastructure_damage = min(raw_infrastructure_damage, max_infrastructure_damage)
    
    total_damage = population_damage + infrastructure_damage
    # return total_damage
    return population_damage, infrastructure_damage

def main():
    file_name = 'GMSL.csv'
    target_year = 2050

    data = read_data(file_name)
    X = np.array(data.index).reshape(-1, 1)
    y = data['GMSL']
    model = fit_linear_regression(X, y)
    
    last_year = pd.to_datetime(data['date'].iloc[-1]).year
    years = np.arange(last_year + 1, target_year + 1)
    pop_damages = []
    infra_damages = []

    coastal_population = float(input("Enter the coastal population: "))
    infrastructure_value = float(input("Enter the infrastructure value: "))
    flood_protection_level = float(input("Enter the flood protection level (in mm): ")) + 104.5

    for year in tqdm(years, desc='Calculating damages'):
        preds = monte_carlo_projection(model, year, data)
        mean_pred = np.mean(preds)
        pop_damage, infra_damage = damage_function(mean_pred, coastal_population, infrastructure_value, flood_protection_level)
        pop_damages.append(pop_damage)
        infra_damages.append(infra_damage)

    def y_fmt(x, _):
        return f'{x:.0f}'

    y_formatter = FuncFormatter(y_fmt)

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 4, 1)
    preds = monte_carlo_projection(model, target_year, data)
    mean_pred = np.mean(preds)
    std_pred = np.std(preds)
    plt.hist(preds, bins=30)
    plt.xlabel('Sea Level Rise (mm)')
    plt.ylabel('Frequency')
    plt.title(f'Sea Level Rise Projections for {target_year}')

    plt.subplot(2, 4, 2)
    sea_level_rise = [np.mean(monte_carlo_projection(model, year, data)) for year in years]
    plt.plot(years, sea_level_rise, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Sea Level Rise (mm)')
    plt.title('Sea Level Rise Over Time')

    plt.subplot(2, 4, 3)
    total_damages = [pop_damage + infra_damage for pop_damage, infra_damage in zip(pop_damages, infra_damages)]
    plt.plot(years, total_damages, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Estimated Total Damage')
    plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.title('Total Damage Estimates Over Time')

    plt.subplot(2, 4, 4)
    plt.plot(years, pop_damages, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Estimated Population Damage')
    plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.title('Population Damage Estimates Over Time')

    plt.subplot(2, 4, 5)
    plt.plot(years, infra_damages, marker='o')
    plt.xlabel('Year')
    plt.ylabel('Estimated Infrastructure Damage')
    plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.title('Infrastructure Damage Estimates Over Time')

    plt.subplot(2, 4, 6)
    plt.scatter(sea_level_rise, total_damages)
    plt.xlabel('Sea Level Rise (mm)')
    plt.ylabel('Estimated Total Damage')
    plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.title('Sea Level Rise vs. Total Damage Estimates')

    plt.subplot(2, 4, 7)
    sorted_preds = np.sort(preds)
    cdf = np.arange(len(sorted_preds)) / (len(sorted_preds) - 1)
    plt.plot(sorted_preds, cdf)
    plt.xlabel('Sea Level Rise (mm)')
    plt.ylabel('Cumulative Probability')
    plt.title(f'Cumulative Distribution Function for {target_year} Projections')

    plt.subplot(2, 4, 8)
    decades = np.arange(last_year + 1, target_year + 1, 10)
    decade_total_damages = [np.sum(total_damages[years.tolist().index(decade):years.tolist().index(decade+10) if decade+10 in years.tolist() else None]) for decade in decades]
    plt.bar(decades, decade_total_damages, width=8)
    plt.xlabel('Decade')
    plt.ylabel('Total Estimated Damage')
    plt.gca().yaxis.set_major_formatter(y_formatter)
    plt.title('Total Damages by Decade')

    plt.tight_layout()
    plt.show()
    print(f"Estimated sea level rise in {target_year}: {mean_pred:.2f} mm (±{std_pred:.2f} mm)")
    print(f"Estimated population damage in {target_year}: {pop_damages[-1]:.2e}")
    print(f"Estimated infrastructure damage in {target_year}: {infra_damages[-1]:.2e}")

if __name__ == "__main__":
    main()