# Simulation on the effect of sea level rise on coastal infrastructure

This is a Python script to estimate the damage caused by sea level rise using Monte Carlo simulations, linear regression, and a simple damage function.

## Requirements

- Python 3.7+
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `tqdm`

## Installation

1. Clone the repository:
```
git clone https://github.com/OsamahAl-Bayati/project-ms.git
cd project-ms
```

2. Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```
py -3 -m venv venv
venv\Scripts\activate
```

3. Install the required packages:
```
pip install -r requirements.txt
```


## Usage

1. Place your `GMSL.csv` file containing sea level data in the repository folder.

2. Run the script:
```
python main.py
```


The script will ask for the following inputs:

- Coastal population
- Infrastructure value
- Flood protection level (in mm)

Enter these values and wait for the calculations and plots to be generated.

3. Analyze the generated plots and read the damage estimates printed in the terminal.




