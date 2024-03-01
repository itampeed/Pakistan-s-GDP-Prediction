import tkinter as tk
from tkinter import ttk
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import geopandas as gpd

# Load the CSV file into a DataFrame
df = pd.read_csv('Annual_GDP_of_Countries.csv')

# Load the world map using geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Create a Tkinter window
root = tk.Tk()
root.title("GDP Analysis")

# Create a tab control
tab_control = ttk.Notebook(root)

# Tab 1: Pakistan's previous GDP analysis
tab1 = ttk.Frame(tab_control)
tab_control.add(tab1, text="Pakistan's Previous GDP")

# Extract Pakistan's GDP data
pakistan_gdp = df[df['Country Name'] == 'Pakistan']
years = pakistan_gdp.columns[1:].astype(int)
gdp_values = pakistan_gdp.iloc[0, 1:].astype(float)

# Plot Pakistan's previous GDP data
fig1, ax1 = plt.subplots()
ax1.plot(years, gdp_values, marker='o', linestyle='-')
ax1.set_title('GDP Analysis of Pakistan')
ax1.set_xlabel('Year')
ax1.set_ylabel('GDP (in Trillions)')
ax1.grid(True)
canvas1 = FigureCanvasTkAgg(fig1, master=tab1)
canvas1.draw()
canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Tab 2: Comparison of different countries' GDP
tab2 = ttk.Frame(tab_control)
tab_control.add(tab2, text="Comparison of Countries")

# Extract GDP data for comparison countries
comparison_countries = ['United States', 'Turkiye', 'China', 'Russian Federation', 'India']
comparison_data = df[df['Country Name'].isin(comparison_countries)]
comparison_years = comparison_data.columns[1:].astype(int)
comparison_gdp_values = comparison_data.iloc[:, 1:].astype(float)

# Plot comparison of different countries' GDP
fig2, ax2 = plt.subplots()
for country in comparison_countries:
    ax2.plot(comparison_years, comparison_gdp_values.loc[comparison_data['Country Name'] == country].values.flatten(), label=country)
ax2.set_title('Comparison of Countries GDP')
ax2.set_xlabel('Year')
ax2.set_ylabel('GDP (in Trillions)')
ax2.grid(True)
ax2.legend()
canvas2 = FigureCanvasTkAgg(fig2, master=tab2)
canvas2.draw()
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Tab 3: World map
tab3 = ttk.Frame(tab_control)
tab_control.add(tab3, text="World Map")

# Plot the world map
fig3, ax3 = plt.subplots()
world.plot(ax=ax3, color='lightgray')
ax3.set_title('World Map')
ax3.set_xlabel('Longitude')
ax3.set_ylabel('Latitude')
ax3.grid(True)
canvas3 = FigureCanvasTkAgg(fig3, master=tab3)
canvas3.draw()
canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Tab 4: GDP analysis of future
tab4 = ttk.Frame(tab_control)
tab_control.add(tab4, text="GDP Analysis of Future")

# Predict the GDP of Pakistan for the future
start_year = 2024
end_year = 2040
X = years.values.reshape(-1, 1)
y = gdp_values.values
poly_features = PolynomialFeatures(degree=3)
X_poly = poly_features.fit_transform(X)
model = LinearRegression()
model.fit(X_poly, y)
years_to_predict = np.arange(start_year, end_year + 1).reshape(-1, 1)
X_poly_predict = poly_features.transform(years_to_predict)
predicted_gdp = model.predict(X_poly_predict)

# Plot the GDP analysis of future
fig4, ax4 = plt.subplots()
ax4.plot(years_to_predict.flatten(), predicted_gdp, marker='o', linestyle='-', color='orange')
ax4.set_title('GDP Analysis of Pakistan (Future Prediction)')
ax4.set_xlabel('Year')
ax4.set_ylabel('GDP (in Trillions)')
ax4.grid(True)
canvas4 = FigureCanvasTkAgg(fig4, master=tab4)
canvas4.draw()
canvas4.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Pack the tab control
tab_control.pack(expand=1, fill='both')

# Run the Tkinter event loop
root.mainloop()