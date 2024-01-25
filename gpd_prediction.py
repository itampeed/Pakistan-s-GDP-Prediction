import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import geopandas as gpd

# Load the CSV file into a DataFrame
df = pd.read_csv('Annual_GDP_of_Countries.csv')

# Load the world map using geopandas
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

def predict_Pakistans_GDP(dataframe, country_name, comparison_countries, start_year, end_year):
    # Extract the GDP data for the specified country
    country_gdp = dataframe[dataframe['Country Name'] == country_name]
    country_gdp_values = country_gdp.iloc[0, 1:].astype(float)
    years = np.array(country_gdp_values.index.astype(int))
    gdp_values = country_gdp_values.values
    X = years.reshape(-1, 1)
    y = gdp_values
    
    # Create and fit a polynomial regression model
    poly_features = PolynomialFeatures(degree=3)
    X_poly = poly_features.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict the GDP for the specified years using polynomial regression
    years_to_predict = np.arange(start_year, end_year + 1).reshape(-1, 1)
    X_poly_predict = poly_features.transform(years_to_predict)
    predicted_gdp_poly = model.predict(X_poly_predict)

    # Extract GDP data for comparison countries
    comparison_data = dataframe[dataframe['Country Name'].isin(comparison_countries)]
    comparison_years = np.array(comparison_data.columns[1:].astype(int))
    comparison_gdp_values = comparison_data.iloc[:, 1:].astype(float).values

    # Plot both the current and predicted GDP growth rates in a single window with four subplots
    fig, axes = plt.subplots(2, 2, gridspec_kw={'hspace': 0.30})

    # Set the space from the bottom to 0.12
    fig.subplots_adjust(left=0.07, bottom=0.12, right=0.96, top=0.9)

    # Add a slider to control the number of years displayed for the original GDP
    ax_slider_original = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow', transform=axes[0, 0].transAxes)
    slider_original = Slider(ax_slider_original, 'Years (Original GDP)', valmin=1, valmax=len(years), valinit=len(years), valstep=1)

    selected_years_original = int(slider_original.val)
    axes[0, 0].clear()
    axes[0, 0].plot(years[:selected_years_original], gdp_values[:selected_years_original], marker='o', linestyle='-')
    axes[0, 0].set_title(f'GDP of {country_name} (1960-2022)')
    axes[0, 0].set_xlabel('Year')
    axes[0, 0].set_ylabel('GDP Growth Rate')
    axes[0, 0].grid(True)

    # Define the update function for the original GDP slider
    def update_original(val):
        selected_years_original = int(slider_original.val)
        axes[0, 0].clear()
        axes[0, 0].plot(years[:selected_years_original], gdp_values[:selected_years_original], marker='o', linestyle='-')
        axes[0, 0].set_title(f'GDP of {country_name} (1960-2022)')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('GDP Growth Rate')
        axes[0, 0].grid(True)
        ax_slider_original.relim()
        ax_slider_original.autoscale_view()
        fig.canvas.draw_idle()

    # Connect the update function to the original GDP slider
    slider_original.on_changed(update_original)

    # Plot the GDP comparison of selected countries
    axes[0, 1].plot(comparison_years, comparison_gdp_values.T, linestyle='-')
    axes[0, 1].set_title('GDP Comparison of countries')
    axes[0, 1].set_xlabel('Year')
    axes[0, 1].set_ylabel('GDP Growth Rate')
    axes[0, 1].grid(True)
    axes[0, 1].legend(comparison_data['Country Name'])

    # Increase width to 12x10 by setting aspect ratio to be equal
    world.plot(ax=axes[1, 0], color='lightgray', aspect='equal')  

    # Plot Pakistan's geography on the world map
    pakistan_geom = world[world['name'] == 'Pakistan'].geometry
    India_geom = world[world['name'] == 'India'].geometry
    China_geom = world[world['name'] == 'China'].geometry
    US_geom = world[world['name'] == 'United States of America'].geometry
    Russia_geom = world[world['name'] == 'Russia'].geometry
    Turkiye_geom = world[world['name'] == 'Turkey'].geometry

    # Fill the entire area of Pakistan with blue
    pakistan_geom.plot(ax=axes[1, 0], color='green')
    India_geom.plot(ax=axes[1, 0], color='orange')
    China_geom.plot(ax=axes[1, 0], color='blue')
    Russia_geom.plot(ax=axes[1, 0], color='red')
    US_geom.plot(ax=axes[1, 0], color='brown')
    Turkiye_geom.plot(ax=axes[1, 0], color='purple')
    
    axes[1, 0].set_title('Pakistan')

    # Plot the predicted GDP data using polynomial regression
    axes[1, 1].plot(years_to_predict.flatten(), predicted_gdp_poly, marker='o', linestyle='-', color='orange')
    axes[1, 1].set_title(f'Predicted GDP Growth Rates of {country_name}')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('GDP Growth Rate')
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

# Predict the GDP growth rates for Pakistan from 2024 to 2040 using polynomial regression
comparison_countries = ['United States', 'Turkiye', 'China', 'Russian Federation', 'India', 'Pakistan']
predict_Pakistans_GDP(df, 'Pakistan', comparison_countries, 2024, 2040)
