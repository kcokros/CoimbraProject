import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from io import BytesIO

# Function to process each sheet in the Excel file
def process_sheet(xls, sheet_name):
    df = pd.read_excel(xls, sheet_name=sheet_name, skiprows=2)
    new_columns = [f'{col} ({unit})' for col, unit in zip(df.columns, df.iloc[0])]
    df.columns = new_columns
    df = df[1:]  # Drop the row with measurement units
    df = df.loc[:, :(df.isnull().all().cumsum() == 1).idxmax()]
    df.dropna(axis=1, how='all', inplace=True)
    return df

# Function to save dataframe to a CSV and return it as a download link
def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False, sep=';', encoding='utf-8')
    output.seek(0)
    return output

# Streamlit app layout
st.sidebar.image: st.sidebar.image("https://i.postimg.cc/hjT72Vcx/logo-black.webp", use_column_width=True)
st.sidebar.title("Coimbra Interactive Map")
page = st.sidebar.radio("Select a Page", ["File Processor", "Interactive Map", "Forecast"])

if page == "File Processor":
    st.title("File Processor")

    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)

        for sheet_name in xls.sheet_names:
            df = process_sheet(xls, sheet_name)
            st.write(f"Preview of {sheet_name}:")
            st.dataframe(df.head())

            # Convert DataFrame to CSV
            csv = to_csv(df)
            st.download_button(
                label=f"Download {sheet_name} as CSV",
                data=csv,
                file_name=f"{sheet_name}.csv",
                mime='text/csv',
            )

elif page == "Interactive Map":
    st.title("Interactive Map")
# Load the shapefile using Geopandas
gdf = gpd.read_file('./maps/AreasEdificadas2018.shp')

# Check the unique categories in the 'TIPO_p' column
categories = gdf['TIPO_p'].unique()

# Sidebar filter for categories
selected_category = st.sidebar.selectbox('Select a Category:', categories)

# Filter the GeoDataFrame based on the selected category from the sidebar
filtered_gdf = gdf[gdf['TIPO_p'] == selected_category]

# Create a Folium map object centered on the filtered data
m = folium.Map(location=[filtered_gdf.geometry.centroid.y.mean(), 
                         filtered_gdf.geometry.centroid.x.mean()],
               zoom_start=12)

# Add the filtered shapefile to the map using GeoJson
folium.GeoJson(
    filtered_gdf,
    name='geojson',
    tooltip=folium.GeoJsonTooltip(fields=['TIPO_p'], labels=True)
).add_to(m)

# Add layer control to toggle on/off
folium.LayerControl().add_to(m)

# Display the map in Streamlit
folium_static(m)

elif page == "Forecast":
    st.title("Forecast")
    st.write("Forecast will be implemented here.")
