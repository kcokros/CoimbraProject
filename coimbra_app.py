import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from io import BytesIO
import json
from keplergl import KeplerGl
import numpy as n
import folium
import leafmap.foliumap as leafmap
import leafmap.colormaps as cm
import seaborn as sns
import pydeck as pdk

# Set Streamlit page configuration to use wide mode
st.set_page_config(layout="wide")

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

# Function to calculate quantile bins
def calculate_quantile_bins(data, num_bins=5):
    quantiles = [i / num_bins for i in range(1, num_bins)]
    bin_edges = list(data.quantile(quantiles))
    min_edge, max_edge = data.min(), data.max()
    bin_edges = [min_edge] + bin_edges + [max_edge]
    return bin_edges

# Function to create a colormap based on quantile bins
def get_color(value, bin_edges):
    cmap = plt.get_cmap('YlOrRd')
    norm = mcolors.BoundaryNorm(bin_edges, cmap.N)
    return mcolors.to_hex(cmap(norm(value)))

# Function to create a Pydeck 3D visualization
def generate_3d_map(geo_data_frame, column_name, elevation_scale):
    geo_data_frame['elevation'] = pd.to_numeric(geo_data_frame[column_name], errors='coerce') * elevation_scale
    layer = pdk.Layer(
        "GeoJsonLayer",
        geo_data_frame,
        opacity=0.8,
        stroked=False,
        filled=True,
        extruded=True,
        wireframe=True,
        get_elevation="elevation",
        get_fill_color="[200, 100, 100]",
        get_line_color="[255, 255, 255, 255]"
    )

    view_state = pdk.ViewState(
        latitude=geo_data_frame.geometry.centroid.y.mean(),
        longitude=geo_data_frame.geometry.centroid.x.mean(),
        zoom=10,
        pitch=45
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style='mapbox://styles/mapbox/light-v9')


# Streamlit app layout
st.sidebar.image("https://i.postimg.cc/hjT72Vcx/logo-black.webp", use_column_width=True)
st.sidebar.title("Coimbra Interactive Map")
page = st.sidebar.radio("Select a Page", ["File Processor", "Interactive Map", "Coimbra District Map", "Forecast"], key='page_select')

if page == "File Processor":
    st.title("File Processor")
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        for sheet_name in xls.sheet_names:
            df = process_sheet(xls, sheet_name)
            st.write(f"Preview of {sheet_name}:")
            st.dataframe(df.head())
            csv = to_csv(df)
            st.download_button(
                label=f"Download {sheet_name} as CSV",
                data=csv,
                file_name=f"{sheet_name}.csv",
                mime='text/csv',
            )

if page == "Interactive Map":
    st.title("Interactive Map")

    year = st.sidebar.slider("Select Year", 2020, 2023, key='year_slider')
    num_bins = st.sidebar.select_slider(
        "Select Number of Bins (Colors) or 'Auto' for automatic binning:",
        options=[0, 2, 3, 4, 5, 6, 7, 8, 9],
        format_func=lambda x: 'Auto' if x == 0 else str(x)
    )
    show_raw_data = st.sidebar.checkbox("Show Raw Data", key='show_raw_data_checkbox')
    level = st.sidebar.radio("Select Geographical Level", ["Municipal", "District"], key='geo_level')
    show_3d = st.sidebar.checkbox("Show 3D View")

    df_path = f'tables/{year}.xlsx'
    df = pd.read_excel(df_path)
    column_names = df.columns.tolist()[5:]
    column_name = st.sidebar.selectbox("Select Column", column_names)

    if level == "Municipal":
        gdf = gpd.read_file('./maps/municipal.shp').to_crs(epsg=4326)
    else:
        gdf = gpd.read_file('./maps/district.shp').to_crs(epsg=4326)
    
    merged = gdf.merge(df, how='left', left_on='NAME_2_cor', right_on='Border' if level == "Municipal" else 'NUTIII_DSG')

    if show_3d:
        elevation_scale = st.slider("Elevation Scale", 1, 500, 100)
        deck_gl = generate_3d_map(merged, column_name, elevation_scale)
        st.pydeck_chart(deck_gl)
    else:
        merged[column_name] = pd.to_numeric(merged[column_name], errors='coerce')
        merged.fillna(0, inplace=True)
        m = leafmap.Map(center=[40.2056, -8.4196], zoom_start=10)

        if num_bins > 0:
            bin_edges = calculate_quantile_bins(merged[column_name], num_bins)
        else:
            bin_edges = calculate_quantile_bins(merged[column_name])

        def style_function(feature):
            value = feature['properties'][column_name]
            fillColor = get_color(value, bin_edges) if value is not None else "transparent"
            return {
                'fillColor': fillColor,
                'color': 'black',
                'weight': 0.5,
                'dashArray': '5, 5',
                'fillOpacity': 0.6,
            }

        geojson_layer = folium.GeoJson(
            data=merged.__geo_interface__,
            style_function=style_function,
            tooltip=folium.GeoJsonTooltip(
                fields=['NAME_2', column_name],
                aliases=['Municipality', column_name.title()],
                style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 3px;")
            )
        ).add_to(m)
        m.to_streamlit(height=700)

    if show_raw_data:
        st.write("Raw Data")
        selected_columns = st.multiselect("Select columns to display:", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[selected_columns])

elif page == "Coimbra District Map":
    st.title("Coimbra Statistical Subsections Interactive Map")
    
    # Define the URL for the Kepler.gl map
    map_url = "https://kepler.gl/demo/map?mapUrl=https://dl.dropboxusercontent.com/scl/fi/t5u7fwfy8xf5zlcqsdv9s/Coolectiva.json?rlkey=fpwk2ahuwa3lzckpt076uk0h3&dl=0"
    
    # Create a full-width container for the iframe
    with st.container():
        st.components.v1.iframe(map_url, width=None, height=720)  # Set width to None for full width

elif page == "Forecast":
    st.title("Forecast")
    # Implement forecast functionality or model here if needed.

