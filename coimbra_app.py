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

# Streamlit app layout
st.sidebar.image("https://i.postimg.cc/hjT72Vcx/logo-black.webp", use_column_width=True)
st.sidebar.title("Coimbra Interactive Map")
page = st.sidebar.radio("Select a Page", ["File Processor", "Interactive Map", "Interactive Map (Alt)", "Forecast"], key='page_select')

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
    # Update the slider to include 'Auto' as 0
    num_bins = st.sidebar.select_slider(
        "Select Number of Bins (Colors) or 'Auto' for automatic binning:",
        options=[0, 2, 3, 4, 5, 6, 7, 8, 9],
        format_func=lambda x: 'Auto' if x == 0 else str(x)
    )
    show_raw_data = st.sidebar.checkbox("Show Raw Data", key='show_raw_data_checkbox')
    level = st.sidebar.radio("Select Geographical Level", ["Municipal", "District"], key='geo_level')

    # Load data and setup
    municipal_gdf = gpd.read_file('./maps/municipal.shp').to_crs(epsg=4326)
    district_gdf = gpd.read_file('./maps/district.shp').to_crs(epsg=4326)
    df_path = f'tables/{year}.xlsx'
    df = pd.read_excel(df_path)
    column_names = df.columns.tolist()[5:]
    column_name = st.sidebar.selectbox("Select Column", column_names)

    # Conditionally merge data based on the selected level
    if level == "Municipal":
        merged = municipal_gdf.merge(df, left_on='NAME_2_cor', right_on='Border')
    elif level == "District":
        merged = district_gdf.merge(df, left_on='NUTIII_DSG', right_on='Border')

    merged[column_name] = pd.to_numeric(merged[column_name], errors='coerce')
    merged.fillna(0, inplace=True)

    if num_bins > 0:  # User selected specific number of bins
        bin_edges = calculate_quantile_bins(merged[column_name], num_bins)
    else:  # Automatic binning with default quantiles
        bin_edges = calculate_quantile_bins(merged[column_name])

    m = leafmap.Map(center=[40.2056, -8.4196], zoom_start=10)

    # Define the style function for choropleth
    def style_function(feature):
        value = feature['properties'][column_name]
        fillColor = get_color(value, bin_edges) if value is not None else "transparent"  # Check if value is None
        return {
            'fillColor': fillColor,
            'color': 'black',
            'weight': 0.5,
            'dashArray': '5, 5',
            'fillOpacity': 0.6,
        }

    # Define a function to create tooltips using HTML
    def get_tooltip_html(properties):
        name = properties.get('NAME_2', 'No name')
        value = properties.get(column_name, 'No data')
        tooltip_html = f"""
        <div style="min-width: 100px;">
            <b>Municipality:</b> {name}<br>
            <b>{column_name}:</b> {value}
        </div>
        """
        return tooltip_html

    # Add GeoJSON layer to the map with tooltip
    geojson_layer = folium.GeoJson(
        data=merged.__geo_interface__,
        style_function=style_function,  # Assuming this is defined elsewhere in your code
        tooltip=folium.GeoJsonTooltip(
            fields=['NAME_2', column_name],
            aliases=['Municipality', column_name.title()],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 3px;")
        )
    ).add_to(m)
    
    # Create layout for the map and the legend
    row3_col1, row3_col2 = st.columns([5, 1])

    # Display the map in the first column
    with row3_col1:
        m.to_streamlit(height=700)

    # Calculate min and max values for the selected column
    min_value = merged[column_name].min()
    max_value = merged[column_name].max()

    # Update the number of colors to match the number of bins or use 5 as default for 'Auto'
    n_colors = num_bins if num_bins > 0 else 5
    colors = cm.get_palette('YlOrRd', n_colors)
    colors = ['#' + color if not color.startswith('#') else color for color in colors]

    # Create a figure for the colormap legend
    fig, ax = plt.subplots(figsize=(2, 6))
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", colors)
    norm = mcolors.BoundaryNorm(bin_edges, cmap.N)
    cb = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label(column_name.replace("_", " ").title())
    plt.tight_layout()

    # Display the legend in the second column
    with row3_col2:
        st.pyplot(fig)

    if show_raw_data:
        st.write("Raw Data")
        selected_columns = st.multiselect("Select columns to display:", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[selected_columns])

elif page == "Interactive Map (Alt)":
    st.title("Interactive Map (Alt) using Kepler.gl")
    geojson_path = './maps/CENSUS_LEVEL.geojson'
    gdf = gpd.read_file(geojson_path)
    if gdf.crs != "epsg:4326":
        gdf = gdf.to_crs(epsg=4326)
    geojson_data = json.loads(gdf.to_json())

    map_config = {
        'version': 'v1',
        'config': {
            'mapState': {
                'latitude': 40.2056,
                'longitude': -8.4196,
                'zoom': 10
            }
        }
    }
    
    map_1 = KeplerGl(height=700)
    map_1.add_data(data=geojson_data, name='Census Data')
    with st.container():
        map_1.save_to_html(file_name='kepler_map.html')
        st.components.v1.html(open('kepler_map.html', 'r').read(), height=700)

elif page == "Forecast":
    st.title("Forecast")
    # Implement forecast functionality or model here if needed.
