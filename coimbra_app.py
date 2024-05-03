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
from ipyvizzu import Chart, Data, Config, Style

# CHANGE: Language dictionaries
texts = {
    'en': {
        'title': 'Interactive Map of Coimbra',
        'file_processor': 'File Processor',
        'interactive_map': 'Interactive Map',
        'district_map': 'Coimbra District Map',
        'forecast': 'Forecast',
        'upload_excel': 'Upload your Excel file',
        'download_csv': 'Download as CSV',
        'select_page': 'Select a Page',
        'select_year': 'Select Year',
        'select_bins': 'Select Number of Bins (Colors):',
        'show_raw_data': 'Show Raw Data',
        'select_geographical_level': 'Select Geographical Level',
        'show_3d_view': 'Show 3D View',
        'elevation_scale': 'Elevation Scale',
        'select_column': 'Select Column',  # Added translation
        'select_color_map': 'Select Color Map'  # Added translation
    },
    'pt': {
        'title': 'Mapa Interativo de Coimbra',
        'file_processor': 'Processador de Arquivos',
        'interactive_map': 'Mapa Interativo',
        'district_map': 'Mapa do Distrito de Coimbra',
        'forecast': 'Previsão',
        'upload_excel': 'Carregue o seu ficheiro Excel',
        'download_csv': 'Baixar como CSV',
        'select_page': 'Selecione uma Página',
        'select_year': 'Selecione o Ano',
        'select_bins': 'Selecione o Número de Divisões (Cores):',
        'show_raw_data': 'Mostrar Dados Brutos',
        'select_geographical_level': 'Selecione o Nível Geográfico',
        'show_3d_view': 'Mostrar Vista 3D',
        'elevation_scale': 'Escala de Elevação',
        'select_column': 'Selecione a Coluna', 
        'select_color_map': 'Selecione o Mapa de Cores' 
    }
}

# Set Streamlit page configuration to use wide mode
st.set_page_config(layout="wide")

# Set the initial state of language
if 'lang' not in st.session_state:
    st.session_state['lang'] = 'en'

# Function to toggle language
def toggle_language():
    st.session_state['lang'] = 'pt' if st.session_state['lang'] == 'en' else 'en'

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
def get_color(value, bin_edges, cmap):
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

def load_census_data():
    return pd.read_excel('./tables/CENSUS_2021.xlsx') 

# Streamlit app layout
# Display the toggle button
st.sidebar.image("https://i.postimg.cc/hjT72Vcx/logo-black.webp", use_column_width=True)
col1, col2, col3 = st.sidebar.columns([1,2,1])
toggle_button_label = "Switch to Português" if st.session_state['lang'] == 'en' else "Switch to English"
with col2:
    if st.button(toggle_button_label):
        toggle_language()
# Using the language setting
lang = st.session_state['lang']
st.sidebar.title(texts[lang]['select_page'])  
page = st.sidebar.radio(texts[lang]['select_page'], [texts[lang]['file_processor'], texts[lang]['interactive_map'], texts[lang]['district_map'], texts[lang]['forecast']], key='page_select')

if page == texts[lang]['file_processor']:
    st.title(texts[lang]['file_processor']) 
    uploaded_file = st.file_uploader(texts[lang]['upload_excel'], type=["xlsx"])
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        header_row = 1 if lang == 'pt' else 0 
        for sheet_name in xls.sheet_names:
            df = process_sheet(xls, sheet_name, header_row)
            st.write(f"Preview of {sheet_name}:")
            st.dataframe(df.head())
            csv = to_csv(df)
            st.download_button(
                label=texts[lang]['download_csv'].format(sheet_name=sheet_name),
                data=csv,
                file_name=f"{sheet_name}.csv",
                mime='text/csv',
            )

if page == texts[lang]['interactive_map']:
    st.title(texts[lang]['interactive_map'])

    # Create columns for controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year = st.slider(texts[lang]['select_year'], 2020, 2023, key='year_slider')
        num_bins = st.select_slider(
            texts[lang]['select_bins'],
            options=[0, 2, 3, 4, 5, 6, 7, 8, 9],
            format_func=lambda x: 'Auto' if x == 0 else str(x)
        )

    with col2:
        level = st.radio(texts[lang]['select_geographical_level'], ["Municipal", "District"], key='geo_level')
        color_map = st.selectbox(
            texts[lang]['select_color_map'],
            options=["Warm Sunset", "Viridis", "Plasma", "Inferno", "Cividis"],
            index=0  # Default to "Warm Sunset"
        )
        
    with col3:
        show_raw_data = st.checkbox(texts[lang]['show_raw_data'], key='show_raw_data_checkbox')
        show_3d = st.checkbox(texts[lang]['show_3d_view'])
        
        if show_3d:
            elevation_scale = st.slider(texts[lang]['elevation_scale'], 1, 500, 100)

    # Load data
    df_path = f'tables/{year}.xlsx'
    df = pd.read_excel(df_path)
    column_names = df.columns.tolist()[5:]
    column_name = st.selectbox(texts[lang]['select_column'], column_names)

    # Load and prepare geographical data based on selected level
    if level == "Municipal":
        gdf = gpd.read_file('./maps/municipal.shp').to_crs(epsg=4326)
        geo_column = 'NAME_2_cor'
    else:
        gdf = gpd.read_file('./maps/district.shp').to_crs(epsg=4326)
        geo_column = 'NUTIII_DSG'

    # Exclude selected municipalities or districts
    available_geographical_units = gdf[geo_column].unique()
    excluded_units = st.multiselect(f'Select {level}s to Exclude:', available_geographical_units, default=[])

    # Filter the GeoDataFrame to exclude the selected units
    gdf = gdf[~gdf[geo_column].isin(excluded_units)]
    
    # Merge data based on the level, using 'Border' as the common column in the Excel data
    merged = gdf.merge(df, how='left', left_on=geo_column, right_on='Border')

    if show_3d:
        deck_gl = generate_3d_map(merged, column_name, elevation_scale)
        st.pydeck_chart(deck_gl)
    else:
        # Ensure the data for the selected column is numeric and handle NaNs
        merged[column_name] = pd.to_numeric(merged[column_name], errors='coerce')
        merged.fillna(0, inplace=True)
        m = leafmap.Map(center=[40.2056, -8.4196], zoom_start=10)

        # Calculate the color bins for the map
        if num_bins > 0:
            bin_edges = calculate_quantile_bins(merged[column_name], num_bins)
        else:
            bin_edges = calculate_quantile_bins(merged[column_name])

        def get_colormap(cmap_name):
            cmap_dict = {
                "Warm Sunset (YlOrRd)": plt.get_cmap('YlOrRd'),
                "Viridis": plt.get_cmap('viridis'),
                "Plasma": plt.get_cmap('plasma'),
                "Inferno": plt.get_cmap('inferno'),
                "Cividis": plt.get_cmap('cividis')
            }
            return cmap_dict.get(cmap_name, plt.get_cmap('YlOrRd'))  # Default to 'YlOrRd' if no match

        # When applying styles in folium:
        current_cmap = get_colormap(color_map)

        # Setup the GeoJson layer with conditional tooltips based on the selected level
        tooltip_fields = ['Border', column_name]  # Tooltip shows the Border and the data column
        tooltip_aliases = [level, column_name.title()]  # Alias reflects the level

        geojson_layer = folium.GeoJson(
            data=merged.__geo_interface__,
            style_function=lambda feature: {
                'fillColor': get_color(feature['properties'][column_name], bin_edges, current_cmap) if feature['properties'][column_name] is not None else "transparent",
                'color': 'black',
                'weight': 0.5,
                'dashArray': '5, 5',
                'fillOpacity': 0.6,
            },
            tooltip=folium.GeoJsonTooltip(
                fields=tooltip_fields,
                aliases=tooltip_aliases,
                style=("background-color: white; color: #333333; font-family: Arial; font-size: 12px; padding: 3px;")
            )
        ).add_to(m)

        # Layout for map and legend
        col_map, col_legend = st.columns([9, 1])
        with col_map:
            m.to_streamlit(height=700)

        with col_legend:
            fig, ax = plt.subplots(figsize=(2, 6))
            cmap = current_cmap
            norm = mcolors.BoundaryNorm(bin_edges, cmap.N)
            colorbar = mcolorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
            colorbar.set_label(f"{column_name.replace('_', ' ').title()}", size=12)
            st.pyplot(fig)

    if show_raw_data:
        selected_columns = st.multiselect("Select columns to display:", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[selected_columns])

elif page == texts[lang]['district_map']:
    st.title(texts[lang]['district_map'])  # CHANGE: Localized title

    # Define the URL for the Kepler.gl map
    map_url = "https://kepler.gl/demo/map?mapUrl=https://dl.dropboxusercontent.com/scl/fi/t5u7fwfy8xf5zlcqsdv9s/Coolectiva.json?rlkey=fpwk2ahuwa3lzckpt076uk0h3&dl=0"
    
    # Create a full-width container for the iframe
    with st.container():
        st.components.v1.iframe(map_url, width=None, height=720)  # Set width to None for full width

elif page == texts[lang]['forecast']:
    st.title(texts[lang]['forecast'])  # CHANGE: Localized title
    census_df = load_census_data()

    # Dynamic Chart Configuration Options
    st.subheader("Chart Generator")
    chart_type = st.selectbox("Select Chart Type", ['Bar', 'Line', 'Area', 'Pie'], key='chart_type')
    x_axis = st.selectbox("Choose the X-axis", census_df.columns, key='x_axis')
    y_axis = st.selectbox("Choose the Y-axis", census_df.columns, key='y_axis')
    color_scheme = st.selectbox("Select Color Scheme", ['classic', 'candy', 'autumn', 'dark'], key='color_scheme')

    if st.button("Generate Chart", key='generate_chart'):
        # Directly setting data during the initialization of the Data object
        data = Data({
            "datasets": [
                {
                    x_axis: census_df[x_axis].tolist(),
                    y_axis: census_df[y_axis].tolist()
                }
            ]
        })

        chart = Chart()
        chart.animate(data)

        config = Config({
            "channels": {
                "y": {"set": y_axis},
                "x": {"set": x_axis},
                "color": {"set": x_axis, "range": {"scheme": color_scheme}}
            },
            "geometry": chart_type.lower()
        })

        chart.animate(config)

        style = Style({
            "plot": {
                "xAxis": {"label": {"fontSize": 14}},
                "yAxis": {"label": {"fontSize": 14}},
                "title": {"fontSize": 20}
            }
        })

        chart.animate(style)

        # Display the chart in Streamlit using HTML
        st.components.v1.html(chart.to_html(), height=500)
