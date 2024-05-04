import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.colorbar as mcolorbar
from io import BytesIO
import json
import os
import base64
from keplergl import KeplerGl
import numpy as n
import folium
import leafmap.foliumap as leafmap
import leafmap.colormaps as cm
import seaborn as sns
import pydeck as pdk
from preprocess import generateDataframes, replaceNewlineWithSpace, add_multiindex_columns, fillRowsInRangeForAll, find_limits_for_all, concatenateRowsWithinLimits

# Language dictionaries
texts = {
    'en': {
        'title': 'Interactive Map of Coimbra',
        'file_processor': 'File Processor',
        'interactive_map': 'Interactive Map',
        'district_map': 'Coimbra District Map',
        'chart_generator': 'Chart Generator',
        'upload_excel': 'Upload your Excel file',
        'download_csv': 'Download as CSV',
        'download_data_csv': 'Download Data as .csv',
        'download_chart_png': 'Download Chart as .png',
        'select_page': 'Select a Page',
        'select_year': 'Select Year',
        'select_bins': 'Select Number of Bins (Colors):',
        'show_raw_data': 'Show Raw Data',
        'select_geographical_level': 'Select Geographical Level',
        'show_3d_view': 'Show 3D View',
        'elevation_scale': 'Elevation Scale',
        'select_column': 'Select Column',
        'select_color_map': 'Select Color Map',
        'show_bar_chart' : 'Show Bar Chart',
        'select_data_source': 'Select Data Source',
        'census_data': 'Census 2021 Data',
        'upload_csv': 'Upload your dataset in CSV format',
        'enter_custom_title': 'Enter a custom title for the chart (leave blank for default):',
        'select_column_to_filter': 'Select Column to Filter Values (optional)',
        'select_values_to_include': 'Select Values to Include',
        'enter_row_ranges': "Enter Row Ranges (optional, e.g., '1-10, 15, 20-30, -5--1')",
        'select_chart_type': 'Select Chart Type',
        'select_x_axis_variable': 'Select X-axis Variable',
        'select_y_axis_variable': 'Select Y-axis Variable',
        'select_color_palette': 'Select Color Palette',
        'generate_chart': 'Generate Chart',
        'advanced_chart_builder': 'Advanced Chart Builder',
        'open_vizzu_builder': 'Open Vizzu Builder',
        'interactive_design_tools': 'Interactive design tools that allow you to create and customize charts.',
        'click_button_above': 'Click the button above to access Vizzu, a tool that allows you to design and customize charts interactively.'
    },
    'pt': {
        'title': 'Mapa Interativo de Coimbra',
        'file_processor': 'Processador de Arquivos',
        'interactive_map': 'Mapa Interativo',
        'district_map': 'Mapa do Distrito de Coimbra',
        'chart_generator': 'Gerador de Gráficos',
        'upload_excel': 'Carregue o seu ficheiro Excel',
        'download_csv': 'Baixar como CSV',
        'download_data_csv': 'Baixar Dados como .csv',
        'download_chart_png': 'Baixar gráfico como .png',
        'select_page': 'Selecione uma Página',
        'select_year': 'Selecione o Ano',
        'select_bins': 'Selecione o Número de Divisões (Cores):',
        'show_raw_data': 'Mostrar Dados Brutos',
        'select_geographical_level': 'Selecione o Nível Geográfico',
        'show_3d_view': 'Mostrar Vista 3D',
        'elevation_scale': 'Escala de Elevação',
        'select_column': 'Selecione a Coluna', 
        'show_bar_chart' : 'Mostrar gráfico de barras',
        'select_color_map': 'Selecione o Mapa de Cores',
        'select_data_source': 'Selecione a Fonte de Dados',
        'census_data': 'Dados do Censo 2021',
        'upload_csv': 'Carregue o seu conjunto de dados em formato CSV',
        'enter_custom_title': 'Insira um título personalizado para o gráfico (deixe em branco para o padrão):',
        'select_column_to_filter': 'Selecione a Coluna para Filtrar Valores (opcional)',
        'select_values_to_include': 'Selecione Valores para Incluir',
        'enter_row_ranges': "Insira os Intervalos de Linhas (opcional, ex.: '1-10, 15, 20-30, -5--1')",
        'select_chart_type': 'Selecione o Tipo de Gráfico',
        'select_x_axis_variable': 'Selecione a Variável do Eixo X',
        'select_y_axis_variable': 'Selecione a Variável do Eixo Y',
        'select_color_palette': 'Selecione a Paleta de Cores',
        'generate_chart': 'Gerar Gráfico',
        'advanced_chart_builder': 'Construtor Avançado de Gráficos',
        'open_vizzu_builder': 'Abrir Construtor Vizzu',
        'interactive_design_tools': 'Ferramentas de design interativas que permitem criar e personalizar gráficos.',
        'click_button_above': 'Clique no botão acima para acessar o Vizzu, uma ferramenta que permite projetar e personalizar gráficos de forma interativa.'
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

def plot_bar_chart(df, geo_column, column, color_map, title=None, axis_orientation='vertical'):
    """ Generate a bar chart with the specified settings. """
    plt.figure(figsize=(12, 8))
    cmap = plt.get_cmap(color_map)
    grouped_data = df.groupby(geo_column)[column].mean().sort_values()

    color_values = cmap(n.linspace(0, 1, len(grouped_data)))

    if axis_orientation == 'vertical':
        ax = grouped_data.plot(kind='bar', color=color_values, edgecolor='black')
        plt.xlabel(geo_column)
        plt.ylabel(column)
    else:
        ax = grouped_data.plot(kind='barh', color=color_values, edgecolor='black')
        plt.ylabel(geo_column)
        plt.xlabel(column)

    plt.title(title if title else f"Bar Chart of {column} by {geo_column}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    return plt.gcf()

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

def load_data():
    return pd.read_excel('./tables/CENSUS_2021.xlsx') 

# Function to parse row input for selecting rows
def parse_row_input(row_input, df_length):
    rows = []
    parts = row_input.split(',')
    for part in parts:
        if '-' in part:
            start, end = map(int, part.split('-'))
            if start < 0:  # Negative indexing
                start = df_length + start
            if end < 0:
                end = df_length + end
            rows.extend(range(start, end + 1))
        else:
            idx = int(part)
            if idx < 0:
                idx = df_length + idx
            rows.append(idx)
    return sorted(set(rows))

# Function to generate and display charts
def generate_chart(data, x_col, y_col, chart_type, palette, chart_title):
    plt.figure(figsize=(15, 8))
    if chart_type == 'Bar Chart':
        chart = sns.barplot(x=x_col, y=y_col, data=data, palette=palette)
    elif chart_type == 'Line Chart':
        chart = sns.lineplot(x=x_col, y=y_col, data=data, palette=palette)
    
    # Apply character limit for display purposes
    chart.set_xticklabels([label.get_text()[:15] + '...' if len(label.get_text()) > 15 else label.get_text() for label in chart.get_xticklabels()])
    plt.xticks(rotation=45)
    # Use user input title or default
    plt.title(chart_title if chart_title else f'{chart_type} for {y_col} by {x_col}')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    
    # Convert plot to PNG for download
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return buf

# Function to get CSV download link
def get_csv(data):
    output = BytesIO()
    data.to_csv(output, index=False)
    output.seek(0)
    return output

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
page = st.sidebar.radio(texts[lang]['select_page'], texts[lang]['file_processor'], texts[lang]['interactive_map'], texts[lang]['district_map'], texts[lang]['chart_generator'], key='page_select')

if page == texts[lang]['file_processor']:
    st.title(texts[lang]['file_processor'])
    uploaded_file = st.file_uploader(texts[lang]['upload_excel'], type=["xlsx"])
    
    if uploaded_file is not None:
        xls = pd.ExcelFile(uploaded_file)
        
        # Process the file
        try:
            # Generate DataFrames and extract units
            dfs, units = generateDataframes(xls, xls.sheet_names)
            lower, upper = find_limits_for_all(dfs)
            dfs = fillRowsInRangeForAll(dfs, lower, upper)
            dfs = concatenateRowsWithinLimits(dfs, lower, upper, units)
            dfs = replaceNewlineWithSpace(dfs)
            dfs = add_multiindex_columns(dfs)
            st.success('Processing Complete!')

            # Display each DataFrame with a download button for the CSV
            for (pt_title, en_title), df in dfs.items():
                st.subheader(f'DataFrame: {en_title}')
                st.dataframe(df)
                to_csv = df.to_csv(index=False).encode('utf-8')
                b64 = base64.b64encode(to_csv).decode()  # some browsers need base64 encoding
                href = f'<a href="data:file/csv;base64,{b64}" download="{en_title.replace("/", "_")}.csv">Download {en_title.replace("/", "_")}.csv</a>'
                st.markdown(href, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

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
        show_bar_chart = st.checkbox(texts[lang]['show_bar_chart'], key='show_bar_chart')
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
            return cmap_dict.get(cmap_name, plt.get_cmap('YlOrRd'))

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



    # After the map display code
    if show_bar_chart:
        axis_orientation = st.selectbox("Choose Bar Chart Orientation", ['vertical', 'horizontal'], index=0)
        custom_bar_chart_title = st.text_input("Custom Title for Bar Chart", "")
        custom_x_label = st.text_input("Custom X Axis Label", "")
        custom_y_label = st.text_input("Custom Y Axis Label", "")
        # Get the current color map as a string suitable for matplotlib
        plt_cmap_name = {
            "Warm Sunset": "YlOrRd",
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Cividis": "cividis"
        }.get(color_map, "YlOrRd")
        
        geo_column = 'NAME_2_cor' if level == "Municipal" else 'NUTIII_DSG'
        
        # Generate the bar chart with a custom or default title
        chart_fig = plot_bar_chart(merged, geo_column, column_name, plt_cmap_name, custom_bar_chart_title.strip() or None, axis_orientation)
        st.pyplot(chart_fig)

    if show_raw_data:
        selected_columns = st.multiselect("Select columns to display:", df.columns.tolist(), default=df.columns.tolist())
        st.dataframe(df[selected_columns])

elif page == texts[lang]['district_map']:
    st.title(texts[lang]['district_map'])

    # Define the URL for the Kepler.gl map
    map_url = "https://kepler.gl/demo/map?mapUrl=https://dl.dropboxusercontent.com/scl/fi/t5u7fwfy8xf5zlcqsdv9s/Coolectiva.json?rlkey=fpwk2ahuwa3lzckpt076uk0h3&dl=0"
    
    # Create a full-width container for the iframe
    with st.container():
        st.components.v1.iframe(map_url, width=None, height=720)

elif page == texts[lang]['district_map']:
    st.title(texts[lang]['district_map'])  # CHANGE: Localized title

    # Define the URL for the Kepler.gl map
    map_url = "https://kepler.gl/demo/map?mapUrl=https://dl.dropboxusercontent.com/scl/fi/t5u7fwfy8xf5zlcqsdv9s/Coolectiva.json?rlkey=fpwk2ahuwa3lzckpt076uk0h3&dl=0"
    
    # Create a full-width container for the iframe
    with st.container():
        st.components.v1.iframe(map_url, width=None, height=720)  # Set width to None for full width

elif page == texts[lang]['chart_generator']:
    st.title(texts[lang]['chart_generator'])

    data_source = st.radio(texts[lang]['select_data_source'], [texts[lang]['census_data'], texts[lang]['upload_csv']])
    data = None
    if data_source == texts[lang]['upload_csv']:
        uploaded_file = st.file_uploader(texts[lang]['upload_csv'], type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
    else:
        data = load_data()

    if data is not None:
        custom_title = st.text_input(texts[lang]['enter_custom_title'])
        
        filter_col = st.selectbox(texts[lang]['select_column_to_filter'], [""] + data.columns.tolist())
        if filter_col:
            unique_values = data[filter_col].dropna().unique()
            selected_values = st.multiselect(texts[lang]['select_values_to_include'], options=unique_values)
            data = data[data[filter_col].isin(selected_values)] if selected_values else data

        row_input = st.text_input(texts[lang]['enter_row_ranges'])
        if row_input:
            selected_rows = parse_row_input(row_input, len(data))
            data = data.iloc[selected_rows]

        if not data.empty:
            chart_type = st.selectbox(texts[lang]['select_chart_type'], ["Bar Chart", "Line Chart", "Other"])
            columns = data.columns.tolist()
            x_col = st.selectbox(texts[lang]['select_x_axis_variable'], columns, index=0)
            y_col = st.selectbox(texts[lang]['select_y_axis_variable'], columns, index=1)
            palette = st.selectbox(texts[lang]['select_color_palette'], sns.palettes.SEABORN_PALETTES.keys(), index=3)
            
            if st.button(texts[lang]['generate_chart']):
                buf = generate_chart(data, x_col, y_col, chart_type, palette, custom_title)
                st.download_button(texts[lang]['download_chart_png'], buf.getvalue(), file_name="chart.png", mime="image/png")
                csv = get_csv(data)
                st.download_button(texts[lang]['download_data_csv'], csv.getvalue(), file_name="data.csv", mime="text/csv")

    # Vizzu Link
    st.markdown(f"## {texts[lang]['advanced_chart_builder']}")
    st.markdown(
        f"""
        <a href="https://vizzu-builder.streamlit.app/" target="_blank">
            <button style="width:100%; height: 50px; color: white; background-color: #FF4B4B; border: none; border-radius: 5px; cursor: pointer;">
                {texts[lang]['open_vizzu_builder']}
            </button>
        </a>
        """,
        unsafe_allow_html=True
    )

    st.text(texts[lang]['click_button_above'])
