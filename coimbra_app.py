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
from pathlib import Path
from locations import portugal_geo_structure 
from preprocess import generateDataframes, replaceNewlineWithSpace, addMultiindexColumns, fillRowsInRangeForAll, find_limits_for_all, concatenateRowsWithinLimits, refineHeaders
from selenium import webdriver
import time

# Language dictionaries
texts = {
    'en': {
        'title': 'Interactive Map of Coimbra',
        'file_processor': 'File Processor',
        'interactive_map': 'Interactive Map',
        'district_map': 'Coimbra Region Map',
        'chart_generator': 'Chart Generator',
        'upload_excel': 'Upload your Excel file',
        'download_csv': 'Download as CSV',
        'download_data_csv': 'Download Data as .csv',
        'download_chart_png': 'Download Chart as .png',
        'select_page': 'Select a Page',
        'select_indicator': 'Select an Indicator to display',
        'select_year': 'Select Year',
        'select_topic': 'Select a Topic to explore',
        'select_bins': 'Select Number of Bins (Colors):',
        'select_areas' : 'Select Areas',
        'select_regions' : 'Select Regions',
        'select_levels': "Select {level}s to Include:",
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
        'choose_bar_orientation':  'Choose bar chart orientation',
        'enter_custom_title': 'Enter a custom title for the chart (leave blank for default):',
        'enter_x_axis_label':'Enter X Axis Label',
        'enter_y_axis_label':'Enter Y Axis Label',
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
        'file_processor': 'Processador de Ficheiros',
        'interactive_map': 'Mapa Interativo',
        'district_map': 'Mapa da Região de Coimbra',
        'chart_generator': 'Gerador de Gráficos',
        'upload_excel': 'Carregue aqui o seu ficheiro em formato Excel',
        'download_csv': 'Descarregar como CSV',
        'download_data_csv': 'Descarregar Dados como CSV',
        'download_chart_png': 'Descarregar gráfico como PNG',
        'select_page': 'Selecione uma Página',
        'select_indicator': 'Selecione um Indicador para visualizar',
        'select_topic': 'Selecione o Tópico que deseja explorar',
        'select_year': 'Selecione o Ano',
        'select_bins': 'Selecione o Número de Divisões (Cores):',
        'select_areas' : 'Selecione as Áreas',
        'select_regions' : 'Selecione as Regiões',
        'select_levels': "Selecione a nivel {level} o que deseja explorar:",
        'show_raw_data': 'Mostrar Tabela de Dados',
        'select_geographical_level': 'Selecione o Nível Geográfico',
        'show_3d_view': 'Mostrar Vista 3D',
        'elevation_scale': 'Escala de Elevação',
        'select_column': 'Selecione a Coluna', 
        'show_bar_chart' : 'Mostrar gráfico de barras',
        'select_color_map': 'Selecione as Cores do Mapa',
        'select_data_source': 'Selecione a Fonte de Dados',
        'census_data': 'Dados do Censo 2021',
        'upload_csv': 'Carregue o seu conjunto de dados em formato CSV',
        'choose_bar_orientation':  'Escolha a orientação do gráfico',
        'enter_custom_title': 'Personalize o título do gráfico (deixe em branco para o padrão):',
        'enter_x_axis_label':'Personalize a legenda do eixo X',
        'enter_y_axis_label':'Personalize a legenda do eixo Y',
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
page = st.sidebar.radio(texts[lang]['select_page'], [texts[lang]['file_processor'], texts[lang]['interactive_map'], texts[lang]['district_map'], texts[lang]['chart_generator']], key='page_select')

if page == texts[lang]['file_processor']:
    st.title(texts[lang]['file_processor'])
    uploaded_file = st.file_uploader(texts[lang]['upload_excel'], type=["xlsx"])
    processed_data = {}
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
            dfs = refineHeaders(dfs)
            dfs = addMultiindexColumns(dfs, lang)
            st.success('Processing Complete!')

            # Display each DataFrame with a download button for the CSV
            for (pt_title, en_title), df in dfs.items():
                title = pt_title if lang == 'pt' else en_title
                st.subheader(f'DataFrame: {title}')
                st.dataframe(df)
                
                processed_data[(pt_title, en_title)] = df
                
                # Store DataFrame in session state
                st.session_state['processed_data'] = processed_data

                # Convert DataFrame to CSV, ensure index is included if it needs to be preserved
                csv = df.to_csv(index=True).encode('utf-8-sig')  # Ensure index is included if needed
                st.download_button(
                    label=f"Download {title.replace('/', '_')}.csv",
                    data=csv,
                    file_name=f"{title.replace('/', '_')}.csv",
                    mime='text/csv'
                )

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")


if page == texts[lang]['interactive_map']:
    st.title(texts[lang]['interactive_map'])

    # Create columns for controls
    col1, col2 = st.columns(2)
    
    with col1:
        num_bins = st.select_slider(
            texts[lang]['select_bins'],
            options=[0, 2, 3, 4, 5, 6, 7, 8, 9],
            format_func=lambda x: 'Auto' if x == 0 else str(x)
        )
        show_bar_chart = st.checkbox(texts[lang]['show_bar_chart'], key='show_bar_chart')
        show_raw_data = st.checkbox(texts[lang]['show_raw_data'], key='show_raw_data_checkbox')

    with col2:
        level = st.radio(texts[lang]['select_geographical_level'], ["Municipal", "Regional"], key='geo_level')
        color_map = st.selectbox(
            texts[lang]['select_color_map'],
            options=["Warm Sunset", "Viridis", "Plasma", "Inferno", "Cividis", "Blues", "Purples"],
            index=0  # Default to "Warm Sunset"
        )

    # Check if processed data is available in session state
    if 'processed_data' in st.session_state and st.session_state['processed_data']:
        # Create a list of titles based on the language
        titles = [(k[0] if lang == 'pt' else k[1]) for k in st.session_state['processed_data'].keys()]
        selected_title = st.selectbox(texts[lang]['select_indicator'], titles)

        # Reverse lookup to get the tuple key from the selected title
        selected_key = next(k for k in st.session_state['processed_data'].keys() if selected_title in k)
            
        # Get the DataFrame associated with the selected key
        df = st.session_state['processed_data'][selected_key]
        #st.write(f"Displaying data for: {selected_title}")        
    else:
        # Assume the base path to the 'Preloaded' folder
        # Set the base path to the 'Preloaded' folder
        base_path = Path("Preloaded")

        def get_years():
            return sorted([name for name in os.listdir(base_path) if os.path.isdir(base_path / name)])

        def list_topics(year, lang):
            year_path = base_path / year

            # Define topic keywords based on language
            topic_keywords = {
                'en': [
                    'Population',
                    'Education',
                    'Culture and sports',
                    'Health',
                    'Labour market',
                    'Social Protection',
                    'Income and living conditions'
                ],
                'pt': [
                    'População',
                    'Educação',
                    'Cultura e desporto',
                    'Saúde',
                    'Mercado de trabalho',
                    'Proteção Social',
                    'Rendimento e condições de vida'
                ]
            }[lang]

            # List only directories that match the topic keywords for the selected language
            topics = [d.name for d in year_path.iterdir() if d.is_dir() and d.name in topic_keywords]
            return sorted(topics)

        def get_indicators(year, topic):
            topic_path = base_path / year / topic
            return sorted([f.name for f in topic_path.glob('*.csv')])

        def load_data(file_path):
            return pd.read_csv(file_path)

        # Streamlit User Interface
        year = st.selectbox(texts[lang]['select_year'], get_years())
        topics = list_topics(year, lang)
        topic = st.selectbox(texts[lang]['select_topic'], topics)
        indicators = get_indicators(year, topic)
        selected_indicator = st.selectbox(texts[lang]['select_indicator'], indicators)

        # Build the path to the selected data
        data_path = base_path / year / topic / selected_indicator
        st.write("Loading data from:", data_path)  # Debug statement to check the path

        # Load and display the data
        if data_path.exists():
            df = load_data(data_path)
            df = df.set_index(df.columns[0])
            df.index.name = None
        else:
            st.error("Selected data file does not exist.")
        
    if df is not None:
        column_names = df.columns.tolist()
        column_name = st.selectbox(texts[lang]['select_column'], column_names)

    # Load and prepare geographical data based on selected level
    if level == "Municipal":
        gdf = gpd.read_file('./maps/municipal.shp').to_crs(epsg=4326)
        geo_column = 'NAME_2_cor'
    else:
        gdf = gpd.read_file('./maps/district.shp').to_crs(epsg=4326)
        geo_column = 'NUTIII_DSG'

    # Conditional Multiselect based on selected level
    if level == "Regional":
        all_regions = list(portugal_geo_structure.keys())
        selected_regions = st.multiselect(texts[lang]['select_areas'], all_regions)
        next_level_options = []
        for region in selected_regions:
            next_level_options.extend(portugal_geo_structure[region].keys())
    elif level == "Municipal":
        all_regions = list(portugal_geo_structure.keys())
        selected_regions = st.multiselect(texts[lang]['select_areas'], all_regions)
        district_options = []
        for region in selected_regions:
            for district in portugal_geo_structure[region].keys():
                district_options.append(district)
        # For Municipal level, change multiselect to "Select District"
        selected_districts = st.multiselect(texts[lang]['select_regions'], sorted(set(district_options)))
        next_level_options = []
        for district in selected_districts:
            for region in selected_regions:
                if district in portugal_geo_structure[region]:
                    next_level_options.extend(portugal_geo_structure[region][district])

    # Multiselect for including districts or municipalities
    #included_units = st.multiselect(f"Select {level}s to Include:", sorted(set(next_level_options)))
    included_units = st.multiselect(texts[lang]['select_levels'].format(level=level), sorted(set(next_level_options)))

    # Load and filter geographical data
    geo_data_path = './maps/municipal.shp' if level == "Municipal" else './maps/district.shp'
    gdf = gpd.read_file(geo_data_path).to_crs(epsg=4326)
    geo_column = 'NAME_2_cor' if level == "Municipal" else 'NUTIII_DSG'
    df = df.reset_index()
    df.rename(columns = {'index':'Border'}, inplace = True)

    if included_units:
        gdf = gdf[gdf[geo_column].isin(included_units)]

    merged = gdf.merge(df, how='left', left_on=geo_column, right_on='Border')

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
            "Cividis": plt.get_cmap('cividis'),
            "Blues": plt.get_cmap('Blues'),
            "Purples": plt.get_cmap('Purples')
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
        colorbar.set_label(f"{column_name.replace('_', ' ').title()}", size=8)
        st.pyplot(fig)

    # Saving the map and legend as png
    if st.button('Save Map and Legend as .png'):
        # Save the map
        map_png = m._to_png()
        map_path = "map.png"
        with open(map_path, "wb") as map_file:
            map_file.write(map_png)
        st.success("Map saved as map.png")

        # Save the legend
        legend_path = "legend.png"
        fig.savefig(legend_path, format='png')
        st.success("Legend saved as legend.png")

    # Saving the map and legend as html
    if st.button('Save Map as .html'):
        map_html = './map.html'
        m.save(map_html)

    # After the map display code
    if show_bar_chart:
        axis_orientation = st.selectbox(texts[lang]['choose_bar_orientation'], ['vertical', 'horizontal'], index=0)
        custom_bar_chart_title = st.text_input(texts[lang]['enter_custom_title'], "")
        custom_x_label = st.text_input(texts[lang]['enter_x_axis_label'], "")
        custom_y_label = st.text_input(texts[lang]['enter_y_axis_label'], "")
        # Get the current color map as a string suitable for matplotlib
        plt_cmap_name = {
            "Warm Sunset": "YlOrRd",
            "Viridis": "viridis",
            "Plasma": "plasma",
            "Inferno": "inferno",
            "Cividis": "cividis",
            "Blues": "Blues",
            "Purples": "Purples"
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
