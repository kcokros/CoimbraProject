import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import folium_static, st_folium
from io import BytesIO
from folium.plugins import FloatImage

# Function to process each sheet in the Excel file based on language selection
def process_sheet(xls, sheet_name, language='English'):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    # English names are in the first row, Portuguese in the last row
    eng_columns = df.iloc[0]
    pt_columns = df.iloc[-1]
    df.columns = eng_columns
    df = df[1:-1]  # Drop the rows with both English and Portuguese names
    df = df.loc[:, :(df.isnull().all().cumsum() == 1).idxmax()]
    df.dropna(axis=1, how='all', inplace=True)
    return df, eng_columns, pt_columns if language == 'English' else pt_columns

# Function to save dataframe to a CSV and return it as a download link
def to_csv(df):
    output = BytesIO()
    df.to_csv(output, index=False, sep=';', encoding='utf-8')
    output.seek(0)
    return output

# Streamlit app layout
language = st.sidebar.selectbox("Choose Language", ["English", "Portuguese"])
st.sidebar.image("https://i.postimg.cc/hjT72Vcx/logo-black.webp", use_column_width=True)
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
##############################################################################################
elif page == "Interactive Map":
    st.title("Interactive Map")

    # Function to load and process data from Excel files
    def load_and_process_data(file_path, language='English'):
        try:
            df = pd.read_excel(file_path)
            eng_columns = df.iloc[0]  # English names are in the first row
            pt_columns = df.iloc[-1]  # Portuguese names are in the last row
            df = df[1:-1]  # Drop the rows with English and Portuguese names
            df.columns = eng_columns  # Temporarily use English for processing
            df = df.loc[:, :(df.isnull().all().cumsum() == 1).idxmax()]
            df.dropna(axis=1, how='all', inplace=True)
            return df, eng_columns, pt_columns
        except Exception as e:
            st.error(f"Failed to load or process the file: {e}")
            return pd.DataFrame(), [], []

    # Load and process the shapefiles/data
    choropleth_gdf = gpd.read_file('./maps/municipal.shp').to_crs(epsg=4326)
    area_type_gdf = gpd.read_file('./maps/AreasEdificadas2018.shp').set_crs(epsg=3763).to_crs(epsg=4326)

    # Define labels and colors for mapping
    tipo_p_labels = {
        1: "Residential (>= 10 Buildings)",
        2: "Residential scattered/isolated",
        3: "Non-Residential"
    }
    area_type_gdf['TIPO_p_label'] = area_type_gdf['TIPO_p'].map(tipo_p_labels)
    tipo_p_colors = {
        "Residential (>= 10 Buildings)": "red",
        "Residential scattered/isolated": "orange",
        "Non-Residential": "purple"
    }

    # Sidebar options for Choropleth
    year = st.sidebar.slider("Select Year", 2020, 2021, 2022, 2023)
    df_path = f'tables/{year}.xlsx'
    df, eng_columns, pt_columns = load_and_process_data(df_path, language)
    column_names = eng_columns if language == 'English' else pt_columns
    column_name = st.sidebar.selectbox("Select Column", column_names[5:])  # Assuming the first 5 columns are metadata or not required

    # Attempt to merge and catch any KeyError
    df.rename(columns={'Portugal': 'Region'}, inplace=True)
    try:
        merged = choropleth_gdf.merge(df, left_on='NAME_2_cor', right_on='Region')
        merged[column_name] = pd.to_numeric(merged[column_name], errors='coerce').fillna(0)
        bin_edges = calculate_quantile_bins(merged[column_name])

        m = folium.Map(location=[39.5, -8.0], zoom_start=7)
        folium.Choropleth(
            geo_data=merged,
            data=merged,
            columns=['NAME_2_cor', column_name],
            key_on='feature.properties.NAME_2_cor',
            fill_color='YlGn',
            fill_opacity=0.4,
            line_opacity=0.2,
            legend_name=f'{column_name} in {year}',
            bins=bin_edges,
            reset=True
        ).add_to(m)
        
        # Add GeoJson for selected area types if any
        selected_tipo_p = st.sidebar.multiselect("Select Residential Density", list(tipo_p_labels.values()), default=[])
        filtered_area_type_gdf = area_type_gdf[area_type_gdf['TIPO_p_label'].isin(selected_tipo_p)]
        if not filtered_area_type_gdf.empty:
            folium.GeoJson(
                filtered_area_type_gdf,
                style_function=lambda feature: {
                    'fillColor': tipo_p_colors[feature['properties']['TIPO_p_label']],
                    'color': 'black',
                    'weight': 0,
                    'fillOpacity': 0.6
                },
                tooltip=folium.GeoJsonTooltip(fields=['TIPO_p_label'], aliases=['Area Type:'])
            ).add_to(m)

        # Display the map
        st_folium(m, width=900, height=700)
    except KeyError as e:
        st.error(f"Failed to find the key in the DataFrame: {str(e)}")
        st.write("Available columns in choropleth_gdf:", choropleth_gdf.columns.tolist())
        st.write("Available columns in df:", df.columns.tolist())

    folium.Choropleth(
        geo_data=merged,
        data=merged,
        columns=['NAME_2_cor', column_name],
        key_on='feature.properties.NAME_2_cor',
        fill_color='YlGn',
        fill_opacity=0.4,
        line_opacity=0.2,
        legend_name=f'{column_name} in {year}',
        bins=bin_edges,
        reset=True
    ).add_to(m)

    # Sidebar options for Area Type Map with no default selection
    st.sidebar.markdown("## Coimbra Region-only Area Options")
    tipo_p_options = area_type_gdf['TIPO_p_label'].unique()
    selected_tipo_p = st.sidebar.multiselect("Select Residential Density", tipo_p_options, default=[])

    # Filter based on selected TIPO_p labels
    filtered_area_type_gdf = area_type_gdf[area_type_gdf['TIPO_p_label'].isin(selected_tipo_p)]

    # Adding Area Type GeoJSON on top of the Choropleth Map
    if not filtered_area_type_gdf.empty:
        folium.GeoJson(
            filtered_area_type_gdf,
            style_function=lambda feature: {
                'fillColor': tipo_p_colors[feature['properties']['TIPO_p_label']],
                'color': 'black',
                'weight': 0,
                'fillOpacity': 0.6
            },
            tooltip=folium.GeoJsonTooltip(fields=['TIPO_p_label'], aliases=['Area Type:'])
        ).add_to(m)

    # Display the map
    st_folium(m, width=900, height=700)


##############################################################################################
elif page == "Forecast":
    st.title("Forecast")
