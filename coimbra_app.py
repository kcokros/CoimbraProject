import streamlit as st
import pandas as pd
import numpy as np
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
    st.write("Interactive Map will be implemented here.")

elif page == "Forecast":
    st.title("Forecast")
    st.write("Forecast will be implemented here.")

elif page == "Forecast":
    st.title("Forecast")
    st.write("Forecast will be implemented here.")
