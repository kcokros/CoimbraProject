# preprocess.py

import pandas as pd

# Function to trim mostly empty columns
def trim_mostly_empty_columns(df, threshold=0.9):
    """
    Trims columns from a pandas DataFrame that are mostly empty beyond a specified threshold.
    
    This function operates in-place on the DataFrame to remove columns where the fraction of missing
    values exceeds the provided threshold. It stops removing columns once a column has less than
    the threshold of missing values.

    Parameters:
        df (pandas.DataFrame): The DataFrame from which columns are to be trimmed.
        threshold (float): The fraction of missing values a column must exceed to be removed.
                           Must be a value between 0 and 1.

    Returns:
        pandas.DataFrame: The DataFrame with columns trimmed according to the specified threshold.

    Raises:
        ValueError: If the threshold is not between 0 and 1.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    df.replace("", pd.NA, inplace=True)  # Replace empty strings with NA to count as missing.
    fraction_empty = df.isna().mean()  # Compute the fraction of NA in each column.
    mostly_empty_cols = fraction_empty[fraction_empty > threshold]  # Identify columns to drop.

    if not mostly_empty_cols.empty:
        # Find the first column where the number of empty cells is below the threshold
        # and drop all columns up to and including this column.
        first_mostly_empty_col = mostly_empty_cols.index[0]
        df = df.loc[:, :first_mostly_empty_col].drop(columns=[first_mostly_empty_col])

    return df

def extract_and_preprocess_sheet(xls, sheet_name):
    """
    Extracts titles and preprocesses a specified sheet from a pandas ExcelFile object.
    
    This function reads a specified sheet from the given ExcelFile object, identifies the rows
    containing the titles based on whether the first cell in the sheet is empty or not, and preprocesses
    the DataFrame by trimming mostly empty columns and removing copyright rows.

    Parameters:
        xls (pandas.ExcelFile): The ExcelFile object to read from.
        sheet_name (str): The name of the sheet to process.

    Returns:
        tuple: A tuple containing the processed DataFrame, Portuguese title, and English title.

    Raises:
        ValueError: If the sheet is empty or the titles cannot be located.
    """
    try:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
    except ValueError:
        raise ValueError(f"Failed to read sheet {sheet_name}.")

    df = trim_mostly_empty_columns(df, threshold=0.9)

    # Check if the first entry is empty and determine the title row accordingly
    if pd.isna(df.iloc[0, 0]) or df.iloc[0, 0].strip() == "":
        pt_title = df.iloc[1, 0]
        en_title = df.iloc[2, 0]
        df = df.drop([0, 1, 2])
    else:
        pt_title = df.iloc[0, 0]
        en_title = df.iloc[1, 0]
        df = df.drop([0, 1])

    df.reset_index(drop=True, inplace=True)

    # Locate the row containing the copyright symbol © INE, I.P and truncate the dataframe
    idx = df.iloc[:, 0].str.contains("© INE, I.P", na=False).idxmax()
    if idx != -1:
        df = df.loc[:idx - 1]
    else:
        raise ValueError(f"No copyright row found in sheet {sheet_name}, processing may be incomplete.")

    return df, pt_title, en_title

def generateDataframes(xls, sheet_names):
    """
    Generates a dictionary of DataFrames from specified sheets in an Excel file, based on content criteria.

    This function processes each sheet specified, checks for specific rows containing 'Portugal' and
    'Porto Santo', and preprocesses those that meet the criteria. Each processed sheet is added to a
    dictionary with titles as keys.

    Parameters:
        xls (pandas.ExcelFile): Excel file object to read from.
        sheet_names (list of str): List of sheet names to process.

    Returns:
        tuple: A tuple containing a dictionary of DataFrames and a dictionary of extracted units.
    
    Raises:
        ValueError: If any sheet specified does not exist or does not meet data criteria.
    """
    data_frames = {}
    extracts = {}
    
    for sheet_name in sheet_names:
        try:
            df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        except ValueError:
            print(f"Sheet {sheet_name} not found or could not be read.")
            continue
        
        # Check if the sheet contains rows 'Portugal' and 'Porto Santo'
        start_idx = df[df.apply(lambda row: row.str.contains('Portugal', na=False).any(), axis=1)].index.min()
        end_idx = df[df.apply(lambda row: row.str.contains('Porto Santo', na=False).any(), axis=1)].index.max()

        if pd.notna(start_idx) and pd.notna(end_idx) and start_idx < end_idx:
            df, pt_title, en_title = extract_and_preprocess_sheet(xls, sheet_name)
            print(f"Sheet {sheet_name} is processed")
            data_frames[(pt_title, en_title)] = df
        else:
            print(f"Sheet {sheet_name} does not contain required rows and it's excluded.")
            continue

    extracts = generateUnits(data_frames)

    for key in data_frames:
        if extracts[key] != '':
            data_frames[key].drop(index=data_frames[key].index[0], axis=0, inplace=True)
            data_frames[key].reset_index(drop=True, inplace=True)

    return data_frames, extracts

def generateUnits(data_frames):
    """
    Extracts units from the first row of each DataFrame in the provided dictionary.

    This function checks the first cell in the first row of each DataFrame for a string indicating
    the unit of measurement (prefixed by 'Unidade: '). If found, it extracts and stores the unit.

    Parameters:
        data_frames (dict): A dictionary of DataFrames, keyed by titles.

    Returns:
        dict: A dictionary containing the extracted units for each DataFrame key.

    Notes:
        Assumes that if a unit is present, it is formatted as 'Unidade: [unit]' in the first cell
        of the DataFrame. If no unit is found, the corresponding value in the return dictionary is
        an empty string.
    """
    generated_units = {}
    for key in data_frames:
        df = data_frames[key]
        try:
            value = df.iloc[0, 0]  # Accessing the first cell of the DataFrame
            if isinstance(value, str) and 'Unidade: ' in value:
                # Extract the unit of measurement
                generated_units[key] = value.split('Unidade: ')[1]
                print(f'Unit extracted for {key}: {generated_units[key]}')
            else:
                generated_units[key] = ''
                print(f'No unit found for {key}.')
        except IndexError:
            # Handling possible index errors if DataFrame is empty or improperly formatted
            generated_units[key] = ''
            print(f'Failed to extract unit for {key}: DataFrame may be empty or incorrectly formatted.')

    return generated_units

def findLimits(df):
    """
    Finds the indices of the rows labeled 'Portugal' and 'Porto Santo' in a DataFrame.

    This function searches for the first occurrences of 'Portugal' and 'Porto Santo' in the
    first column of the DataFrame and returns their indices. If either is not found, None is returned for that label.

    Parameters:
        df (pandas.DataFrame): The DataFrame to search.

    Returns:
        tuple: A tuple containing the index of 'Portugal' and the index of 'Porto Santo'.
               Each element is None if the respective label is not found.

    Raises:
        ValueError: If the DataFrame is empty or if the first column is not suitable for searching (e.g., not a string type).
    """
    if df.empty:
        raise ValueError("The DataFrame is empty.")
    
    try:
        # Check data type consistency in the first column
        if df.iloc[:, 0].dtype.kind not in 'OU':
            raise ValueError("The first column is not of a string type suitable for searching.")

        # Find the index of the first occurrence of 'Portugal'
        portugal_index = df.index[df.iloc[:, 0] == 'Portugal'].tolist()

        # Find the index of the first occurrence of 'Porto Santo'
        porto_santo_index = df.index[df.iloc[:, 0] == 'Porto Santo'].tolist()

        return portugal_index[0] if portugal_index else None, porto_santo_index[0] if porto_santo_index else None

    except Exception as e:
        # General exception to catch unexpected issues
        raise ValueError("An error occurred while searching the DataFrame.") from e

def find_limits_for_all(dataframes):
    """
    Finds the indices of 'Portugal' and 'Porto Santo' for all DataFrames in a dictionary.

    This function iterates through a dictionary of DataFrames, applying the `findLimits`
    function to each to identify the indices of rows labeled 'Portugal' and 'Porto Santo'.
    The results are stored in two dictionaries that map DataFrame keys to their respective
    indices for 'Portugal' and 'Porto Santo'.

    Parameters:
        dataframes (dict): A dictionary of DataFrames keyed by an identifier (e.g., title tuple).

    Returns:
        tuple of dicts: Two dictionaries, the first mapping keys to the index of 'Portugal' and
                        the second mapping keys to the index of 'Porto Santo'.

    Notes:
        If a DataFrame does not contain the rows for 'Portugal' or 'Porto Santo', or if an error
        occurs during processing, None is returned for that entry.
    """
    lower_bound = {}
    upper_bound = {}

    for key, dataframe in dataframes.items():
        try:
            # Apply findLimits to each DataFrame, ensuring any issues are caught and handled.
            lower_bound[key], upper_bound[key] = findLimits(dataframe)
        except ValueError as e:
            # Handle the case where findLimits raises a ValueError, typically due to an empty DataFrame or incorrect types.
            print(f"Error finding limits in DataFrame {key}: {e}")
            lower_bound[key] = None
            upper_bound[key] = None

    return lower_bound, upper_bound


def fillRowsInRangeForAll(dataframes, lower_limits, upper_limits):
    """
    Fills null values in specified ranges for each DataFrame in a given dictionary.

    This function applies forward filling of null values for each DataFrame based on specified
    lower and upper limits. Null values before the upper limit and after the lower limit are filled.
    The operation is performed in a copy of the original DataFrame to prevent modifications to the original data.

    Parameters:
        dataframes (dict): A dictionary of pandas DataFrames, keyed by identifiers.
        lower_limits (dict): A dictionary of lower limit indices for each DataFrame.
        upper_limits (dict): A dictionary of upper limit indices for each DataFrame.

    Returns:
        dict: A dictionary containing the modified DataFrames with null values filled within the specified ranges.

    Notes:
        If either the lower or upper limit for a DataFrame is None, no filling is performed for that limit.
    """
    filled_dataframes = {}
    for key, df in dataframes.items():
        if df.empty:
            print(f"Warning: DataFrame with key {key} is empty. No filling applied.")
            filled_dataframes[key] = df
            continue

        lower_limit = lower_limits.get(key)
        upper_limit = upper_limits.get(key)

        filled_df = df.copy()  # Make a copy to avoid modifying the original DataFrame

        try:
            # Fill null values before upper limit if the upper limit is provided and valid
            if upper_limit is not None and upper_limit < len(df):
                filled_df.iloc[:upper_limit, :] = filled_df.iloc[:upper_limit, :].fillna(method='ffill', axis=1)
            # Fill null values after lower limit if the lower limit is provided and valid
            if lower_limit is not None and lower_limit + 1 < len(df):
                filled_df.iloc[lower_limit + 1:, :] = filled_df.iloc[lower_limit + 1:, :].fillna(method='ffill', axis=1)
        
        except Exception as e:
            print(f"Error filling DataFrame with key {key}: {e}")
            filled_dataframes[key] = df  # Store the original DataFrame in case of failure
            continue

        filled_dataframes[key] = filled_df

    return filled_dataframes

def concatenateRowsWithinLimits(dataframes, lower_limits, upper_limits, unit_values):
    """
    Concatenates new rows within specified limits for each DataFrame in a given dictionary.

    For each DataFrame, creates two new rows: one summarizing data above the lower limit
    and one below the upper limit, then concatenates these at the respective positions.
    Special symbols in values are replaced and units are appended if applicable.

    Parameters:
        dataframes (dict): A dictionary of pandas DataFrames, keyed by identifiers.
        lower_limits (dict): A dictionary mapping keys to lower limit indices for each DataFrame.
        upper_limits (dict): A dictionary mapping keys to upper limit indices for each DataFrame.
        unit_values (dict): A dictionary mapping keys to unit strings for each DataFrame.

    Notes:
        Modifies the DataFrames in-place within the passed dictionary.
    """
    for key, df in dataframes.items():
        if df.empty:
            print(f"Warning: DataFrame with key {key} is empty. No operations performed.")
            continue

        lower_limit = lower_limits.get(key)
        upper_limit = upper_limits.get(key)
        unit = unit_values.get(key)

        new_row_below_upper = None
        new_row_above_lower = None

        try:
            # Create a new row summarizing data below the upper limit if it exists and is within bounds
            if upper_limit is not None and upper_limit < len(df):
                new_row_below_upper = {'Municipalities': 'Municipalities'}
                for col_idx, col_name in enumerate(df.columns):
                    if col_idx != 0:
                        values_after_upper = ', '.join(str(val) for val in df.iloc[upper_limit + 1:, col_idx].dropna())
                        new_row_below_upper[col_name] = values_after_upper + (f" ({unit})" if unit else "")

            # Create a new row summarizing data above the lower limit if it exists and is within bounds
            if lower_limit is not None and lower_limit > 0:
                new_row_above_lower = {'Municipios': 'Municipios'}
                for col_idx, col_name in enumerate(df.columns):
                    if col_idx != 0:
                        values_before_lower = ', '.join(str(val) for val in df.iloc[:lower_limit, col_idx].dropna())
                        new_row_above_lower[col_name] = values_before_lower + (f" ({unit})" if unit else "")

            # Append new rows and adjust DataFrame
            if new_row_below_upper:
                df = pd.concat([pd.DataFrame([new_row_below_upper], columns=df.columns), df], ignore_index=True)
            if new_row_above_lower:
                df = pd.concat([df, pd.DataFrame([new_row_above_lower], columns=df.columns)], ignore_index=True)

        except Exception as e:
            print(f"Error processing DataFrame with key {key}: {e}")
            continue

        dataframes[key] = df

    return dataframes


def replaceNewlineWithSpace(dataframes):
    """
    Replaces newline characters with spaces in the first two rows of each DataFrame in a dictionary.

    This function modifies the DataFrames in-place, specifically targeting the first two rows,
    which are often used for headers or important descriptive information.

    Parameters:
        dataframes (dict): A dictionary of pandas DataFrames, keyed by identifiers.

    Returns:
        dict: The dictionary of DataFrames after replacing newline characters in the first two rows.

    Notes:
        Modifies the DataFrames in-place within the passed dictionary.
    """
    for key, df in dataframes.items():
        if df.empty:
            print(f"Warning: DataFrame with key {key} is empty. No newline replacement performed.")
            continue

        try:
            # Ensure there are at least two rows to process
            max_rows = min(2, len(df))
            for idx in range(max_rows):
                # Replace '\n' with ' ' in each cell of the row if it's a string
                df.iloc[idx] = df.iloc[idx].apply(lambda x: x.replace('\n', ' ') if isinstance(x, str) else x)
        
        except Exception as e:
            print(f"Error processing DataFrame with key {key}: {e}")
            continue

        dataframes[key] = df

    return dataframes

def addMultiindexColumns(dataframes, lang):
    """
    Converts the first two rows of each DataFrame into a MultiIndex for the columns,
    sets the first column as the index, and updates the DataFrames in place.

    Parameters:
        dataframes (dict): A dictionary of pandas DataFrames, keyed by identifiers.

    Returns:
        dict: The dictionary of DataFrames with MultiIndex columns and updated indices.

    Raises:
        ValueError: If any DataFrame does not have at least two rows or the required columns to set as MultiIndex.
    """
    for key, df in dataframes.items():
        if df.empty:
            print(f"Warning: DataFrame with key {key} is empty. No MultiIndex conversion performed.")
            continue

        if len(df) < 2:
            raise ValueError(f"DataFrame with key {key} does not have enough rows to set a MultiIndex.")

        try:
            # Use the first two rows to create a MultiIndex for the columns
            multiindex_columns = pd.MultiIndex.from_arrays(df.iloc[0:2].values, names=["Portuguese", "English"])
            df = df.iloc[2:]  # Remove the rows used for the MultiIndex
            df.columns = multiindex_columns

            # Set the first column as the DataFrame index and remove its name
            df.iloc[:, 0] = df.iloc[:, 0].str.lstrip() # remove empty spaces before string
            df = df.set_index(df.columns[0])
            df.index.name = None
            
            if lang == 'en':
                df.columns = df.columns.get_level_values(0)  # English titles
            else:
                df.columns = df.columns.get_level_values(1)  # Portuguese titles

            dataframes[key] = df.reset_index(drop=True)

            dataframes[key] = df
        except Exception as e:
            print(f"Error setting MultiIndex in DataFrame with key {key}: {e}")
            continue

    return dataframes

def refineHeaders(dataframes):
    """
    Refines each DataFrame by replacing the second row with the last row and removing rows with empty cells at the start.

    Parameters:
        dataframes (dict): A dictionary of pandas DataFrames, keyed by identifiers.

    Returns:
        dict: The dictionary of refined DataFrames.
    """
    for key, df in dataframes.items():
        if df.empty:
            print(f"Warning: DataFrame with key {key} is empty. No refinement performed.")
            continue

        # Replace the second row with the last row
        df.iloc[1] = df.iloc[-1]

        # Remove the duplicated last row since it's now moved to the second position
        df = df.drop(df.index[-1])

        # Remove all rows that have an empty cell in the first column, except for the first two rows
        df = df.drop(df[(df.iloc[:, 0].isna()) & (df.index > 1)].index)

        # Update the DataFrame in the dictionary after modifications
        dataframes[key] = df

    return dataframes


if __name__ == "__main__":
    # Code here will only run if this script is executed directly, and not when imported.
    # This can be used for testing the functions within this module.
    print("Running module tests...")
    # Example: Load an Excel file and test the functions
    test_path = 'path_to_test_excel_file.xlsx'
    test_xls = pd.ExcelFile(test_path)
    test_sheet_name = 'Sheet1'
    df, pt_title, en_title = extract_and_preprocess_sheet(test_xls, test_sheet_name)
    print(f"Processed DataFrame: {df.head()}")
    print(f"Portuguese Title: {pt_title}, English Title: {en_title}")
