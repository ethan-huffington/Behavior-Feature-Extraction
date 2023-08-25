#=======================================================================
#                               IMPORTS
#=======================================================================

import boto3
import botocore
import numpy as np
import pandas as pd
import datetime as dt
import re
import random
from io import StringIO

# Typehints
from pandas import DataFrame 
from typing import Any 
from typing import Dict, List 

#=======================================================================
#                               DATA HANDLING
#=======================================================================

def load_to_dict(s3_url, dtype_dict):
    s3 = boto3.resource('s3')

    # Parse the bucket name and prefix from the URL
    bucket_name, prefix = s3_url.replace("s3://", "").split('/', 1)
    bucket = s3.Bucket(bucket_name)

    dfs = {}  # dictionary to hold the dataframes

    for obj in bucket.objects.filter(Prefix=prefix):
        # Only process CSV files
        if obj.key.endswith('.csv'):
            # Get the file name without the extension
            filename = obj.key.split('/')[-1].split('.')[0]

            # Create a presigned URL to download the file
            url = boto3.client('s3').generate_presigned_url(
                ClientMethod='get_object',
                Params={'Bucket': bucket_name, 'Key': obj.key},
                ExpiresIn=3600)

            # Load the data from the file
            df = pd.read_csv(url)

            # Convert the dtypes of the columns based on the specified mapping
            for current_dtype, target_dtype in dtype_dict.items():
                # Select the columns that have the current dtype
                cols = df.select_dtypes(include=[current_dtype]).columns
                # Convert the dtypes of these columns to the target dtype
                df[cols] = df[cols].astype(target_dtype)

            # Add the dataframe to the dictionary with the filename as the key
            dfs[filename] = df

    return dfs

def get_file_paths(bucket_name, prefix):
    s3 = boto3.client('s3')
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    
    # Check if the prefix is empty
    if response.get('Contents'):
        # Get the S3 paths of the files
        file_paths = ['s3://' + bucket_name + '/' + item['Key'] for item in response['Contents']]
    else:
        file_paths = []
    
    return file_paths

def create_stim_session_list(df):
    temp_ID_list = []
    temp_test_list = []

    for index, row in df.iterrows():
        # Get current animal ID
        ID = str(row['animal_id_1'])
        ID2 = str(row['animal_id_2'])

        # Find Pretest (S01) Value
        pretest_df = df.loc[(df['animal_id_1'] == ID) & (df['animal_id_2'] == ID2) & (df['session_num'] == 'S01'), 'stim_zone']
        pretest = pretest_df.values[0] if not pretest_df.empty else 'XXX'

        # Find Acquisition (S02) Value
        acquisition_df = df.loc[(df['animal_id_1'] == ID) & (df['animal_id_2'] == ID2) & (df['session_num'] == 'S02'), 'stim_zone']
        acquisition = acquisition_df.values[0] if not acquisition_df.empty else 'XXX'

        # Find Reversal (S03) Value
        reversal_df = df.loc[(df['animal_id_1'] == ID) & (df['animal_id_2'] == ID2) & (df['session_num'] == 'S03'), 'stim_zone']
        reversal = reversal_df.values[0] if not reversal_df.empty else '-'

        # Add test strings to one
        concat = pretest + acquisition + reversal

        temp_ID_list.append(ID)
        temp_test_list.append(concat)

    df['stim_session_list'] = temp_test_list

    return df

def view_rand_df(df_dict, info_df):
    
    key, value = random.choice(list(df_dict.items()))
    
    print(f"Dataframe: {key}")
    print(f"stim_session_list: {info_df[info_df['session_id'] == key]['stim_session_list'].values[0]}")
    return value.head(3)

def remove_columns(df_dictionary, col_removal_list):
    removed_cols_summary = {}  # Dictionary to store removed columns summary

    # Iterate over each dataframe in the dictionary
    for key, df in df_dictionary.items():
        # Find the intersection of columns in df and col_removal_list
        cols_to_remove = set(df.columns) & set(col_removal_list)

        if cols_to_remove:
            # Remove the columns
            df.drop(columns=list(cols_to_remove), inplace=True)

            # Store the removed columns in the summary
            removed_cols_summary[key] = list(cols_to_remove)

        # Replace the old df with the new one in the dictionary
        df_dictionary[key] = df

    return df_dictionary, removed_cols_summary

def format_cols(df_dict, col_rename_dict):
    formatted_df_dict = {}

    # Iterate over each dataframe in the input dictionary
    for key, df in df_dict.items():
        # Create a new dataframe with the same data as df
        formatted_df = df.copy()

        # Convert all column names to lower case and replace spaces with underscores
        formatted_df.columns = [col.lower().replace(' ', '_') for col in df.columns]
        
        # Apply column renaming
        formatted_df.rename(columns=col_rename_dict, inplace=True)
        
        # Add the formatted dataframe to the new dictionary
        formatted_df_dict[key] = formatted_df

    return formatted_df_dict

def remove_numbers_from_columns(df_dict):
    # Create a new dictionary to store the processed dataframes
    cleaned_df_dict = {}
    
    # Iterate over each dataframe in the input dictionary
    for key, df in df_dict.items():
        # Create a new dataframe with the same data as df
        cleaned_df = df.copy()
        
        # Remove numbers from the column names
        cleaned_df.columns = [re.sub(r'\d+', '', col) for col in df.columns]
        
        # Add the cleaned dataframe to the new dictionary
        cleaned_df_dict[key] = cleaned_df
    
    return cleaned_df_dict

#=======================================================================
#                               TRACE CREATION
#=======================================================================

def zone_entry(df_dict, info_df, stim_col_idx: int = 7):
  returned_dict = {}

  for key, df in df_dict.items():
    # Get Indicies of Zone Entry (Returns a Tuple containing an array)
    indicies = np.where((df.iloc[:,stim_col_idx] == True) & (df.iloc[:,stim_col_idx].shift(1) == False))

    # Assert that indicies are found
    assert len(indicies[0]) != 0, f"No indices found for key: {key}"

    # Index into the idx-containing tuple and convert to list
    indicies = list(indicies[0])

    # Make a list of new df's that have been split on specified indicies
    new_frames = np.array_split(df,indicies)

    # Create 'trace_num' and 'session_id' columns
    new_frames = [frame.assign(trace_num=i, session_id=key) for i, frame in enumerate(new_frames)]

    # Initialize an empty list to store the filtered dataframes
    new_frames_trimmed = []

    for trace_df in new_frames:
      trace_df_trimmed = trace_df[trace_df.iloc[:,stim_col_idx] == False]
      if not trace_df_trimmed.empty and trace_df.shape[0] > 1:
        new_frames_trimmed.append(trace_df_trimmed)

    # Assert that new_frames_trimmed is not empty
    assert len(new_frames_trimmed) != 0, f"No dataframes were added to new_frames_trimmed for key: {key}"

    # load df lists in place of df's
    returned_dict[key] = new_frames_trimmed # Load dict with list of trace df's

  return returned_dict



def trim_to_5(dict_list_df: Dict[str,List],
              stim_col_idx: int = 7,
              time_col_idx:int = 0) -> Dict[str,List]:
  """
    Trims down df's to five seconds

    - From entry event, walk back timedelta = 5 seconds

    Args:
        stim_col_idx: Index position of stim column
        dict_list_df: A DICT of LISTS of DFs

    Returns:
        Same DICT of LISTS of DF but with trimmed df lenghts
       Dict
          ├── list [df1, df2, .........dfn]
          ├── list [df1, df2, .........dfn]
          └── list [df1, df2, .........dfn]
  """
  new_dict = {}

  for key, df_list in dict_list_df.items():

    new_df_list = []

    for df in df_list:

      #Casting stim column as Integer dtype
      df.iloc[:,stim_col_idx] = df.iloc[:,stim_col_idx].astype(int)

      # Casting time column as datetime dtype
      df.iloc[:,time_col_idx] = pd.to_datetime(df.iloc[:,time_col_idx], unit='s')

      # Threshold Time (In Seconds)
      threshold = dt.timedelta(0,5)

      # Elapsed Time
      elapsed_time = (df.iloc[:,time_col_idx].iloc[-1]) - (df.iloc[:,time_col_idx])

      # Get idx for 5 seconds elapsed time
      five_sec_idx = np.where( elapsed_time < threshold)

      # Create new df, filtered on 5 sec idx
      trimmed_df = df.iloc[five_sec_idx].copy()
        
      #Casting stim column back to bool
      trimmed_df.iloc[:,stim_col_idx] = trimmed_df.iloc[:,stim_col_idx].astype(bool)

      # Update df list
      if trimmed_df.shape[0] >= 5: # Filter out all dfs with fewer than 5 rows
          new_df_list.append(trimmed_df)

    new_dict[key] =  new_df_list

  return new_dict

def interpolate_trace(trace_df: DataFrame,
                      centerX_col_idx:int = 1,
                      centerY_col_idx: int = 2,
                      time_col_idx: int = 0) -> DataFrame:
    """
    Takes in a DataFrame containing trace data and returns a 5x linearly interpolated DataFrame between each datapoint.

    Args:
        trace_df (DataFrame): A DataFrame containing trace data with columns 'Time', 'Centre_Position_X', and 'Centre_Position_Y'.
                              'Time' column should be of datetime type.

    Returns:
        DataFrame: A DataFrame containing the 5x linearly interpolated trace data with columns 'Centre_Position_X', 'Centre_Position_Y', and 'Time_d'.
                   'Time_d' column represents the time difference in seconds between each row and the maximum time value in the input DataFrame.
    """

    # Create Time Delta Col
    trace_df['Time_d'] = (trace_df.iloc[:, time_col_idx].max()) - (trace_df.iloc[:, time_col_idx])

    # Convert Time Delta column to seconds and round to two decimal places
    trace_df['Time_d'] = round(trace_df['Time_d'].dt.total_seconds(), 2)

    # Grab 'Time_d' column index
    time_d_idx = trace_df.columns.get_loc('Time_d')

    interp_cols = trace_df.iloc[:, [centerX_col_idx, centerY_col_idx, time_d_idx]]

    def interpolate_n_times(df: DataFrame, n: int) -> DataFrame:
        for _ in range(n):
            df = pd.concat([df, pd.DataFrame(index=df.index)]).sort_index(kind='stable', ignore_index=True)
            df = df.interpolate(method='linear')
        return df

    # Apply 5x interpolation
    result = interpolate_n_times(interp_cols, 5)

    return result


def remove_columns_from_dict(df_dict, columns_to_remove):
    """
    Removes the specified columns from all DataFrames in a dictionary of lists of DataFrames.

    Args:
        df_dict (dict): Dictionary of lists of DataFrames.
        columns_to_remove (list): List of column names to remove.

    Returns:
        dict: Dictionary with the same structure, but with the specified columns removed from all DataFrames.
    """
    # Create a new dictionary to store the modified DataFrames
    new_dict = {}
    
    # Iterate through the keys and lists of DataFrames in the original dictionary
    for key, df_list in df_dict.items():
        # Iterate through the DataFrames in the list and remove the specified columns
        new_list = [df.drop(columns=columns_to_remove, errors='ignore') for df in df_list]
        # Add the modified list of DataFrames to the new dictionary
        new_dict[key] = new_list

    return new_dict

#=======================================================================
#                               EMBEDDING & CLUSTERING
#=======================================================================

def extract_features_batch(s3_paths, model, s3_client, bucket_name, batch_size=32):
    all_batches = []
    
    batch_features = []
    for idx, s3_path in tqdm(enumerate(s3_paths), total=len(s3_paths), desc="Processing Images"):
        obj = s3_client.get_object(Bucket=bucket_name, Key=s3_path)
        img_data = obj['Body'].read()
        
        # Load the image as a 224x224 array
        img = Image.open(BytesIO(img_data)).resize((224, 224))
        
        # Convert from 'PIL.Image.Image' to numpy array
        img = np.array(img)

        # Ensure the image is RGB format
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        # Check the shape
        if img.shape != (224, 224, 3):
            print(f"Unexpected shape for {s3_path}: {img.shape}")
            return None

        # Reshape and preprocess as before
        reshaped_img = img.reshape(1, 224, 224, 3)
        imgx = preprocess_input(reshaped_img)
        features = model.predict(imgx, use_multiprocessing=True)
        
        batch_features.append(features)

        # If batch is full or we're at the last image, process the batch
        if (idx + 1) % batch_size == 0 or idx == len(s3_paths) - 1:
            batch_array = np.vstack(batch_features)
            batch_df = create_dataframe(batch_array, s3_paths, start_idx=idx+1-len(batch_features), end_idx=idx)
            all_batches.append(batch_df)
            
            # Clear the batch
            batch_features.clear()

    return all_batches

def create_dataframe(batch_array, s3_paths, start_idx, end_idx):
    df = pd.DataFrame(batch_array)
    df['s3_path'] = s3_paths[start_idx:end_idx+1]
    return df


#=======================================================================
#                               ANALYSIS
#=======================================================================

def optimize_dataframe_dtypes(df, inplace=False):
    """
    Optimize the data types of a pandas DataFrame.
    
    Parameters:
    - df: The original DataFrame.
    - inplace: If True, modify the DataFrame in place. Otherwise, return a new DataFrame.
    
    Returns:
    - Optimized DataFrame (if inplace is False).
    """
    
    # Helper function to downcast numeric columns to the smallest possible dtype
    def downcast_numeric(series):
        if pd.api.types.is_float_dtype(series.dtype):
            return pd.to_numeric(series, downcast="float")
        elif pd.api.types.is_integer_dtype(series.dtype):
            return pd.to_numeric(series, downcast="integer")
        return series
    
    # Helper function to downcast object columns to category if appropriate
    def downcast_object(series):
        # Only downcast object type if unique values are less than half the total length
        if pd.api.types.is_object_dtype(series.dtype) and len(series.unique()) / len(series) < 0.5:
            return series.astype("category")
        return series
    
    # Apply the downcasting
    optimized_df = df.copy(deep=not inplace)
    optimized_df = optimized_df.apply(downcast_numeric).apply(downcast_object)

    if inplace:
        df[optimized_df.columns] = optimized_df
    else:
        return optimized_df

def concatenate_csvs_from_s3(bucket_name, prefix):
    """
    Concatenate multiple CSV files from an S3 directory into a single DataFrame.
    
    Parameters:
    - bucket_name: Name of the S3 bucket.
    - prefix: Directory path in the bucket containing the CSV files.
    
    Returns:
    - A single concatenated and memory-optimized DataFrame.
    """
    
    # Initialize the s3 client
    s3 = boto3.client('s3')
    
    # List all files in the directory
    paginator = s3.get_paginator('list_objects_v2')
    csv_paths = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if 'Contents' in page:
            csv_paths.extend([item['Key'] for item in page['Contents'] if item['Key'].endswith('.csv')])
    
    dataframes = []
    for file_key in csv_paths:
        csv_obj = s3.get_object(Bucket=bucket_name, Key=file_key)
        body = csv_obj["Body"].read().decode('utf-8')
        df = pd.read_csv(StringIO(body))
        optimize_dataframe_dtypes(df, inplace=True)
        dataframes.append(df)
    
    # Concatenate all dataframes into a single one
    final_df = pd.concat(dataframes, ignore_index=True)
    
    return final_df
    
#     return result
def tally_transitions(df, T_COL_IDX, D_COL_IDX, H_COL_IDX, V_COL_IDX):
    # Create a dictionary to map column indexes to their respective new column names
    col_mapping = {
        T_COL_IDX: 't_total',
        D_COL_IDX: 'd_total',
        H_COL_IDX: 'h_total',
        V_COL_IDX: 'v_total'
    }
    
    for col_idx, new_col in col_mapping.items():
        col_name = df.columns[col_idx]
        # Check for transitions from False to True
        df[new_col] = (df[col_name] == True) & (df[col_name].shift(1) == False)
        # Convert the boolean to int (True to 1 and False to 0)
        df[new_col] = df[new_col].astype(int)
    
    return df

def calculate_trace_length(df, time_col_idx, x_col_idx, y_col_idx, trace_num_idx, session_id_idx):
    # Calculate the Euclidean distance
    df['distance'] = ((df.iloc[:, x_col_idx] - df.iloc[:, x_col_idx].shift())**2 + 
                      (df.iloc[:, y_col_idx] - df.iloc[:, y_col_idx].shift())**2)**0.5
    
    # Setting the first row's distance to zero for each combination of trace_num and session_id
    df['first_row'] = df.groupby([df.iloc[:, session_id_idx], df.iloc[:, trace_num_idx]]).cumcount() == 0
    df.loc[df['first_row'], 'distance'] = 0
    df.drop(columns=['first_row'], inplace=True)  # Dropping the helper column
    
    # Group by session_id and trace_num and sum the distances
    result = df.groupby([df.iloc[:, session_id_idx], df.iloc[:, trace_num_idx]])[['distance']].sum()
    result.rename(columns={'distance': 'trace_length'}, inplace=True)
    result.reset_index(inplace=True)
    
    return result

def calculate_area_span(df, x_col_idx, y_col_idx, trace_num_idx, session_id_idx):
    # Group by session_id and trace_num and calculate the difference between max and min for x and y
    aggregated = df.groupby([df.iloc[:, session_id_idx], df.iloc[:, trace_num_idx]])\
                   .apply(lambda group: pd.Series({
                       'x_span': group.iloc[:, x_col_idx].max() - group.iloc[:, x_col_idx].min(),
                       'y_span': group.iloc[:, y_col_idx].max() - group.iloc[:, y_col_idx].min()
                   })).reset_index()
    
    # Calculate the "area span" as the product of x_span and y_span
    aggregated['xy_area_span'] = (aggregated['x_span'] * aggregated['y_span']) / 100
    
    # Drop the intermediate columns x_span and y_span
    aggregated.drop(columns=['x_span', 'y_span'], inplace=True)
    
    return aggregated

