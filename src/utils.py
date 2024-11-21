import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_feature_names(encoder, categorical_cols):
    """
    Get feature names from encoder, handling different scikit-learn versions.
    
    Args:
    - encoder: Fitted OneHotEncoder
    - categorical_cols: Original categorical column names
    
    Returns:
    - list: Feature names for encoded columns
    """
    try:
        # First try get_feature_names_out (newer versions of scikit-learn)
        return encoder.get_feature_names_out(categorical_cols)
    except AttributeError:
        try:
            # Then try get_feature_names (older versions)
            return encoder.get_feature_names(categorical_cols)
        except AttributeError:
            # If both fail, generate feature names manually
            n_values = encoder.n_values_ if hasattr(encoder, 'n_values_') else [len(cats) for cats in encoder.categories_]
            feature_names = []
            for i, (col, n) in enumerate(zip(categorical_cols, n_values)):
                for j in range(n):
                    feature_names.append(f"{col}_{j}")
            return feature_names

def encode_categorical_columns(df, encoding_method="onehot", encoder=None):
    """
    Converts categorical columns in the DataFrame to either Label Encoding or One-Hot Encoding.
    Can use a pre-fitted encoder if provided.

    Args:
    - df (pd.DataFrame): The input dataframe.
    - encoding_method (str): The encoding method. Either 'label' or 'onehot'.
    - encoder (LabelEncoder/OneHotEncoder, optional): Pre-fitted encoder to use.

    Returns:
    - df_encoded (pd.DataFrame): The DataFrame with encoded categorical columns.
    - encoder: The encoder used (either provided or newly created)
    """
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # If no categorical columns, return original df and encoder
    if len(categorical_cols) == 0:
        return df, encoder

    if encoding_method == "label":
        # Use provided encoder or create new one
        if encoder is None:
            encoder = LabelEncoder()
            for col in categorical_cols:
                df[col] = encoder.fit_transform(df[col])
        else:
            # Use pre-fitted encoder
            for col in categorical_cols:
                df[col] = encoder.transform(df[col])
                
    elif encoding_method == "onehot":
        if encoder is None:
            encoder = OneHotEncoder(sparse_output=False)  # Ensure dense array output
            encoded_data = encoder.fit_transform(df[categorical_cols])
        else:
            # Use pre-fitted encoder
            encoded_data = encoder.transform(df[categorical_cols])
        
        # Get feature names using the helper function
        feature_names = get_feature_names(encoder, categorical_cols)
        
        # Convert the encoded data into a DataFrame
        encoded_df = pd.DataFrame(
            encoded_data, 
            columns=feature_names,
            index=df.index  # Ensure index matches original DataFrame
        )
        
        # Drop the original categorical columns and concatenate the one-hot encoded columns
        df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)

    return df, encoder

def scale_numerical_columns(df, scaling_method="standard", scaler=None):
    """
    Scales numerical columns in the DataFrame using the specified scaling method.
    Can use a pre-fitted scaler if provided.

    Args:
    - df (pd.DataFrame): The input dataframe.
    - scaling_method (str): The scaling method. Either 'standard' or 'minmax'.
    - scaler (StandardScaler/MinMaxScaler, optional): Pre-fitted scaler to use.

    Returns:
    - df_scaled (pd.DataFrame): The DataFrame with scaled numerical columns.
    - scaler: The scaler used (either provided or newly created)
    """
    # Identify numerical columns (int, float)
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # If no numerical columns, return original df and scaler
    if len(numerical_cols) == 0:
        return df, scaler
    
    # Initialize the scaler if not provided
    if scaler is None:
        if scaling_method == "standard":
            scaler = StandardScaler()
        elif scaling_method == "minmax":
            scaler = MinMaxScaler()
        else:
            raise ValueError("Invalid scaling method. Choose 'standard' or 'minmax'.")
        
        # Fit and transform with new scaler
        df_scaled = df.copy()
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        # Use pre-fitted scaler
        df_scaled = df.copy()
        df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])
        
    return df_scaled, scaler

def classify_columns(df):
    """
    Classify columns as numerical or categorical based on data type and heuristics.
    Args:
    - df: pandas DataFrame
    Returns:
    - categorical_cols: List of column names considered categorical
    - numerical_cols: List of column names considered numerical
    """
    numerical_cols = []
    categorical_cols = []
    datetime_cols = []
    text_cols = []

    for col in df.columns:
        # Check for numerical columns (int, float)
        if pd.api.types.is_numeric_dtype(df[col]):
            numerical_cols.append(col)
        # Check for categorical columns (object, category)
        elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            # Heuristic: If the number of unique values is less than 20% of total rows, classify as categorical
            if df[col].nunique() / len(df) < 0.2:
                categorical_cols.append(col)
            else:
                # Treat columns with too many unique values as textual or identifiers
                print(f"Column '{col}' has high cardinality and may be an identifier.")
                text_cols.append(col)
        else:
            # For other types (like datetime), we might need additional handling
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col])
                datetime_cols.append(col)

    return numerical_cols, categorical_cols, datetime_cols, text_cols


def preprocess_data(df, target_column=None, encoder=None, scaler=None):
    """
    Preprocess data for training or prediction, with optional pre-fitted transformers
    and optional target column.

    Args:
    - df (pd.DataFrame): Input dataframe
    - target_column (str, optional): Name of the target column. If None, only features are processed
    - encoder (LabelEncoder/OneHotEncoder, optional): Pre-fitted encoder
    - scaler (StandardScaler/MinMaxScaler, optional): Pre-fitted scaler

    Returns:
    - X_preprocessed: Preprocessed features
    - y: Target variable if target_column provided, None otherwise
    - encoder: The encoder used
    - scaler: The scaler used
    """
    # Handle target column if provided
    if target_column is not None and target_column in df.columns:
        X = df.drop(target_column, axis=1)
        y = df[target_column]
    else:
        X = df.copy()
        y = None

    # Classify columns
    numerical_cols, categorical_cols, datetime_cols, text_cols = classify_columns(X)
    
    # Initialize list to store processed dataframes
    feature_dfs = []
    
    # Process categorical columns if they exist
    if categorical_cols:
        df_encoded, encoder = encode_categorical_columns(
            X[categorical_cols].copy(), 
            encoding_method="onehot",
            encoder=encoder
        )
        if not df_encoded.empty:
            feature_dfs.append(df_encoded)

    # Process numerical columns if they exist
    if numerical_cols:
        df_standard_scaled, scaler = scale_numerical_columns(
            X[numerical_cols].copy(), 
            scaling_method="standard",
            scaler=scaler
        )
        if not df_standard_scaled.empty:
            feature_dfs.append(df_standard_scaled)

    # Combine all processed features
    if feature_dfs:
        X_preprocessed = pd.concat(feature_dfs, axis=1)
    else:
        X_preprocessed = pd.DataFrame()
    
    return X_preprocessed, y, encoder, scaler