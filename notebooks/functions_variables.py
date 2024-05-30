def encode_tags(df):

    """Use this function to manually encode tags from each sale.
    You could also provide another argument to filter out low 
    counts of tags to keep cardinality to a minimum.
       
    Args:
        pandas.DataFrame

    Returns:
        pandas.DataFrame: modified with encoded tags
    """
   # Extract all unique tags
    all_tags = df['tags'].tolist()
    unique_tags = list(set([tag for sublist in all_tags for tag in sublist]))

    # Create a column for each unique tag and initialize with 0
    for tag in unique_tags:
        df[tag] = 0

    # Set the appropriate columns to 1 where the tag is present
    for index, row in df.iterrows():
        for tag in row['tags']:
            df.at[index, tag] = 1

    return df

def train_models(name, model, X_train, Y_train, X_val, Y_val):
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_val)
    
    r2 = r2_score(Y_val, Y_pred)
    mse = mean_squared_error(Y_val, Y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_val, Y_pred)

    print(f'{name} R2: {r2}')
    print(f'{name} MSE: {mse}')
    print(f'{name} RMSE: {rmse}')
    print(f'{name} MAE: {mae}')
    print('\n')
    
    return {'Model': name, 'R2': r2, 'MSE': mse, 'RMSE': rmse, 'MAE': mae}

def extract_info(result):
    """Extract relevant information from the JSON response
    Args:
        result (dict): JSON response from data
    Returns:
        dict: relevant information from the JSON response
    """

    info = {
        'permalink': result.get('permalink'),
        'status': result.get('status'),
        'list_date': result.get('list_date'),
        'sold_date': result['description'].get('sold_date'),
        'list_price': result.get('list_price'),
        'sold_price': result['description'].get('sold_price'),
        'year_built': result['description'].get('year_built'),
        'beds': result['description'].get('beds'),
        'baths': result['description'].get('baths'),
        'sqft': result['description'].get('sqft'),
        'lot_sqft': result['description'].get('lot_sqft'),
        'garage': result['description'].get('garage'),
        'tags': result.get('tags'),
        'office_name': result['source']['agents'][0].get('office_name'),
        'source_type': result['source']['type']
    }

    return info


