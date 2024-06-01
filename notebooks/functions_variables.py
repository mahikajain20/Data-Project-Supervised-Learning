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

def custom_cross_validation_no_city(training_data, n_splits =5):
    assert isinstance(training_data, pd.DataFrame), "training_data should be a DataFrame"
    assert isinstance(n_splits, int), "n_splits should be an integer"
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    training_folds = []
    validation_folds = []
    for train_index, val_index in kf.split(training_data):
        training_folds.append(training_data.iloc[train_index])
        validation_folds.append(training_data.iloc[val_index])

    return training_folds, validation_folds

# develop your custom functions here

def custom_cross_validation(training_data, n_splits =5):
    '''creates n_splits sets of training and validation folds

    Args:
      training_data: the dataframe of features and target to be divided into folds
      n_splits: the number of sets of folds to be created

    Returns:
      A tuple of lists, where the first index is a list of the training folds, 
      and the second the corresponding validation fold

    Example:
        >>> output = custom_cross_validation(train_df, n_splits = 10)
        >>> output[0][0] # The first training fold
        >>> output[1][0] # The first validation fold
        >>> output[0][1] # The second training fold
        >>> output[1][1] # The second validation fold... etc.
    '''
    assert isinstance(training_data, pd.DataFrame), "training_data should be a DataFrame"
    assert isinstance(n_splits, int), "n_splits should be an integer"
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    training_folds = []
    validation_folds = []
    for train_index, val_index in kf.split(training_data):
        
        #Compute city means
        city_means = training_folds[-1].groupby('city')['sold_price'].mean().rename('mean_price_city').reset_index()

        #Merge city means with training and validation folds
        training_folds[-1] = training_folds[-1].merge(city_means, on='city', how='left')
        validation_folds[-1] = validation_folds[-1].merge(city_means, on='city', how='left')
        
        training_folds.append(training_data.iloc[train_index])
        validation_folds.append(training_data.iloc[val_index])

    return training_folds, validation_folds

def hyperparameter_search(training_folds, validation_folds, param_grid, name):
    '''outputs the best combination of hyperparameter settings in the param grid, 
    given the training and validation folds

    Args:
      training_folds: the list of training fold dataframes
      validation_folds: the list of validation fold dataframes
      param_grid: the dictionary of possible hyperparameter values for the chosen model

    Returns:
      A list of the best hyperparameter settings based on the chosen metric

    Example:
        >>> param_grid = {
          'max_depth': [None, 10, 20, 30],
          'min_samples_split': [2, 5, 10],
          'min_samples_leaf': [1, 2, 4],
          'max_features': ['sqrt', 'log2']} # for random forest
        >>> hyperparameter_search(output[0], output[1], param_grid = param_grid) 
        # assuming 'ouput' is the output of custom_cross_validation()
        [20, 5, 2, 'log2'] # hyperparams in order
    '''
    assert isinstance(training_folds, list), "training_folds should be a list"
    assert isinstance(validation_folds, list), "validation_folds should be a list"
    assert isinstance(param_grid, dict), "param_grid should be a dictionary"
    best_hyperparameters = []
    best_score = float('inf')
    for params in ParameterGrid(param_grid):
        #print(params)
        scores = []
        for t_fold, v_fold in zip(training_folds, validation_folds):
            if name == 'DecisionTree':
                model = DecisionTreeRegressor(**params, random_state=42)
            if name == 'RandomForest':
                model = RandomForestRegressor(**params, random_state=42)
            if name == 'XGBoost':
                #dtrain = xgb.DMatrix(data=t_fold, label=v_fold, enable_categorical=True)
                model = XGBRegressor(**params, random_state=42)
  
                
            x_train = t_fold.drop(columns='sold_price')
            y_train = t_fold['sold_price']
            x_val = v_fold.drop(columns='sold_price')
            y_val = v_fold['sold_price']

            model.fit(x_train, y_train)
            preds = model.predict(x_val)
            score = np.sqrt(mean_squared_error(y_val, preds))
            scores.append(score)
        if np.mean(scores) < best_score:
            best_score = np.mean(scores)
            best_hyperparameters = params
        
    return best_hyperparameters, best_score

# # Create unit tests
# def test_functions(df):

#     # Test custom_cross_validation
#     training_folds, validation_folds = custom_cross_validation(df, n_splits=2)
#     assert isinstance(training_folds, list), "Output of custom_cross_validation should be a list"
#     assert isinstance(validation_folds, list), "Output of custom_cross_validation should be a list"

#     # Test hyperparameter_search
#     param_grid = {
#         'max_depth': [None, 10],
#         'min_samples_split': [2, 5],
#     }
#     best_hyperparameters, best_score = hyperparameter_search(training_folds, validation_folds, param_grid, 'DecisionTree')
#     assert isinstance(best_hyperparameters, dict), "Output of hyperparameter_search should be a dictionary"
#     assert isinstance(best_score, float), "Output of hyperparameter_search should be a float"