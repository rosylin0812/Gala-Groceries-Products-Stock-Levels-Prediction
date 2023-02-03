# Python file should be able to load a CSV file into a data frame, 
# then immediately start training on that data.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Load data
def load_data(path: str= None):
    """
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    :param      path (optional): str, relative path of the CSV file

    :return     df: pd.DataFrame
    """

    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# create our target variable `y` and independent variables `X`
def create_target_independent_var(target: str= 'estimated_stock_pct', df: pd.DataFrame= None):
    """
    This function takes a Pandas DataFrame to split the columns 
    into a target column and a set of columns of independent variables. 

    :param      target: str, target variable. df: aimed dataframe.

    :return     y: panda Series. X: panda dataframe
    """
    # check to see if target is in the dataframe
    if target not in df:
        raise Exception(f"target: {target} is not in the dataframe.")
    
    y = df[target]
    X = df.drop(columns=[target])
    
    return y, X


# train the model
def train_algorithm_cv(X: pd.DataFrame= None, y: pd.Series= None, k: int=10):
    
    """This function takes the predictors and target variables to train a 
        randomforest model with cross k folds. Using cross-validation,
        performace metrics (mean absolute error) will be output for each fold.
        Average mean absolute error and average accuracy will be output as well.
        
        : param    y: panda Series. X: panda dataframe
        : return   console will print performace metrics for each fold.
    """

    # Create a list that will store the accuracies(mae) of each fold
    Mae =[]
    
    # a loop to run K folds of cross-validation
    for fold in range(0, k):
        
        
        # Create training and test sample dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=split, random_state=42)

        # Instantiate algorithm
        scaler = StandardScaler()
        model = RandomForestRegressor()

        # Scale X data, we scale the data because it helps the algorithm to converge
        # and helps the algorithm to not be greedy with large values
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Train model
        trained_model = model.fit(X_train, y_train)

        # Generate predictions on test sample
        y_pred = trained_model.predict(X_test)

        # Compute accuracy, using mean absolute error
        mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
        Mae.append(mae)
        print(f"Fold {fold + 1}: MAE = {mae:.3f}")

        print(f"Average MAE: {(sum(Mae) / len(Mae)):.2f}")
        print('Accuracy:', 1- (sum(Mae) / len(Mae))*2)