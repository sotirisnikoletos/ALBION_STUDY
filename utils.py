import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # Enables IterativeImputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV, SelectKBest, chi2, f_classif
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns

def evaluate_model_performance(y_true, y_pred, y_prob=None):
    """
    Evaluates the model's performance using Accuracy, Precision, Recall, F1-Score, and ROC AUC.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for ROC curve. If None, ROC will not be plotted.
    
    Prints:
        Accuracy, Precision, Recall, F1-Score.
        Optionally, plots the ROC AUC curve if `y_prob` is provided.
    """
    # Calculate Accuracy, Precision, Recall, F1-Score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # If probabilities are provided, plot the ROC curve
    if y_prob is not None:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        print(f"ROC AUC: {roc_auc:.4f}")
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC)')
        plt.legend(loc="lower right")
        plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()






# 1. Recursive Feature Elimination with Cross-Validation (RFECV)
def rfecv_feature_selection(X, y, estimator=None, step=1, cv=5):
    """
    Perform feature selection using Recursive Feature Elimination with Cross-Validation.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        estimator: Estimator used for feature elimination (e.g., LogisticRegression, RandomForestClassifier).
        step (int): The number of features to remove at each iteration.
        cv (int): The number of folds in cross-validation.
        
    Returns:
        selected_features (list): List of selected feature indices.
        rfecv (RFECV): Fitted RFECV object with selected features.
    """
    if estimator is None:
        estimator = RandomForestClassifier()
    
    rfecv = RFECV(estimator=estimator, step=step, cv=StratifiedKFold(cv))
    rfecv.fit(X, y)
    
    selected_features = X.columns[rfecv.support_].tolist()  # Get column names of selected features
    return selected_features, rfecv


# 2. Univariate Feature Selection (SelectKBest)
def univariate_feature_selection(X, y, k=10, score_func=f_classif):
    """
    Select the top k features using univariate statistical tests.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        k (int): Number of top features to select.
        score_func: The scoring function to use (e.g., f_classif for classification).
        
    Returns:
        selected_features (list): List of selected feature names.
        selector (SelectKBest): Fitted SelectKBest object with selected features.
    """
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    
    selected_features = X.columns[selector.get_support()].tolist()
    return selected_features, selector


# 3. Lasso (L1 Regularization) for Feature Selection
def lasso_feature_selection(X, y, alpha=0.1):
    """
    Perform feature selection using Lasso (L1 regularization).
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        alpha (float): Regularization strength.
        
    Returns:
        selected_features (list): List of selected feature names.
        lasso (LassoCV): Fitted LassoCV model.
    """
    lasso = LassoCV(alpha=alpha)
    lasso.fit(X, y)
    
    selected_features = X.columns[lasso.coef_ != 0].tolist()
    return selected_features, lasso


# 4. Tree-based Feature Selection (Random Forest / XGBoost)
def tree_based_feature_selection(X, y, model_type="random_forest", n_estimators=100):
    """
    Perform feature selection using tree-based models like Random Forest or XGBoost.
    
    Parameters:
        X (pd.DataFrame or np.ndarray): Feature matrix.
        y (pd.Series or np.ndarray): Target vector.
        model_type (str): Choose between 'random_forest' or 'xgboost'.
        n_estimators (int): Number of trees in the forest (for RandomForestClassifier).
        
    Returns:
        selected_features (list): List of selected feature names.
        model: Fitted model (RandomForestClassifier or XGBClassifier).
    """
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=n_estimators)
    elif model_type == "xgboost":
        model = XGBClassifier(n_estimators=n_estimators)
    else:
        raise ValueError("Model type must be 'random_forest' or 'xgboost'")
    
    model.fit(X, y)
    
    feature_importances = model.feature_importances_
    selected_features = X.columns[feature_importances > np.mean(feature_importances)].tolist()
    
    return selected_features, model






# Function: Replace missing values with np.nan and drop columns with high missing percentages
def handle_missing_values(df, threshold):
    """
    Replaces missing values with np.nan and drops columns where the percentage
    of missing values exceeds the given threshold.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame to process.
        threshold (float): Threshold percentage (0-100) for dropping columns. 
                           Columns with more missing values than this are dropped.
    
    Returns:
        pd.DataFrame: Processed DataFrame with columns dropped as per the threshold.
    """
    # Replace missing-like values (e.g., None, '') with np.nan
    df = df.replace({None: np.nan, '': np.nan, '999': np.nan, 999: np.nan, 999.0:np.nan,'99':np.nan,' ':np.nan})
    
    # Calculate the percentage of missing values per column
    missing_percentages = df.isna().mean() * 100
    
    # Drop columns exceeding the threshold
    cols_to_drop = missing_percentages[missing_percentages > threshold].index
    df = df.drop(columns=cols_to_drop)
    
    return df


# Function: Remove highly correlated columns
def remove_highly_correlated_columns(df, threshold=0.95):
    """
    Removes columns from the DataFrame that are highly correlated above the specified threshold.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame to process. Assumes numerical columns only.
        threshold (float): Correlation threshold above which columns are removed.
    
    Returns:
        pd.DataFrame: DataFrame with one of each pair of highly correlated columns removed.
    """
    # Calculate the correlation matrix
    corr_matrix = df.corr().abs()
    
    # Create an upper triangle matrix of the correlation matrix
    upper_triangle = corr_matrix.where(
        ~np.tril(np.ones(corr_matrix.shape)).astype(bool)
    )
    
    # Identify columns to drop based on the threshold
    columns_to_drop = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > threshold)
    ]
    
    # Drop the identified columns
    df = df.drop(columns=columns_to_drop)
    
    return df


# Function: Remove columns with variance below a threshold
def remove_low_variance_columns(df, threshold=0.0):
    """
    Removes columns from the DataFrame that have variance below the specified threshold.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with numerical columns only.
        threshold (float): Minimum variance a column must have to be retained.
    
    Returns:
        pd.DataFrame: DataFrame with low-variance columns removed.
    """
    selector = VarianceThreshold(threshold=threshold)
    selected_data = selector.fit_transform(df)
    selected_columns = df.columns[selector.get_support()]
    return pd.DataFrame(selected_data, columns=selected_columns, index=df.index)


# Function: Simple Imputation
def simple_impute(df, strategy='mean', fill_value=None):
    """
    Performs simple imputation to fill missing values in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values.
        strategy (str): The imputation strategy - 'mean', 'median', 'most_frequent', or 'constant'.
        fill_value (Any): Value to use if strategy is 'constant'. Default is None.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns, index=df.index)


# Function: Iterative Imputation
def iterative_impute(df, max_iter=10, random_state=0):
    """
    Performs iterative imputation to fill missing values in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values.
        max_iter (int): Maximum number of imputation iterations.
        random_state (int): Random state for reproducibility.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns, index=df.index)


# Function: KNN Imputation
def knn_impute(df, n_neighbors=5, weights='uniform'):
    """
    Performs KNN imputation to fill missing values in the DataFrame.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with missing values.
        n_neighbors (int): Number of neighbors to use for imputation.
        weights (str): Weight function used in prediction - 'uniform' or 'distance'.
    
    Returns:
        pd.DataFrame: DataFrame with missing values imputed.
    """
    imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
    imputed_data = imputer.fit_transform(df)
    return pd.DataFrame(imputed_data, columns=df.columns, index=df.index)
