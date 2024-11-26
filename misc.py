import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import miceforest as mf
from missforest import MissForest
from sklearn.impute import SimpleImputer

df=pd.read_csv('knn5_time_only_for_0-1.csv')
print(df.shape)
columns_to_drop = [col for col in df.columns if col.lower().startswith('z')]

# Print columns to drop
print("Columns starting with 'z' or 'Z':", columns_to_drop)

# Drop these columns from the DataFrame
df = df.drop(columns=columns_to_drop)

# Print the resulting DataFrame
print("\nResulting DataFrame:")
print(df.shape)
df.to_csv('albion_with_time_diagnosis_final.csv',index=False)
'''


object_columns = df.select_dtypes(include=['object']).columns.to_list()

print(len(object_columns))
target_column = 'DIAGNOSIS'
X = df.drop(columns=[target_column])  # Features
y = df[target_column]  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
missforest_imputer = MissForest(early_stopping=False,max_iter=10)

# Impute the target variable (y) using SimpleImputer (most frequent strategy)
imputer_y = SimpleImputer(strategy='most_frequent')  # For classification, 'most_frequent' makes sense
y_train_imputed = imputer_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_imputed = imputer_y.transform(y_test.values.reshape(-1, 1)).ravel()

# Impute the feature matrix (X) using MissForest (a Random Forest-based imputer)
# MissForest requires reshaping data to a format that fits its input, as it expects the data to be in 2D arrays
X_train_imputed = missforest_imputer.fit_transform(X_train)
X_test_imputed = missforest_imputer.transform(X_test)

# Standardize the features (after imputation)
scaler = StandardScaler()
X_train_imputed = scaler.fit_transform(X_train_imputed)
X_test_imputed = scaler.transform(X_test_imputed)

from catboost import CatBoostClassifier, EShapCalcType, EFeaturesSelectionAlgorithm
from catboost import Pool




train_pool = Pool(X_train_imputed, y_train_imputed)
test_pool = Pool(X_test_imputed, y_test_imputed)
def select_features_adult(algorithm: EFeaturesSelectionAlgorithm, steps: int = 1):
    print('Algorithm:', algorithm)
    model = CatBoostClassifier(iterations=500, random_seed=0)
    summary = model.select_features(
        train_pool,
        eval_set=test_pool,
        features_for_select=list(range(X.shape[1])),
        num_features_to_select=10,
        steps=steps,
        algorithm=algorithm,
        shap_calc_type=EShapCalcType.Regular,
        train_final_model=True,
        logging_level='Silent',
        plot=True
    )
    print('Selected features:', summary['selected_features_names'])
    return summary

adult_shap_summary = select_features_adult(algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues, steps=7)
print(adult_shap_summary)

    # Create the LogisticRegression model with Elastic Net penalty





import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix


rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=0, max_features='sqrt',\
                            )
rf.fit(X_train_imputed, y_train_imputed)

# Make predictions for the test set
y_pred_test = rf.predict(X_test_imputed)

# Print accuracy for the training and test sets
print("Accuracy on training set: {:.3f}".format(rf.score(X_train_imputed, y_train_imputed)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test_imputed, y_test_imputed)))
print(confusion_matrix(y_test_imputed, y_pred_test))


import shap
shap.initjs()

samples = X_train_imputed

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(samples, approximate=False, check_additivity=False)

shap.summary_plot(shap_values[1], samples)

def make_shap_waterfall_plot(shap_values, features, num_display=20):
    
    
    A function for building a SHAP waterfall plot.
    
    SHAP waterfall plot is used to visualize the most important features in a descending order.
    
    Parameters:
    shap_values (list): SHAP values obtained from a model
    features (pandas DataFrame): a list of features used in a model
    num_display(int): number of features to display
    
    Returns:
    matplotlib.pyplot plot: SHAP waterfall plot
    
    
    
    column_list = features.columns
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order = np.sort(feature_ratio)[::-1]
    cum_sum = np.cumsum(feature_ratio_order)
    column_list = column_list[:num_display]
    feature_ratio_order = feature_ratio_order[:num_display]
    cum_sum = cum_sum[:num_display]
    
    num_height = 0
    if (num_display >= 20) & (len(column_list) >= 20):
        num_height = (len(column_list) - 20) * 0.4
        
    fig, ax1 = plt.subplots(figsize=(8, 8 + num_height))
    ax1.plot(cum_sum[::-1], column_list[::-1], c='blue', marker='o')
    ax2 = ax1.twiny()
    ax2.barh(column_list[::-1], feature_ratio_order[::-1], alpha=0.6)
    
    ax1.grid(True)
    ax2.grid(False)
    ax1.set_xticks(np.arange(0, round(cum_sum.max(), -1)+1, 10))
    ax2.set_xticks(np.arange(0, round(feature_ratio_order.max(), -1)+1, 10))
    ax1.set_xlabel('Cumulative Ratio')
    ax2.set_xlabel('Composition Ratio')
    ax1.tick_params(axis="y", labelsize=13)
    plt.ylim(-1, len(column_list))

make_shap_waterfall_plot(shap_values[1], samples)
'''