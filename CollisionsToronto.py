import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import folium
import io
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


df = pd.read_csv("KSI.csv")
print(df.head(2))

# EXPLORATORY DATA ANALYSIS (EDA)
print(df.describe())

sort_NULLS = df.isnull().sum()
print("Nulls per column before encoding \n", sort_NULLS.sort_values(ascending=False))

#CHART Heatmap of nulls
plt.figure(figsize=(20, 5))
sns.heatmap(df.isnull(), cmap='plasma', cbar_kws={'label': 'Missing Data'}, yticklabels=False)
plt.title("Nulls per Feature")
plt.xlabel("Features")
plt.ylabel("Samples")
plt.show()


# Target column processing
print("\n\nReplacing values in Target for Non Fatal or Fatal result\n")
print(df['ACCLASS'].unique())
df['ACCLASS'] = df['ACCLASS'].replace({'Non-Fatal Injury': 'Non Fatal', 'Property Damage Only': 'Non Fatal'})

# Drop rows where ACCLASS is null
df.dropna(subset=['ACCLASS'], inplace=True)
print(df['ACCLASS'].unique())

# Convert DATE column
df['DATE'] = pd.to_datetime(df['DATE'], format='%Y/%m/%d %H:%M:%S%z', errors='coerce')
df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month
df['DAY'] = df['DATE'].dt.day


#GRAPH CHART Accident per year
grouped_year = df.groupby(['YEAR', 'ACCLASS'])['ACCLASS'].count().unstack()
plt.figure(figsize=(15, 6))
ax = grouped_year.plot(kind='bar', stacked=True, figsize=(15, 6), colormap="coolwarm")
plt.title("Accidents per Year")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")
plt.xticks(rotation=0)
plt.legend(title="ACCLASS")
for container in ax.containers:
    ax.bar_label(container, fmt='%d', label_type='edge', fontsize=10, padding=3)
plt.show()


#GRAPH Map with accidents location
map_center = [df['LATITUDE'].mean(), df['LONGITUDE'].mean()]
map = folium.Map(location=map_center, zoom_start=12)

# Add scatter points to the map
for index, row in df.iterrows():
    folium.CircleMarker(location=[row['LATITUDE'], row['LONGITUDE']],
        radius=0.1, color='red', fill=True, fill_opacity=0.2).add_to(map)

map

#This code was added due to some problems with VScode to visualize the map
map.save("mapa_accidentes.html")
import webbrowser
webbrowser.open("mapa_accidentes.html")


#CHART Fatal or not Fatal Chart
impactType = df['ACCLASS'].value_counts()
plt.pie(impactType, labels=impactType.index, autopct='%1.1f%%')
plt.title('Fatal or not Fatal', fontsize=16)
plt.xlabel( '', fontsize=11)
plt.ylabel('')
plt.show()

#CHART Accident per District
impactType = df['DISTRICT'].value_counts()
plt.pie(impactType, labels=impactType.index, autopct='%1.1f%%')
plt.title('Accidents per District', fontsize=16)
plt.xlabel( '', fontsize=11)
plt.ylabel('')
plt.show()


#DATA CLEANING AND FEATRUE ENGINEERING

# Convert TIME column to ranges
def timeRange(time):
    time = int(time)
    hr = time // 100
    return f"{hr} - {hr + 1} hrs"

df['TIME(range)'] = df['TIME'].apply(timeRange)


# Fill missing values for binary columns
binaryYesNo = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK', 'TRSN_CITY_VEH', 
               'EMERG_VEH', 'PASSENGER', 'SPEEDING', 'AG_DRIV', 'REDLIGHT', 'ALCOHOL', 'DISABILITY']
df[binaryYesNo] = df[binaryYesNo].fillna("No")

# Replace missing values with unknown if cyclist is related
def nan_cyclist_columns(row):
    if pd.isna(row['CYCLISTYPE']):  
        row['CYCLISTYPE'] = "Unknown" if row['CYCLIST'] == "Yes" else "Cyclist not involved"
    if pd.isna(row['CYCACT']): 
        row['CYCACT'] = "Unknown" if row['CYCLIST'] == "Yes" else "Cyclist not involved"
    if pd.isna(row['CYCCOND']):
        row['CYCCOND'] = "Unknown" if row['CYCLIST'] == "Yes" else "Cyclist not involved"
    return row

df = df.apply(nan_cyclist_columns, axis=1)

# Replace missing values with unknown if pedestrian-related
def nan_pedestrian_columns(row):
    if pd.isna(row['PEDTYPE']):  
        row['PEDTYPE'] = "Unknown" if row['PEDESTRIAN'] == "Yes" else "Pedestrian not involved"
    if pd.isna(row['PEDACT']): 
        row['PEDACT'] = "Unknown" if row['PEDESTRIAN'] == "Yes" else "Pedestrian not involved"
    if pd.isna(row['PEDCOND']):
        row['PEDCOND'] = "Unknown" if row['PEDESTRIAN'] == "Yes" else "Pedestrian not involved"
    return row

df = df.apply(nan_pedestrian_columns, axis=1)

# Replace missing values in driver-related columns
def nan_drive_columns(row):
    if pd.isna(row['DRIVACT']):  
        row['DRIVACT'] = "Unknown" 
    if pd.isna(row['DRIVCOND']): 
        row['DRIVCOND'] = "Unknown" 
    if pd.isna(row['MANOEUVER']): 
        row['MANOEUVER'] = "Unknown" 
    if pd.isna(row['INITDIR']): 
        row['INITDIR'] = "Unknown" 
    return row

df = df.apply(nan_drive_columns, axis=1)

# Replace missing values with most frequent values
columns_moda = ['ACCLOC', 'INITDIR', 'VEHTYPE', 'STREET2', 'ROAD_CLASS', 
                'LOCCOORD', 'DISTRICT', 'TRAFFCTL', 'RDSFCOND', 'VISIBILITY', 
                'INVTYPE', 'IMPACTYPE']

for col in columns_moda:
    if df[col].isnull().sum() > 0:
        most_frequent = df[col].mode()[0]  
        df[col] = df[col].fillna(most_frequent)


# Drop unnecessary columns
not_necessary_columns = ['X', 'Y', 'LATITUDE', 'LONGITUDE', 'ObjectId', 'INDEX_', 'ACCNUM', 
                         'YEAR', 'DATE', 'TIME', 'HOOD_140', 'OFFSET', 'WARDNUM', 'FATAL_NO', 
                         'INJURY', 'NEIGHBOURHOOD_140', 'HOOD_158']
df.drop(columns=not_necessary_columns, inplace=True)



# Select Numeric & Categorical Features
numeric_f = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categoric_f = df.select_dtypes(include=['object']).columns.tolist()

# Pipelines to impute the rest of missing data and encode 
numeric_imputed = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                  ('scaler', RobustScaler())])
categoric_imputed = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                   ('encoder', OneHotEncoder(drop='first', sparse_output=False))])

preprocessor = ColumnTransformer(transformers=[('num', numeric_imputed, numeric_f), 
                                               ('cat', categoric_imputed, categoric_f)])

# Apply Preprocessing
df_processed = pd.DataFrame(preprocessor.fit_transform(df))

# Get Column Names
encoded_cols = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(categoric_f)
df_processed.columns = numeric_f + list(encoded_cols)

#Find the new name of ACCLASS after encoding
target_ACCLASS = [col for col in df_processed.columns if 'ACCLASS' in col][0]

# Assign X and Y / Features and Target
X = df_processed.drop(columns=[target_ACCLASS]) 
y = df_processed[target_ACCLASS]

# Feature Selection (TOP 30)
feature_selector = RandomForestClassifier(n_estimators=100, random_state=1)
feature_selector.fit(X, y)
selector = SelectFromModel(feature_selector, max_features=30, prefit=True)
X_Top = selector.transform(X)

#Shows the name of the features
selected_features = X.columns[selector.get_support()]
print("\nFeatures selected by RandomForestClassifier:\n", selected_features)

#SMOTE
X_Top, y = SMOTE(sampling_strategy=0.3, random_state=1, k_neighbors=2).fit_resample(X_Top, y)

#Shuffle
X_Top, y = shuffle(X_Top, y, random_state=1)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_Top, y, test_size=0.2, random_state=1, stratify=y)

#GridSearch
print("\n\nStarts GridSearch\n\n")

# SVM params
param_grid_svm = {'C': [10], 'gamma': [0.1, 1, 3], 'kernel': ['rbf', 'poly']}
grid_search_svm = GridSearchCV(SVC(class_weight='balanced'), param_grid_svm, cv=5, scoring='accuracy', verbose=1)
grid_search_svm.fit(X_train, y_train)
best_svm_model = grid_search_svm.best_estimator_

#SVM metrics
print("*** Metrics for SVM ***")
y_pred_svm = best_svm_model.predict(X_test)
print("\nBest parameters for SVM:", grid_search_svm.best_params_)
print("\nSVM Accuracy:", accuracy_score(y_test, y_pred_svm))
print("\nSVM Precision:", precision_score(y_test, y_pred_svm))
print("\nSVM Recall:", recall_score(y_test, y_pred_svm))
print("\nSVM F1 Score:", f1_score(y_test, y_pred_svm))
print("\nSVM AUROC:", roc_auc_score(y_test, y_pred_svm))
cv_scores_svm = cross_val_score(best_svm_model, X_Top, y, cv=5, scoring='accuracy')
print("\nSVM Cross Validation Scores:", cv_scores_svm)
print("\nSVM Mean Accuracy:", np.mean(cv_scores_svm))
print("\nSVM Standard Deviation:", np.std(cv_scores_svm))

#RandomForest Params
param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [10, 20]}
grid_search_rf = GridSearchCV(RandomForestClassifier(class_weight='balanced'), param_grid_rf, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

#RandomForest Metrics
print("\n*** Metrics for RandomForest ***\n")
y_pred_rf = best_rf_model.predict(X_test)
print("\nBest parameters for RandomForest:", grid_search_rf.best_params_)
print("\nRandomForest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandomForest Precision:", precision_score(y_test, y_pred_rf))
print("\nRandomForest Recall:", recall_score(y_test, y_pred_rf))
print("\nRandomForest F1 Score:", f1_score(y_test, y_pred_rf))
print("\nRandomForest AUROC:", roc_auc_score(y_test, y_pred_rf))
cv_scores_rf = cross_val_score(best_rf_model, X_Top, y, cv=3, scoring='accuracy')
print("\nRandomForest Cross Validation Scores:", cv_scores_rf)
print("\nRandomForest Mean Accuracy:", np.mean(cv_scores_rf))
print("\nRandomForest Standard Deviation:", np.std(cv_scores_rf))

#GradientBoosting Params
param_grid_gb = {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1, 0.2]}
grid_search_gb = GridSearchCV(GradientBoostingClassifier(), param_grid_gb, cv=5, scoring='accuracy', verbose=1)
grid_search_gb.fit(X_train, y_train)
best_gb_model = grid_search_gb.best_estimator_

#GradientBoosting Metrics
print("\n*** Metrics for GradientBoosting ***\n")
y_pred_gb = best_gb_model.predict(X_test)
print("\nBest parameters for GradientBoosting:", grid_search_gb.best_params_)
print("\nGradientBoosting Accuracy:", accuracy_score(y_test, y_pred_gb))
print("\nGradientBoosting Precision:", precision_score(y_test, y_pred_gb))
print("\nGradientBoosting Recall:", recall_score(y_test, y_pred_gb))
print("\nGradientBoosting F1 Score:", f1_score(y_test, y_pred_gb))
print("\nGradientBoosting AUROC:", roc_auc_score(y_test, y_pred_gb))
cv_scores_gb = cross_val_score(best_gb_model, X_Top, y, cv=3, scoring='accuracy')
print("\nGradientBoosting Cross Validation Scores:", cv_scores_gb)
print("\nGradientBoosting Mean Accuracy:", np.mean(cv_scores_gb))
print("\nGradientBoosting Standard Deviation:", np.std(cv_scores_gb))   
