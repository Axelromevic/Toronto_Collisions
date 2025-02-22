# Forecasting for Toronto Collision Accidents
<br><br>
This is a project developed during my studies of Software Engineering Technologies - Artificial Intelligence.

All the information used for the project can be found in the Public Safety Data Portal of Toronto Police Service.

The objective of the project was to create a ML model to predict whether the scenario result in a fatal or non fatal accident, based on the data contained in the Killed or Seriously Injured (KSI) Collissions dataset until 2021.
<br><br>
The next is a brief information contained in the dataset:

This dataset size is 56 columns with 17488 rows and contains all the factors involved in the accidents considered by the Toronto Police.

|FIELD|  FIELD NAME  |  DESCRIPTION  |
|-----|--------------|---------------|
|1 |	INDEX_ 	|Unique Identifier| 
|2 |	ACCNUM 	|Accident Number| 
|3 |	YEAR 	|Year Collision Occurred| 
|4 |	DATE 	|Date Collision Occurred| 
|5 |	TIME 	|Time Collision Occurred| 
|7 |	STREET1 	|Street Collision Occurred| 
|8 |	STREET2 	|Street Collision Occurred| 
|9 |	OFFSET 	|Distance and direction of the Collision| 
|10 |	ROAD_CLASS 	|Road Classification| 
|11 |	DISTRICT 	|City District| 
|12 |	WARDNUM 	|City of Toronto Ward collision occurred| 
|13 |	LATITUDE 	|Latitude| 
|14 |	LONGITUDE 	|Longitude| 
|15 |	LOCCOORD 	|Location Coordinate| 
|16 |	ACCLOC 	|Collision Location| 
|17 |	TRAFFCTL 	|Traffic Control Type| 
|18 |	VISIBILITY 	|Environment Condition| 
|19 |	LIGHT 	|Light Condition| 
|20 |	RDSFCOND 	|Road Surface Condition| 
|21 |	ACCLASS 	|Classification of Accident| 
|22 |	IMPACTYPE 	|Initial Impact Type| 
|23 |	INVTYPE 	|Involvement Type| 
|24 |	INVAGE 	|Age of Involved Party| 
|25 |	INJURY 	|Severity of Injury| 
|26 |	FATAL_NO 	|Sequential Number| 
|27 |	INITDIR 	|Initial Direction of Travel| 
|28 |	VEHTYPE 	|Type of Vehicle| 
|29 |	MANOEUVER 	|Vehicle Manouever| 
|30 |	DRIVACT 	|Apparent Driver Action| 
|31 |	DRIVCOND 	|Driver Condition |
|32 |	PEDTYPE 	|Pedestrian Crash Type - detail |
|33 |	PEDACT 	|Pedestrian Action |
|34 |	PEDCOND 	|Condition of Pedestrian |
|35 |	CYCLISTYPE 	|Cyclist Crash Type - detail |
|36 |	CYCACT 	|Cyclist Action |
|37 |	CYCCOND 	|Cyclist Condition |
|38 |	PEDESTRIAN 	|Pedestrian Involved In Collision |
|39 |	CYCLIST 	|Cyclists Involved in Collision |
|40 |	AUTOMOBILE 	|Driver Involved in Collision |
|41 |	MOTORCYCLE 	|Motorcyclist Involved in Collision |
|42 |	TRUCK 	|Truck Driver Involved in Collision |
|43 |	TRSN_CITY_VEH 	|Transit or City Vehicle Involved in Collision |
|44 |	EMERG_VEH 	|Emergency Vehicle Involved in Collision |
|45 |	PASSENGER 	|Passenger Involved in Collision |
|46 |	SPEEDING 	|Speeding Related Collision |
|47 |	AG_DRIV 	|Aggressive and Distracted Driving Collision |
|48 |	REDLIGHT 	|Red Light Related Collision |
|49 |	ALCOHOL 	|Alcohol Related Collision |
|50 |	DISABILITY 	|Medical or Physical Disability Related Collision |
|51 |	HOOD_158 	|Unique ID for City of Toronto Neighbourhood (new) |
|52 |	NEIGHBOURHOOD_158 	|City of Toronto Neighbourhood name (new) |
|53 |	HOOD_140 	|Unique ID for City of Toronto Neighbourhood (old) |
|54 |	NEIGHBOURHOOD_140 	|City of Toronto Neighbourhood name (old) |
|55 |	DIVISION 	|Toronto Police Service Division |
|56 |	ObjectID 	|Unique Identifier (auto generated) |

*Note: the 6th feature was omited in the dataset from the origin.
<br><br>
## Graphical Representation of the Information in the dataset.

This is a bar chart shows the quantity of accidents registered per year from 2006 to 2021, and divided by Fatal or non Fatal accident. This chart shows tendency of redution in th total number of accidents, but the number of fatal accidents is similar every year.

![Accidents per year](https://github.com/user-attachments/assets/a9fdaf48-6160-4f1a-ad6d-30b61339fdfa)

The next graph represents the collissions coordinates registered in the Toronto region from 2006 - 2021. Showing the majority accumulation of accidents in Downtown.

![image](https://github.com/user-attachments/assets/3ce2ee09-e913-4247-9332-c7a505c76cc8)

*More graphics samples are contained in the code.
<br><br>
## A little view of the dataset

Statistis
|     |         X    |         Y   |     INDEX_|  ... |    LONGITUDE  |  FATAL_NO   |   ObjectId|
|-----|--------------|-------------|-----------|------|---------------|-------------|-----------|
|count  |1.748800e+04  |1.748800e+04  |1.748800e+04  |...  |17488.000000  |773.000000  |17488.000000|  
|mean  |-8.838301e+06  |5.420763e+06  |3.643356e+07  |...    |-79.395806   |29.399741  | 8744.500000|  
|std    |1.160273e+04  |8.655506e+03  |3.695031e+07  |...    |  0.104229   |17.974056  | 5048.495089|  
|min   |-8.865305e+06  |5.402162e+06  |3.363207e+06  |...    |-79.638390   | 1.000000  |    1.000000|  
|25%   |-8.846498e+06  |5.413284e+06  |5.373756e+06  |...    |-79.469447   |14.000000  | 4372.750000|  
|50%   |-8.838366e+06  |5.419556e+06  |7.517348e+06  |...    |-79.396390   |28.000000  | 8744.500000|  
|75%   |-8.829649e+06  |5.427830e+06  |8.072255e+07  |...    |-79.318090   |43.000000  |13116.250000|  
|max   |-8.807929e+06  |5.443099e+06  |8.166694e+07  |...    |-79.122974   |78.000000  |17488.000000| 

The dataset showed many rows with missing data, therefore, it required a data cleaning process:

|Feature  |Nulls|
|---------|-----|
|EMERG_VEH            |17445|
|DISABILITY           |17013|
|CYCCOND              |16759|
|CYCACT               |16758|
|CYCLISTYPE           |16749|
|ALCOHOL              |16726|
|FATAL_NO             |16715|
|TRUCK                |16428|
|TRSN_CITY_VEH        |16419|
|REDLIGHT             |16035|
|MOTORCYCLE           |16006|
|CYCLIST              |15661|
|SPEEDING             |15047|
|PEDTYPE              |14567|
|PEDACT               |14531|
|PEDCOND              |14530|
|OFFSET               |14460|
|PASSENGER            |10867|
|PEDESTRIAN           |10409|
|DRIVCOND             | 8669|
|DRIVACT              | 8666|
|AG_DRIV              | 8449|
|INJURY               | 8273|
|MANOEUVER            | 7405|
|ACCLOC               | 5450|
|INITDIR              | 4937|
|VEHTYPE              | 3033|
|AUTOMOBILE           | 1628|
|STREET2              | 1592|
|WARDNUM              |  525|
|ROAD_CLASS           |  376|
|LOCCOORD             |   95|
|DISTRICT             |   35|
|TRAFFCTL             |   34|
|RDSFCOND             |   23|
|VISIBILITY           |   18|
|INVTYPE              |   14|
|ACCLASS              |    7|
|IMPACTYPE            |    4|
|ACCNUM               |    0|
|INDEX_               |    0|
|INVAGE               |    0|
|LIGHT                |    0|
|LONGITUDE            |    0|
|LATITUDE             |    0|
|YEAR                 |    0|
|DATE                 |    0|
|TIME                 |    0|
|STREET1              |    0|
|X                    |    0|
|Y                    |    0|
|HOOD_158             |    0|
|NEIGHBOURHOOD_158    |    0|
|HOOD_140             |    0|
|NEIGHBOURHOOD_140    |    0|
|DIVISION             |    0|
|ObjectId             |    0|

<br><br>
The dataset was cleaned and treated in order to select the most correlated feateures, then the feature were passed through SMOTE  and Shuffle before passing the data through the models.

For this excercise was used Grid Search in order to compare the models; Support Vector Machine, Random Forest and Gradient Boosting.<br><br>


The results were the next:
<br><br>


*** Metrics for Support Vector Machine (SVM) ***

Best parameters for SVM: {'C': 10, 'gamma': 1, 'kernel': 'rbf'}

SVM Accuracy: 0.9337595907928389

SVM Precision: 0.9381574752948677

SVM Recall: 0.9783909574468085

SVM F1 Score: 0.9578519121236778

SVM AUROC: 0.8816566760626504

SVM Cross Validation Scores: [0.9314578  0.93580563 0.92787724 0.93299233 0.93657289]

SVM Mean Accuracy: 0.9329411764705882

SVM Standard Deviation: 0.0031381814118011
<br><br>


*** Metrics for RandomForest ***

Best parameters for RandomForest: {'max_depth': 20, 'n_estimators': 100}

RandomForest Accuracy: 0.9209718670076726

RandomForest Precision: 0.9118095819346964

RandomForest Recall: 0.9933510638297872

RandomForest F1 Score: 0.9508353221957041

RandomForest AUROC: 0.8364759753738737

RandomForest Cross Validation Scores: [0.91775357 0.91483812 0.91543892]

RandomForest Mean Accuracy: 0.9160102009574745

RandomForest Standard Deviation: 0.0012569106955989328

Fitting 5 folds for each of 6 candidates, totalling 30 fits
<br><br>


*** Metrics for GradientBoosting ***

Best parameters for GradientBoosting: {'learning_rate': 0.2, 'n_estimators': 100}

GradientBoosting Accuracy: 0.8601023017902814

GradientBoosting Precision: 0.8502704241389126

GradientBoosting Recall: 0.9930186170212766

GradientBoosting F1 Score: 0.9161171599447937

GradientBoosting AUROC: 0.7049350291314809

GradientBoosting Cross Validation Scores: [0.85330674 0.8634341  0.86003683]

GradientBoosting Mean Accuracy: 0.8589258880278458

GradientBoosting Standard Deviation: 0.004208443832231231

SVM was the model with the best result.
<br><br>

Note: The results can be improved and there are other model such as XGBoost that could give a better result, but I decided to keeo the results I got during the course.

<br><br>
This project makes use of the 'Killed or Seriously Injured (KSI)' traffic collision dataset obtained from the Toronto Police Service via their Public Safety Data Portal. Since 2006, the dataset has provided free access to traffic data relating to those killed or badly injured in traffic crashes. The data is available under the Open Government Licence - Toronto.

The next is the link to visit the Toronto Police Service Public Safety Data Portal.
https://data.torontopolice.on.ca/pages/ksi?utm_source=chatgpt.com
