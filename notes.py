

###NOTES###

'''
You can either define a utility function (or fitness function) that measures
how good your model is, or you can define a cost function that measures how bad it is. 
For linear regression problems, people
'''

'''
> What I already know:

.head()
.info()
.describe()
.value_counts()
plotting

> What I learned:

• Stratified Splitting based on unbalanced categorical data
• for data of numerical features, use hist()(pairplot on seaborn); 
• this shows how the data is distributed along, its limits and boundaries, 
biases and disproportions.
• clean the data, for missing data, either you get more data; or 
'''

```Splitting the data```

'''
create a test set 80/10/10 ;
• or 70/15/15
• 70 training set
• 10/15 test and validation set

Using StratifiedShuffleSplit
'''

'''
typically use a cost function that measures the distance between the linear model’s predictions and the training examples; the objective is to minimize this distance.
 Possible solutions for overfitting are:
to simplify the model
 constrain it (i.e., regularize it)
 or get a lot more training data
You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. The goal is to shortlist a few (two to five) promising models.  You can easily save Scikit-Learn models by using Python’s pickle module
'''


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strain_train_set = housing_loc[train_index]
    strain_test_set = housing.loc[test_index]
# housing = strat_train_set.copy()

'''
• Get rid of the corresponding districts.
• Get rid of the whole attribute.
• Set the values to some value (zero, the mean, the median, etc.).
'''

```sklearn's SimpleImputer gets the job done```

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
imputer.fit(housing_num)

# to check values
imputer.statistics_

# to apply it to the dataset
X = imputer.transform(housing_num)

# transform into dataframe
housing_tr = pd.DataFrame(X, columns=housing.columns)

'''
for categorical data we use sklearn's label encoder
import ordinalencoder to fit transform the array into its proper form
then fit transform it into the one hot encoder
'''

'''
Pipelines create a sequence of data processing
e.q.  conversion of text categorical data into numerical categorical 
encoders as you can see in this example
'''

```basically a structurized filter of operations to simplify the data```

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# call the pipeline function and add in functions **to apply** 
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribute_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)

# column transformer applies the function towards the data
try:
    from sklearn.compose import ColumnTransformer
except ImportError:
    from future_encoders import ColumnTransformer

num_attribs = list(housing_num)
cat_attributes = ['ocean_proximity']

full_pipeline = ColumnTransfer([
    ('num',num_pipeline, num_attribs),
    ('cat',OneHotEncoder(), cat_attributes)
])
housing_prepared = full_pipeline.fit_transform(housing)
#housing_prepared

```Selecting a model and training data```

'''
Sklearns Regression models
Linear regression 
Random Forest
Decision Tree

from here we call the functions
fit the housing_prepared, and housing_labels data (X_train,y_train)
'''

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 

```fr LINEAR REGRESSION```
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# display scores
> Predictions:	 [210644.6  317768.81 210956.43  59218.99 189747.56]
> Labels:		 [286600.0, 340600.0, 196900.0,  46300.0, 254500.0]
```from here we compare the two data, some predictions are very far off (more than 30%) ```

# here we use RMSE to find out our errors
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
>>>lin_rmse
> 68628.19819848923
```underfitting```

```DECISION TREE REGRESSOR```
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
>>>tree_rmse
> 0.0 
```overfitting```

```RANDOM FOREST REGRESSOR```
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor(n_estimators=10, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
>>>forest_rmse

```using of **VALIDATION** set with k-fold```
```from here we split the data into folds and test them```
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
    scoring = 'neg_mean_squared_error', cv=10)
    # cv being the number of fold splits
rmse_scores = np.sqrt(-scores)

###RESULTS

>>>display_scores(tree_rmse_scores)
> Scores:  [69327.01708558 65486.39211857 71358.25563341 69091.37509104
            70570.20267046 75529.94622521 69895.20650652 70660.14247357
            75843.74719231 68905.17669382]
> Mean:  70666.74616904806
> Standard Deviation:  2928.322738055112

>>>display_scores(tree_rmse_scores)
> Scores:  [66782.73843989 66960.118071   70347.95244419 74739.57052552
            8031.13388938 71193.84183426 64969.63056405 68281.61137997
            71552.91566558 67665.10082067]
> Mean:  69052.46136345083
> Standard Deviation:  2731.674001798349

>>>display_scores(forest_rmse_scores)
> Scores:  [51646.44545909 48940.60114882 53050.86323649 54408.98730149
           50922.14870785 56482.50703987 51864.52025526 49760.85037653
           55434.21627933 53326.10093303]
> Mean:  52583.72407377466
> Standard Deviation:  2298.353351147122
```RFR HAVE BETTER VALIDATION RESULTS```

```FINE TUNING AND GRID SEARCH```
'''
gridsearch allows you to fiddle with hyperparameters and apply them to the model
without having to tune the hyperparameters manually
'''

from sklearn.model_selection import GridSearchCV
# parameter grid
param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]}
    #bootstrap = method for sampling data points
    {'bootstrap':False, 'n_estimators':[3,10], 'max_features':[2,3,4]}
]

# calls upon randomforestfunction
forest_reg = RandomForestRegressor()

# grid search then uses rfr as parameter for searching
grid_search = GridSearchCV(forest_reg, param_grid,cv=5,
                            scoring ='neg_mean_squared_error')

grid_search.fit(housing_prepared, housing_labels)

# shows the best parameter
>>>grid_search.best_params_
> {'max_features': 6, 'n_estimators': 30}

# calls upon the best estimators
>>>grid_search.best_estimator_
RandomForestRegressor(bootstrap=True, ccp_alpha=0.0, criterion='mse',
                      max_depth=None, max_features=6, max_leaf_nodes=None,
                      max_samples=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=1,
                      min_samples_split=2, min_weight_fraction_leaf=0.0,
                      n_estimators=30, n_jobs=None, oob_score=False,
                      random_state=None, verbose=0, warm_start=False)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score']):
    print(np.sqrt(-mean_score),params)

>
63753.5461528816 {'max_features': 2, 'n_estimators': 3}
56152.21937257319 {'max_features': 2, 'n_estimators': 10}
52795.21689339568 {'max_features': 2, 'n_estimators': 30}
60451.094331092594 {'max_features': 4, 'n_estimators': 3}
53192.357091620055 {'max_features': 4, 'n_estimators': 10}
50502.27548914166 {'max_features': 4, 'n_estimators': 30}
59451.09352442982 {'max_features': 6, 'n_estimators': 3}
52024.08066167432 {'max_features': 6, 'n_estimators': 10}
49977.58768754372 {'max_features': 6, 'n_estimators': 30}           <-```best estimator```
58376.64679593403 {'max_features': 8, 'n_estimators': 3}
52089.93695027458 {'max_features': 8, 'n_estimators': 10}
50309.329963579 {'max_features': 8, 'n_estimators': 30}
62494.38975288301 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
54586.08947518622 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
60092.32083232561 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
52607.240151793594 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
58543.89023752621 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
51816.00617104736 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}

```Analyzing the Best Models and Their errors```

feature_importances = grid_search.best_estimator_.feature_importances_
>>> feature_importances
>      array([7.88085467e-02, 7.33496388e-02, 4.09954065e-02, 1.67091258e-02,
       1.73554988e-02, 1.77318205e-02, 1.59227275e-02, 3.07956727e-01,
       5.38513930e-02, 1.10407460e-01, 9.81576768e-02, 7.59214979e-03,
       1.53346255e-01, 9.45085982e-05, 2.55826347e-03, 5.16280248e-03])

cat_one_hot_attribs = list(encoder.classes_)
sorted(zip(feature_importances, attibutes), reverse=True)

[(0.30795672652438416, 'median_income'),
 (0.15334625523523016, 'INLAND'),
 (0.11040745956098118, 'pop_per_hhold'),
 (0.09815767677490277, 'bedrooms_per_room'),
 (0.07880854666610768, 'longitude'),
 (0.07334963881963506, 'latitude'),
 (0.053851393042369335, 'rooms_per_hhold'),
 (0.040995406457984815, 'housing_median_age'),
 (0.017731820497973446, 'population'),
 (0.017355498758235867, 'total_bedrooms'),
 (0.016709125785504313, 'total_rooms'),
 (0.0159227275319888, 'households'),
 (0.007592149789429611, '<1H OCEAN'),               <- ```Remove```
 (0.005162802483529534, 'NEAR OCEAN'),              <- ```Remove```
 (0.002558263473579651, 'NEAR BAY'),                <- ```Remove```
 (9.450859816373917e-05, 'ISLAND')]

final_model = grid_search.best_estimator

X_test = strat_test_set.drop("median_house_value",axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
>final_rmse
>>>48078.53091897726



