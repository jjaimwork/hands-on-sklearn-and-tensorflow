

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

'''Sklearns
Linear regression 
Random Forest
Decision Tree

from here we call the functions
fit the housing_prepared, and housing_labels data (X_train,y_train)

'''



'''

typically use a cost function that measures the distance between the linear model’s predictions and the training examples; the objective is to minimize this distance.
 Possible solutions for overfitting are:
to simplify the model
 constrain it (i.e., regularize it)
 or get a lot more training data
You should save every model you experiment with, so you can come back easily to any model you want. Make sure you save both the hyperparameters and the trained parameters, as well as the cross-validation scores and perhaps the actual predictions as well. The goal is to shortlist a few (two to five) promising models.  You can easily save Scikit-Learn models by using Python’s pickle module
'''