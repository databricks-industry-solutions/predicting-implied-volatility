# Databricks notebook source
import pyspark.pandas as ps
from databricks import feature_store
import mlflow
import databricks.automl_runtime
import time

from mlflow.tracking import MlflowClient
import os
import uuid
import shutil
import pandas as pd

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load Data from the Databricks Feature Store

# COMMAND ----------

fs = feature_store.FeatureStoreClient()
features_df = fs.read_table('feature_store_implied_volatility.features')
labels_df = fs.read_table('feature_store_implied_volatility.labels')

# COMMAND ----------

features_df = features_df.toPandas()
labels_df =  labels_df.toPandas()

# COMMAND ----------

features_df = features_df.iloc[:, 1:]
features_df['target'] = labels_df['0.05_0.95']

# COMMAND ----------

target_col = "target"
training_col = list(features_df.columns)[:-1]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Select supported columns
# MAGIC Select only the columns that are supported. This allows us to train a model that can predict on a dataset that has extra columns that are not used in training.
# MAGIC `[]` are dropped in the pipelines. See the Alerts tab of the AutoML Experiment page for details on why these columns are dropped.

# COMMAND ----------

from databricks.automl_runtime.sklearn.column_selector import ColumnSelector
supported_cols = training_col
col_selector = ColumnSelector(supported_cols)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preprocessors

# COMMAND ----------

transformers = []

# COMMAND ----------

# MAGIC %md
# MAGIC ### Numerical columns
# MAGIC 
# MAGIC Missing values for numerical columns are imputed with mean by default.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

num_imputers = []
num_imputers.append(("impute_mean", SimpleImputer(), training_col))

numerical_pipeline = Pipeline(steps=[
    ("converter", FunctionTransformer(lambda df: df.apply(pd.to_numeric, errors="coerce"))),
    ("imputers", ColumnTransformer(num_imputers)),
    ("standardizer", StandardScaler()),
])

transformers.append(("numerical", numerical_pipeline, training_col))

# COMMAND ----------

from sklearn.compose import ColumnTransformer

preprocessor = ColumnTransformer(transformers, remainder="passthrough", sparse_threshold=0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Train - Validation - Test Split
# MAGIC Split the input data into 3 sets:
# MAGIC - Train (60% of the dataset used to train the model)
# MAGIC - Validation (20% of the dataset used to tune the hyperparameters of the model)
# MAGIC - Test (20% of the dataset used to report the true performance of the model on an unseen dataset)

# COMMAND ----------

from sklearn.model_selection import train_test_split

split_X = features_df.drop([target_col], axis=1)
split_y = features_df[target_col]

# Split out train data
X_train, split_X_rem, y_train, split_y_rem = train_test_split(split_X, split_y, train_size=0.6, random_state=224145758)

# Split remaining data equally for validation and test
X_val, X_test, y_val, y_test = train_test_split(split_X_rem, split_y_rem, test_size=0.5, random_state=224145758)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Train regression model
# MAGIC - Log relevant metrics to MLflow to track runs
# MAGIC - All the runs are logged under [this MLflow experiment](#mlflow/experiments/3624174420594729/s?orderByKey=metrics.%60val_r2_score%60&orderByAsc=false)
# MAGIC - Change the model parameters and re-run the training cell to log a different trial to the MLflow experiment
# MAGIC - To view the full list of tunable hyperparameters, check the output of the cell below

# COMMAND ----------

from xgboost import XGBRegressor

# COMMAND ----------

import mlflow
import sklearn
from sklearn import set_config
from sklearn.pipeline import Pipeline

set_config(display='diagram')

xgb_regressor = XGBRegressor(
  colsample_bytree=0.6385875217228281,
  learning_rate=0.10603131742006,
  max_depth=6,
  min_child_weight=8,
  n_estimators=148,
  n_jobs=100,
  subsample=0.5203076979604147,
  verbosity=0,
  random_state=224145758,
)

model = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
    ("regressor", xgb_regressor),
])

# Create a separate pipeline to transform the validation dataset. This is used for early stopping.
pipeline = Pipeline([
    ("column_selector", col_selector),
    ("preprocessor", preprocessor),
])

mlflow.sklearn.autolog(disable=True)
pipeline.fit(X_train, y_train)
X_val_processed = pipeline.transform(X_val)

# COMMAND ----------

try:
  username = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
except:
  username = str(uuid.uuid1()).replace("-", "")

# COMMAND ----------

# Enable automatic logging of input samples, metrics, parameters, and models
mlflow.sklearn.autolog(log_input_examples=True, silent=True)

#experiment_id_ = mlflow.create_experiment("Implied Volatility Prediction")
experiment_name = experiment_name = f'/Users/{username}/implied_volatility'
#mlflow.set_experiment(experiment_name)
try:
  experiment_id_ = mlflow.create_experiment(experiment_name)
except:
  experiment_id_ = mlflow.get_experiment_by_name(experiment_name).experiment_id

with mlflow.start_run(experiment_id=experiment_id_, run_name=f"implied_volatility_{time.time()}") as mlflow_run:
    model.fit(X_train, y_train, regressor__early_stopping_rounds=5, regressor__eval_set=[(X_val_processed,y_val)], regressor__verbose=False)
    
    # Training metrics are logged by MLflow autologging
    # Log metrics for the validation set
    xgb_val_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_val, y_val, prefix="val_")

    # Log metrics for the test set
    xgb_test_metrics = mlflow.sklearn.eval_and_log_metrics(model, X_test, y_test, prefix="test_")

    # Display the logged metrics
    xgb_val_metrics = {k.replace("val_", ""): v for k, v in xgb_val_metrics.items()}
    xgb_test_metrics = {k.replace("test_", ""): v for k, v in xgb_test_metrics.items()}
    display(pd.DataFrame([xgb_val_metrics, xgb_test_metrics], index=["validation", "test"]))

# COMMAND ----------

display(spark.read.format("mlflow-experiment").load(experiment_id_))

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ### MLFlow UI
# MAGIC 
# MAGIC MLFlow has a UI component, making tracking of experiments extremely easy.
# MAGIC 
# MAGIC <img src='https://bbb-databricks-demo-assets.s3.amazonaws.com/Screenshot+2022-07-29+at+1.55.57+PM.png'  style="float: left" width="1150px" />

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Feature importance
# MAGIC 
# MAGIC SHAP is a game-theoretic approach to explain machine learning models, providing a summary plot
# MAGIC of the relationship between features and model output. Features are ranked in descending order of
# MAGIC importance, and impact/color describe the correlation between the feature and the target variable.
# MAGIC - Generating SHAP feature importance is a very memory intensive operation, so to ensure that AutoML can run trials without
# MAGIC   running out of memory, we disable SHAP by default.<br />
# MAGIC   You can set the flag defined below to `shap_enabled = True` and re-run this notebook to see the SHAP plots.
# MAGIC - To reduce the computational overhead of each trial, a single example is sampled from the validation set to explain.<br />
# MAGIC   For more thorough results, increase the sample size of explanations, or provide your own examples to explain.
# MAGIC - SHAP cannot explain models using data with nulls; if your dataset has any, both the background data and
# MAGIC   examples to explain will be imputed using the mode (most frequent values). This affects the computed
# MAGIC   SHAP values, as the imputed samples may not match the actual data distribution.
# MAGIC 
# MAGIC For more information on how to read Shapley values, see the [SHAP documentation](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html).

# COMMAND ----------

# Set this flag to True and re-run the notebook to see the SHAP plots
shap_enabled = False

# COMMAND ----------

if shap_enabled:
    from shap import KernelExplainer, summary_plot
    # Sample background data for SHAP Explainer. Increase the sample size to reduce variance.
    train_sample = X_train.sample(n=min(100, X_train.shape[0]))

    # Sample some rows from the validation set to explain. Increase the sample size for more thorough results.
    example = X_val.sample(n=min(10, X_val.shape[0]))

    # Use Kernel SHAP to explain feature importance on the example from the validation set.
    predict = lambda x: model.predict(pd.DataFrame(x, columns=X_train.columns))
    explainer = KernelExplainer(predict, train_sample, link="identity")
    shap_values = explainer.shap_values(example, l1_reg=False)
    summary_plot(shap_values, example)
