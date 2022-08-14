# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Orchestrate everything together
# MAGIC 
# MAGIC Once we have the logic of both data generation (using the academic paper) and model training (using MLFlow) we need to be able to productionalize everything together.
# MAGIC 
# MAGIC ### Why do we use Databricks Workflows?
# MAGIC 
# MAGIC 1. Workflows runs diverse workloads for the full data and AI lifecycle on any cloud. Orchestrate Delta Live Tables and Jobs for SQL, Spark, notebooks, dbt, ML models and more.
# MAGIC 2. An easy point-and-click authoring experience for all your data teams, not just those with specialized skills.
# MAGIC 3. Have full confidence in your workflows leveraging our proven experience running tens of millions of production workloads daily across AWS, Azure and GCP.
# MAGIC 4. Remove operational overhead with a fully managed orchestration service, so you can focus on your workflows not on managing your infrastructure.

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC <img src="https://bbb-databricks-demo-assets.s3.amazonaws.com/workflows+for+IV.jpg" style="float: left" width="1100px" />

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC ## How to programmatically create and run a Databricks Workflow
# MAGIC 
# MAGIC #### Prerequisites
# MAGIC 
# MAGIC After creating your cluster, make sure to install the ```tf_quant_finance``` and ```tensorflow==2.9.0``` libraries.

# COMMAND ----------

import dbclient
import json, pprint, requests

# COMMAND ----------

workflow_name = 'Implied Volatility Predication on Databricks'
cluster_name = '<<ENTER_THE_NAME_OF_YOUR_CLUSTER>>'
feature_engineering_notebook = '/Users/<<ENTER_YOUR_USERNAME>>/Implied Volatility Prediction - 2. Create Features'
ml_notebook = '/Users/<<ENTER_YOUR_USERNAME>>/Implied Volatility Prediction - 3. ML'
spark_version = '11.0.x-gpu-ml-scala2.12'
note_type = 'g4dn.2xlarge'
cron_sch = '20 30 * * * ?'
feature_engineering_task = 'implied_volatility_data_generation'
ml_task = 'implied_volatility_ml'
retry_on_timeout = False
max_workers = 4
tags = {}

# COMMAND ----------


j1_param = {
    "name": workflow_name,
    "tags": tags,
    "tasks": [
        # Feature engineering task
        {
            "task_key":feature_engineering_task,
            "description":"Generate the synthetic data and store into Databricks Feature Store",
            "depends_on": [],
            
            "job_cluster_key": cluster_name,
    
            "notebook_task":{
                "notebook_path": feature_engineering_notebook,
                "base_parameters":{}
            },
            "retry_on_timeout": retry_on_timeout
        },
        # ML prediction task
        {
            "task_key": ml_task,
            "description":"Use MLFlow to train a ML model to predict the IV",
            "depends_on": [{"task_key": feature_engineering_task}],
            
            "job_cluster_key": cluster_name,

            "notebook_task":{
                "notebook_path": ml_notebook,
                "base_parameters":{}
            },
            "retry_on_timeout": retry_on_timeout
        }
    ],
    
    # Use the aforementioned cluster
    "job_clusters":[
        {
            "job_cluster_key": cluster_name,
            "new_cluster":{
                "spark_version": spark_version,
                "node_type_id": note_type
            ,
            "autoscale":{
                "min_workers": 1,
                "max_workers": max_workers
                },
            }
        }
    ],
    
    "email_notifications":{},
    "schedule": {
        "quartz_cron_expression": cron_sch,
        "timezone_id": "Asia/Bangkok",
        "pause_status": "PAUSED"
    },
    "max_concurrent_runs": 1,
    "format": "MULTI_TASK"

}

# COMMAND ----------

api_client = dbclient(token, url)
api_response = api_client.create_job(job_param=j1_param)
print(api_response)

# COMMAND ----------

job_id = api_response['job_id']
print(job_id)
r1 = {
    "job_id": job_id,
    "notebook_params":{}
}

# COMMAND ----------

api_client.run_job(job_param=r1)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Workflow monitoring
# MAGIC 
# MAGIC As your organization creates data and ML workflows, it becomes imperative to manage and monitor them without needing to deploy additional infrastructure. Workflows integrates with existing resource access controls in Databricks, enabling you to easily manage access across departments and teams. Additionally, Databricks Workflows includes native monitoring capabilities so that owners and managers can quickly identify and diagnose problems.
# MAGIC 
# MAGIC <img src="https://bbb-databricks-demo-assets.s3.amazonaws.com/workflows+for+IV+-+2.jpg" style="float: left" width="1000px" />
