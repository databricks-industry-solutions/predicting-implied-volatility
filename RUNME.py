# Databricks notebook source
# MAGIC %md This notebook sets up the companion cluster(s) to run the solution accelerator. It also creates the Workflow to create a Workflow DAG and illustrate the order of execution. Feel free to interactively run notebooks with the cluster or to run the Workflow to see how this solution accelerator executes. Happy exploring!
# MAGIC 
# MAGIC The pipelines, workflows and clusters created in this script are user-specific, so you can alter the workflow and cluster via UI without affecting other users. Running this script again after modification resets them.
# MAGIC 
# MAGIC **Note**: If the job execution fails, please confirm that you have set up other environment dependencies as specified in the accelerator notebooks. Accelerators sometimes require the user to set up additional cloud infra or data access, for instance. 

# COMMAND ----------

# DBTITLE 0,Install util packages
# MAGIC %pip install git+https://github.com/databricks-academy/dbacademy-rest git+https://github.com/databricks-academy/dbacademy-gems git+https://github.com/databricks-industry-solutions/notebook-solution-companion

# COMMAND ----------

from solacc.companion import NotebookSolutionCompanion

# COMMAND ----------

job_json = {
        "timeout_seconds": 7200,
        "max_concurrent_runs": 1,
        "tags": {
            "usage": "solacc_automation",
            "group": "FSI_solacc_automation"
        },
        "tasks": [
            {
                "job_cluster_key": "PIV_cluster",
                "libraries": [],
                "notebook_task": {
                    "notebook_path": f"01_introduction"
                },
                "task_key": "PIV_01",
                "description": ""
            },
            {
                "job_cluster_key": "PIV_cluster",
                "notebook_task": {
                    "notebook_path": f"02_create_features"
                },
                "task_key": "PIV_02",
                "depends_on": [
                    {
                        "task_key": "PIV_01"
                    }
                ]
            },
            {
                "job_cluster_key": "PIV_cluster",
                "notebook_task": {
                    "notebook_path": f"03_train_models"
                },
                "task_key": "PIV_03",
                "depends_on": [
                    {
                        "task_key": "PIV_02"
                    }
                ]
            }
        ],
        "job_clusters": [
            {
                "job_cluster_key": "PIV_cluster",
                "new_cluster": {
                    "spark_version": '11.0.x-gpu-ml-scala2.12',
                "spark_conf": {
                    "spark.databricks.delta.formatCheck.enabled": "false"
                    },
                    "num_workers": 4,
                    "node_type_id": {"AWS": "g4dn.2xlarge", "MSA": "Standard_NC4as_T4_v3", "GCP": "a2-highgpu-1g"},
                    "custom_tags": {
            "usage": "solacc_automation",
            "group": "FSI_solacc_automation"
        },
                }
            }
        ]
    }

# COMMAND ----------

dbutils.widgets.dropdown("run_job", "False", ["True", "False"])
run_job = dbutils.widgets.get("run_job") == "True"
NotebookSolutionCompanion().deploy_compute(job_json, run_job=run_job)

# COMMAND ----------


