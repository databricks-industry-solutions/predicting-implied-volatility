# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # A day in the life of a Quantitative Researcher
# MAGIC In this solution we will reproduce the most common tasks quantitative researchers perform - 1. developing new quantitative models like asset allocation or novel risk-adjusted performance metrics (to account for non-standard risk) using academic papers and 2. designing experiments to test these models.
# MAGIC 
# MAGIC We will implement the logic of the following academic paper (Deep Learning Volatility, 2019, Horvath et al), build the proposed model, and productionalize everything using various Databricks services (see the Architecture at the end of this notebook).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # "Deep Learning Volatility"
# MAGIC ## *A deep neural network perspective on pricing and calibration in (rough) volatility models*
# MAGIC 
# MAGIC 
# MAGIC The aim of this paper is the build a neural networks that is an off-line approximation of complex pricing functions, which are difficult to represent or time-consuming to evaluate by other means. In turn this solves the calibration bottleneck posed by a slow pricing of derivative contracts.
# MAGIC 
# MAGIC <img src="https://bbb-databricks-demo-assets.s3.amazonaws.com/Screenshot+2022-07-24+at+5.09.41+PM.png" style="float: left" width="750px" />
# MAGIC 
# MAGIC Link to the paper - **https://arxiv.org/pdf/1901.09647.pdf**

# COMMAND ----------

# MAGIC %md
# MAGIC # Why Databricks Lakehouse for "Deep Learning Volatility"?
# MAGIC 
# MAGIC 1. **Scale**: The burst capacity of [Databricks Runtime](https://docs.databricks.com/runtime/mlruntime.html) and [Photon](https://www.databricks.com/product/photon) can run this very computationally intensive synthetic data generation algorithm (from the paper) extremely quickly and in a cost-efficient way.
# MAGIC 2. **DataOps - Feature Store**: [Databricks Feature Store](https://databricks.com/blog/2022/04/29/announcing-general-availability-of-databricks-feature-store.html) can keep these generated features in a highly efficient format (as Delta tables) making them ready for online and offline training, eliminating the need to re-run these algorithms many times and avoid additional costs.
# MAGIC 3. **Collaboration between R and python**: After generating the synthetic data, we need to test the quality of the data and identify potential statistical issues, such as heteroskedasticity, as we will use this data to train regression models. For this purpose, we will use R packages, as R is specifically built for performing statistical tasks. This will visualize the simplicity of using python and R in the same [Interactive Databricks Notebook](https://databricks.com/product/collaborative-notebooks), and utilizing the strengths of each language (and supported libraries), without having to re-write R libraries or provision additional Notebooks or clusters.
# MAGIC     - We will also use the built-in dashboarding capabilities of the Databricks Notebook to visualize the pair-plot of the generated data, and
# MAGIC     - The automated *Data Profile* feature of the Databricks Notebooks will help us observe the distribution and overall quality of the generated data.
# MAGIC 4. **MLOps - ML experiments and deployment**: This paper requires training many models at the same time. Keeping track of each model (hyper-param tunning, computation time, feature selection, and many others) can become very ineffective when handling so many models simultaneously. That is where [MLFlow](https://databricks.com/product/managed-mlflow)'s Experiments tracking comes to help and streamline model development (see Notebook *Implied Volatility Prediction - 2. ML*).
# MAGIC 5. **Productionalization**: Finally, we use [Databricks Workflows](https://databricks.com/blog/2022/05/10/introducing-databricks-workflows.html) to orchestrate the end-to-end execution and deployment. Databricks Workflows is the fully-managed orchestration service for all your data, analytics, and AI needs. Tight integration with the underlying lakehouse platform ensures you create and run reliable production workloads on any cloud while providing deep and centralized monitoring with simplicity for end-users (see Notebook *Implied Volatility Prediction - 4. Productionalizing*).

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC # Architecture
# MAGIC 
# MAGIC This is the architecture of the solutions we will build.
# MAGIC 
# MAGIC <img src='https://bbb-databricks-demo-assets.s3.amazonaws.com/IV_arch.png' style="float: left" width="1250px" />
