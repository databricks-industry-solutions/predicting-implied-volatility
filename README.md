<img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/fs-lakehouse-logo-transparent.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

# A day in the life of a Quantitative Researcher

In this solution we will reproduce the most common tasks quantitative researchers perform, namely 1. developing new quantitative models like asset allocation or novel risk-adjusted performance metrics (to account for non-standard risk) using academic papers and 2. designing experiments to test these models.

We will implement the logic of the following academic paper (_Deep Learning Volatility_, 2019, Horvath et al), build the proposed model, and productionalize everything using various Databricks services (see the Architecture at the end of this notebook).

The aim of the paper is the build a neural networks that is an off-line approximation of complex pricing functions, which are difficult to represent or time-consuming to evaluate by other means. In turn this solves the calibration bottleneck posed by a slow pricing of derivative contracts.

Link to the paper - https://arxiv.org/pdf/1901.09647.pdf

## Why Databricks Lakehouse for "Deep Learning Volatility"?

1. **Scale**: The burst capacity of [Databricks Runtime](https://docs.databricks.com/runtime/mlruntime.html) and [Photon](https://www.databricks.com/product/photon) can run this very computationally intensive synthetic data generation algorithm (from the paper) extremely quickly and in a cost-efficient way.
2. **DataOps - Feature Store**: [Databricks Feature Store](https://databricks.com/blog/2022/04/29/announcing-general-availability-of-databricks-feature-store.html) can keep these generated features in a highly efficient format (as Delta tables) making them ready for online and offline training, eliminating the need to re-run these algorithms many times and avoid additional costs.
3. **Collaboration between R and python**: After generating the synthetic data, we need to test the quality of the data and identify potential statistical issues, such as heteroskedasticity, as we will use this data to train regression models. For this purpose, we will use R packages, as R is specifically built for performing statistical tasks. This will visualize the simplicity of using python and R in the same [Interactive Databricks Notebook](https://databricks.com/product/collaborative-notebooks), and utilizing the strengths of each language (and supported libraries), without having to re-write R libraries or provision additional Notebooks or clusters.
    - We will also use the built-in dashboarding capabilities of the Databricks Notebook to visualize the pair-plot of the generated data, and
    - The automated *Data Profile* feature of the Databricks Notebooks will help us observe the distribution and overall quality of the generated data.
4. **MLOps - ML experiments and deployment**: This paper requires training many models at the same time. Keeping track of each model (hyper-param tunning, computation time, feature selection, and many others) can become very ineffective when handling so many models simultaneously. That is where [MLFlow](https://databricks.com/product/managed-mlflow)'s Experiments tracking comes to help and streamline model development (see Notebook *03_ML*).
5. **Productionalization**: Finally, we use [Databricks Workflows](https://databricks.com/blog/2022/05/10/introducing-databricks-workflows.html) to orchestrate the end-to-end execution and deployment. Databricks Workflows is the fully-managed orchestration service for all your data, analytics, and AI needs. Tight integration with the underlying lakehouse platform ensures you create and run reliable production workloads on any cloud while providing deep and centralized monitoring with simplicity for end-users (see Notebook *04_Productionalizing*).


___
<boris.banushev@databricks.com>

___


<img src='https://bbb-databricks-demo-assets.s3.amazonaws.com/IV_arch.png' style="float: left" width="1250px" />

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |
| Tensorflow                             | Machine Learning        | Apache 2.0 | https://github.com/tensorflow/tensorflow            |
| TF Quant Finance                       | Machine Learning        | Apache 2.0 | https://github.com/google/tf-quant-finance          |