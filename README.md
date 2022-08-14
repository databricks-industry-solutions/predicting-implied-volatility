<img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/fs-lakehouse-logo-transparent.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

## A day in the life of a Quantitative Researcher

*In this solution we will reproduce the most common tasks quantitative researchers perform - 1. developing new quantitative models like asset allocation or novel risk-adjusted performance metrics (to account for non-standard risk) using academic papers and 2. designing experiments to test these models.

We will implement the logic of the following academic paper (Deep Learning Volatility, 2019, Horvath et al), build the proposed model, and productionalize everything using various Databricks services (see the Architecture at the end of this notebook).*

___
<boris.banushev@databricks.com>

___


<img src='https://bbb-databricks-demo-assets.s3.amazonaws.com/IV_arch.png' style="float: left" width="1250px" />

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| Tensorflow                             | Machine Learning        | Apache 2.0 | https://github.com/tensorflow/tensorflow            |