# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/on-shelf-availability. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/on-shelf-availability.

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to forecast sales over an historical period and then use those forecasted values to identify potential on-shelf availability concerns.  This notebook has been developed by [Tredence](https://www.tredence.com/) in partnership with Databricks.

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as f

import pandas as pd
import numpy as np
import math
from datetime import timedelta
	
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

# COMMAND ----------

# MAGIC %md ## Step 1: Access Data
# MAGIC 
# MAGIC In the last notebook, we identified inventory problems through the detection of excessive phantom inventory, stocking levels below safety stock thresholds and unexpected numbers of consecutive days with zero sales.  We might label these *out-of-stock* issues in that they identify scenarios where product is simply not available to be sold.
# MAGIC 
# MAGIC In this notebook, we want to add a fourth inventory scenario, one where insufficient stock may not fully prevent sales but where they may cause us to miss our sales expectations.  Misplacement of product in the store or displays which give the customer a sense a product is not available are both examples of the kinds of issues we might describe as *on-shelf availability* problems.
# MAGIC 
# MAGIC With this fourth scenario, we will generate a forecast for sales and identify historical sales values that were depressed relative to what was expected. These periods of lower than expected sales may then be investigated as periods potentially experiencing OSA challenges. To generate this forecast, we must first access our historical sales data:

# COMMAND ----------

# DBTITLE 1,Read the Input Datasets
inventory_flagged = spark.table('osa.inventory_flagged')

# COMMAND ----------

# MAGIC %md ## Step 2: Generate Forecast
# MAGIC 
# MAGIC Unlike most forecasting exercises, our goal is not to predict future values but instead to generate *expected* values for the historical period.  To do this, we may make use of a variety of forecasting techniques. Most enterprises already have established preferences for sales forecasting so instead of wading into the conversation about which techniques are best in different scenarios, we will make use of a [simple exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing) as a placeholder technique so that we might focus on the analysis against the forecasted values in later steps.
# MAGIC 
# MAGIC Our challenge now is to generate a forecast for each store-SKU combination in our dataset.  Leveraging a forecast scaling technique [previously demonstrated](https://databricks.com/blog/2021/04/06/fine-grained-time-series-forecasting-at-scale-with-facebook-prophet-and-apache-spark-updated-for-spark-3.html), we will write a function capable of generating a forecast for a given store-SKU combination and then apply it to all store-SKU combinations in our dataset in a scalable, distributed manner:

# COMMAND ----------

# DBTITLE 1,Define Function to Generate Forecast for a Store-SKU
alpha_value = 0.8 # smoothing factor

# function to generate a forecast for a store-sku
def get_forecast(keys, inventory_pd: pd.DataFrame) -> pd.DataFrame:
  
  # identify store and sku
  store_id = keys[0]
  sku = keys[1]
  
  # identify date range for predictions
  history_start = inventory_pd['date'].min()
  history_end = inventory_pd['date'].max()
  
  # organize data for model training
  timeseries = (
    inventory_pd
      .set_index('date', drop=True, append=False) # move date to index
      .sort_index() # sort on date-index
    )['total_sales_units'] # just need this one field
  
  # fit model to timeseries
  model = SimpleExpSmoothing(timeseries, initialization_method='heuristic').fit(smoothing_level=alpha_value)
  
  # predict sales across historical period
  predictions = model.predict(start=history_start, end=history_end)
  
  # convert timeseries to dataframe for return
  predictions_pd = predictions.to_frame(name='predicted_sales_units').reset_index() # convert to df
  predictions_pd.rename(columns={'index':'date'}, inplace=True) # rename 'index' column to 'date'
  predictions_pd['store_id'] = store_id # assign store id
  predictions_pd['sku'] = sku # assign sku
  
  return predictions_pd[['date', 'store_id', 'sku', 'predicted_sales_units']]

# structure of forecast function output
forecast_schema = StructType([
  StructField('date', DateType()), 
  StructField('store_id', IntegerType()), 
  StructField('sku', IntegerType()), 
  StructField('predicted_sales_units', FloatType())
  ])

# COMMAND ----------

# DBTITLE 1,Generate Forecasts for All Store-SKUs
# get forecasted values for each store-sku combination
forecast = (
  inventory_flagged
    .groupby(['store_id','sku'])
      .applyInPandas(
        get_forecast, 
        schema=forecast_schema
        )
    .withColumn('predicted_sales_units', f.expr('ROUND(predicted_sales_units,0)')) # round values to nearest integer
    )

display(forecast)

# COMMAND ----------

# DBTITLE 1,Persist Forecasts
(
  forecast
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema', 'true')
    .saveAsTable('osa.inventory_forecast')
  )

# COMMAND ----------

# MAGIC %md ## Step 3: Identify *Off* Sales Issues
# MAGIC 
# MAGIC With forecasts in-hand, we will now look for historical periods where there is not only a lower than expected number of sales (relative to our forecasts) but where this difference grows over a number of days. Identifying these periods may help us identify on-shelf availability (OSA) concerns we may need to address.  
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/osa_tredence_offsales.jpg' width=75%>
# MAGIC 
# MAGIC Of course, not every missed sales target is an OSA event.  To focus our attention, we will look for periods of sustained misses where the miss is sizeable relative to our expectations.  In the code that follows, we require 4-days of increasing misses with an average daily miss of 20% or more of the expected sales. Some organizations may wish to increase or decrease these threshold requirements depending on the nature of their business:

# COMMAND ----------

# DBTITLE 1,Flag Off-Sales Events
inventory_forecast = spark.table('osa.inventory_forecast')

osa_flag_output = (
  
  inventory_flagged.alias('inv')
    .join(inventory_forecast.alias('for'), on=['store_id','sku','date'], how='leftouter')
    .selectExpr(
      'inv.*',
      'for.predicted_sales_units'
      )
             
    # calculating difference between forecasted and actual sales units
    .withColumn('units_difference', f.expr('predicted_sales_units - total_sales_units'))
    .withColumn('units_difference', f.expr('COALESCE(units_difference, 0)'))

    # check whether deviation has been increasing over past 4 days
    .withColumn('osa_alert_inc_deviation', f.expr('''
      CASE 
        WHEN units_difference > LAG(units_difference, 1) OVER(PARTITION BY store_id, sku ORDER BY date) AND 
             LAG(units_difference, 1) OVER(PARTITION BY store_id, sku ORDER BY date) > LAG(units_difference, 2) OVER(PARTITION BY store_id, sku ORDER BY date) AND 
             LAG(units_difference, 2) OVER(PARTITION BY store_id, sku ORDER BY date) > LAG(units_difference, 3) OVER(PARTITION BY store_id, sku ORDER BY date)
             THEN 1
        ELSE 0 
        END'''))
    .withColumn('osa_alert_inc_deviation', f.expr('COALESCE(osa_alert_inc_deviation, 0)'))

    # rolling 4 day average of sales units
    .withColumn('sales_4day_avg', f.expr('AVG(total_sales_units) OVER(PARTITION BY store_id, sku ORDER BY date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)'))

    # rolling 4 day average of forecasted units
    .withColumn('predictions_4day_avg', f.expr('AVG(predicted_sales_units) OVER(PARTITION BY store_id, sku ORDER BY date ROWS BETWEEN 3 PRECEDING AND CURRENT ROW)'))

    # calculating deviation in rolling average of sales and forecast units
    .withColumn('deviation', f.expr('(predictions_4day_avg - sales_4day_avg) / (predictions_4day_avg+1)'))
    .withColumn('deviation', f.expr('COALESCE(deviation, 0)'))

    # Considering 20% deviation as the threshold for OSA flag
    .withColumn('off_sales_alert', f.expr('''
      CASE 
        WHEN deviation > 0.20  AND osa_alert_inc_deviation = 1 THEN 1
        ELSE 0
        END'''))

    .select('date', 
            'store_id', 
            'sku', 
            'predicted_sales_units', 
            'off_sales_alert',
            'oos_alert', 
            'zero_sales_flag', 
            'phantom_inventory', 
            'phantom_inventory_ind')
    )

display(osa_flag_output)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the [Databricks License](https://databricks.com/db-license-source).  All included or referenced third party libraries are subject to the licenses set forth below.
