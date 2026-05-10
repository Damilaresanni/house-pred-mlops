import pandas as pd

import great_expectations as gx

df = pd.read_csv("mlops/house-pred-mlops/data/raw/house_prices.csv")

context = gx.get_context()

data_source = context.data_sources.add_pandas("pandas")

data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")

batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

expectation = gx.expectations.ExpectColumnValuesToBeBetween (
    column="", min_value= 0, max_value=5, severity= "warning"
)