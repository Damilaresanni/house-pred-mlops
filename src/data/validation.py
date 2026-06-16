import pandas as pd
import great_expectations as gx
from great_expectations.expectations import core as cr
import json


csv_path = "/home/fidisroxy/development/mlops/house-pred-mlops/data/raw/house_prices.csv"

dataframe = pd.read_csv(csv_path)

context = gx.get_context(mode="file")

data_source_name= "house_data"
data_source = context.data_sources.(name=data_source_name)

data_asset_name = "house_data_asset"
data_asset= data_source.add_dataframe_asset(name=data_asset_name)


batch_definition_name= "house_data_batch"
batch_definition = data_asset.add_batch_definition_whole_dataframe(
    batch_definition_name
)

batch_parameters = {"dataframe":dataframe}

batch = batch_definition.get_batch(batch_parameters=batch_parameters)



column_names = ['Title','Description','Amount(in rupees)','Price (in rupees)', 
                'location', 'Carpet Area','Status', 'Floor','Transaction',
                'Furnishing','facing','overlooking','Society','Bathroom','Balcony',
                'Car Parking', 'Ownership','Super Area']

is_exists = cr.expect_column_to_exist.ExpectColumnToExist(
   column= "Amount(in rupees)")
is_exists_price = cr.expect_column_to_exist.ExpectColumnToExist(
   column= "Price (in rupees)")
price_not_null = cr.expect_column_values_to_not_be_null.ExpectColumnValuesToNotBeNull(column="Price (in rupees)")



suite_name = "raw_expectation_suite"
suite = context.suites.add(
    gx.ExpectationSuite(
        name=suite_name,
        expectations=[is_exists,is_exists_price,price_not_null]  
    )
)





# Validate Batch.
validation_definition = context.validation_definitions.add(
    gx.ValidationDefinition(
        name="validation definition",
        data=batch_definition,
        suite=suite,
    )
)

update_docs_action = gx.checkpoint.actions.UpdateDataDocsAction(
    name="update_my_docs"
)

checkpoint = context.checkpoints.add(
    gx.Checkpoint(
        name="my_pipeline_checkpoint",
       validation_definitions=[validation_definition],
       actions=[update_docs_action]
    )
)

validation_result = checkpoint.run(
    batch_parameters={"dataframe":dataframe}
)   



print(f"Validation Successful: {validation_result.success}")  

context.view_validation_result(validation_result)   


                 
# Force Great Expectations to register and build 'local_site'
print("Configuring and building Data Docs...")
context.build_data_docs()

# Let's verify exactly where it ended up
docs_site_info = context.get_docs_sites_urls()
if docs_site_info:
    for site in docs_site_info:
        print(f"\n✅ SUCCESS! Site created at: {site['site_url']}")
else:
    print("\n❌ Site still empty. Forcing a direct configuration build...")
    # Fallback: Tell the context exactly where to put it
    context.view_validation_result(validation_result)