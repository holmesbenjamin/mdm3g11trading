import bootstrapPrices as bsp
import basicFeatureExtraction as bfe
import matplotlib.pyplot as plt
import tsfresh as tsf
from tsfresh.utilities.dataframe_functions import roll_time_series

# --- Step 0: Set the working commodity and other constants --- #
working_commodity = "BRENT CRUDE"
working_days = 260

# --- Step 1: Bootstrap the prices of a commodity --- #

# Set the number of bootstrap samples
num_bootstrap_samples = 3

# Extract the prices and returns of the selected commodity
commodity_px = bsp.close_price_extraction(working_commodity)
commodity_returns = bsp.prices_to_returns(commodity_px)

# Generate stationary bootstrap samples
commodity_returns_samples = bsp.generate_stationary_bootstrap_sample(commodity_returns, num_bootstrap_samples)

# Convert the returns samples back to prices
commodity_initial_px = commodity_px.values[0]
commodity_px_samples = bsp.returns_to_prices(commodity_returns_samples, commodity_initial_px)

# # Plot the prices of the samples and the original commodity
# ax = commodity_px_samples.plot(figsize=(20, 4))
# ax = commodity_px.plot(figsize=(20, 4), ax=ax, linestyle="--", color='k', label=working_commodity)
# ax.legend()
# plt.show()

# --- Step 2: Extraction of features --- #
df = bfe.create_tsfresh_flat_dataframe(commodity_returns, f'{working_commodity} returns')
df_rolled = roll_time_series(df, column_id="id", min_timeshift=working_days, max_timeshift=working_days)
