import os
import pandas as pd
import pymc as pm
import numpy as np
import arviz as az
import pytensor
from tqdm import tqdm  # Import tqdm for progress bar

pytensor.config.on_opt_error = "warn"
pytensor.config.cxx = ""


# Define a function to encapsulate the analysis logic
def run_analysis():
    # Step 1: Load all CSV files from the folder
    folder_path = r"C:\Users\Akyf Thanish\Desktop\PR PRoject\t"
    print("Loading CSV files from the folder...")
    all_data = []

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            data = pd.read_csv(file_path)
            all_data.append(data)

    # Concatenate all data into a single DataFrame
    df = pd.concat(all_data, ignore_index=True)
    print("Loaded data from {} files.".format(len(all_data)))

    # Step 2: Data Preprocessing
    print("Preprocessing data...")
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"])

    # Scale the data
    df = df[["STATION", "TEMP", "DEWP", "PRCP", "WDSP", "MXSPD"]].dropna()
    df["TEMP"] = (df["TEMP"] - df["TEMP"].mean()) / df["TEMP"].std()
    df["DEWP"] = (df["DEWP"] - df["DEWP"].mean()) / df["DEWP"].std()
    df["PRCP"] = (df["PRCP"] - df["PRCP"].mean()) / df["PRCP"].std()
    df["WDSP"] = (df["WDSP"] - df["WDSP"].mean()) / df["WDSP"].std()
    df["MXSPD"] = (df["MXSPD"] - df["MXSPD"].mean()) / df["MXSPD"].std()

    stations = df["STATION"].astype("category").cat.codes.values
    n_stations = len(np.unique(stations))
    print("Number of stations:", n_stations)

    # Extract the data to model
    temperature = df["TEMP"].values
    dew_point = df["DEWP"].values
    precipitation = df["PRCP"].values
    wind_speed = df["WDSP"].values
    max_wind_speed = df["MXSPD"].values

    # Step 3: Building the Bayesian Hierarchical Model
    print("Building the Bayesian Hierarchical Model...")
    with pm.Model() as hierarchical_model:
        # Tighter priors
        mu_temp_group = pm.Normal("mu_temp_group", mu=0, sigma=1)
        sigma_temp_group = pm.HalfNormal("sigma_temp_group", sigma=1)

        mu_dewp_group = pm.Normal("mu_dewp_group", mu=0, sigma=1)
        sigma_dewp_group = pm.HalfNormal("sigma_dewp_group", sigma=1)

        mu_prcp_group = pm.Normal("mu_prcp_group", mu=0, sigma=1)
        sigma_prcp_group = pm.HalfNormal("sigma_prcp_group", sigma=1)

        mu_wdsp_group = pm.Normal("mu_wdsp_group", mu=0, sigma=1)
        sigma_wdsp_group = pm.HalfNormal("sigma_wdsp_group", sigma=1)

        mu_temp = pm.Normal(
            "mu_temp", mu=mu_temp_group, sigma=sigma_temp_group, shape=n_stations
        )
        sigma_temp = pm.HalfNormal("sigma_temp", sigma=1, shape=n_stations)

        mu_dewp = pm.Normal(
            "mu_dewp", mu=mu_dewp_group, sigma=sigma_dewp_group, shape=n_stations
        )
        sigma_dewp = pm.HalfNormal("sigma_dewp", sigma=1, shape=n_stations)

        mu_prcp = pm.Normal(
            "mu_prcp", mu=mu_prcp_group, sigma=sigma_prcp_group, shape=n_stations
        )
        sigma_prcp = pm.HalfNormal("sigma_prcp", sigma=1, shape=n_stations)

        mu_wdsp = pm.Normal(
            "mu_wdsp", mu=mu_wdsp_group, sigma=sigma_wdsp_group, shape=n_stations
        )
        sigma_wdsp = pm.HalfNormal("sigma_wdsp", sigma=1, shape=n_stations)

        # Observed variables
        temp_obs = pm.Normal(
            "temp_obs",
            mu=mu_temp[stations],
            sigma=sigma_temp[stations],
            observed=temperature,
        )
        dewp_obs = pm.Normal(
            "dewp_obs",
            mu=mu_dewp[stations],
            sigma=sigma_dewp[stations],
            observed=dew_point,
        )
        prcp_obs = pm.Normal(
            "prcp_obs",
            mu=mu_prcp[stations],
            sigma=sigma_prcp[stations],
            observed=precipitation,
        )
        wdsp_obs = pm.Normal(
            "wdsp_obs",
            mu=mu_wdsp[stations],
            sigma=sigma_wdsp[stations],
            observed=wind_speed,
        )

        # Sampling from the posterior with better initialization
        print("Sampling from the posterior...")
        trace = pm.sample(
            500,
            return_inferencedata=True,
            cores=2,
            tune=500,
            init="auto",
            progressbar=True,
        )

    # Step 4: Posterior Analysis and Visualization
    print("Analyzing the posterior...")
    summary = az.summary(trace)
    print(summary)

    print("Plotting trace...")
    az.plot_trace(trace)

    # Posterior predictive check (optional)
    with hierarchical_model:
        print("Generating posterior predictive checks...")
        posterior_predictive = pm.sample_posterior_predictive(trace)

    # Updated line to use the correct function for ArviZ
    az.plot_ppc(az.from_pymc(trace, posterior_predictive=posterior_predictive))
    print("Analysis complete!")


# Ensure the script runs only if it's the main module
if __name__ == "__main__":
    run_analysis()
