# Time Series Modeling & Forecasting: ARIMA vs. LSTM

A comparative time-series forecasting case study evaluating a traditional statistical model, **ARIMA**, and a deep-learning model, **Long Short-Term Memory (LSTM)**. The study uses two time-series datasets: Amazon (AMZN) stock closing prices and the `jj` dataset. It explores preprocessing, stationarity, model training, forecast reconstruction, evaluation, and approximately 24-month future forecasts.

## Overview

Time-series forecasting uses historical observations to estimate future values. This project compares two approaches with different strengths:

- **ARIMA** models temporal relationships in a differenced series.
- **LSTM** uses a recurrent neural network to learn temporal patterns from scaled sequential data.

The models are evaluated using **Root Mean Squared Error (RMSE)** and **Mean Absolute Error (MAE)**.

## Datasets

| Dataset | Forecast target | Frequency | Approximate 24-month forecast horizon |
|---|---|---|---:|
| AMZN | `Close` | Business days | 504 steps |
| `jj` | `data` series | Daily | 720 steps |

The date field is converted to datetime and used as the time-series index. Missing values are checked and handled using forward filling. The report identifies 45 missing values in each AMZN column and 7,496 missing values in the `jj` target series before filling.

## Project Workflow

### 1. Data preparation and exploration

- Convert the date column to datetime and set it as the index.
- Select AMZN's `Close` series and the `jj` dataset's target series.
- Inspect missing values and forward-fill gaps.
- Use seasonal decomposition to inspect trend, seasonal, and residual components.
- Apply the Augmented Dickey-Fuller (ADF) test to assess stationarity.

### 2. Stationarity and differencing

Both series were identified as non-stationary in the case study. First-order differencing was applied:

\[
y'_t = y_t - y_{t-1}
\]

A second ADF test was used to check the differenced series for stationarity.

### 3. ARIMA forecasting

- Split the time series into training and test sets using an 80/20 split.
- Apply first-order differencing manually before fitting the model.
- Fit an ARIMA model with order **(1, 0, 1)** to the differenced series. The integration parameter is `d=0` because differencing was performed outside the model.
- Forecast on the differenced scale.
- Reconstruct forecasts on the original scale using cumulative summation and the last observed original value.
- Generate approximately 24 months of future forecasts: around 504 business-day steps for AMZN and 720 daily steps for `jj`.

### 4. LSTM forecasting

- Scale the series with `MinMaxScaler`.
- Prepare sequential inputs with a **60-time-step sliding window**.
- Train a **two-layer LSTM network** with a Dense output layer on 80% of the data.
- Use early stopping to help limit overfitting.
- Transform predictions back to the original scale.
- Generate approximately 24 months of future predictions recursively.

### 5. Evaluation

Compare predictions with actual observations and calculate RMSE and MAE. Forecast plots are used to inspect model behavior on the test period and over the future forecast horizon.

## Results

The following metrics are reported in the case-study document:

| Dataset | Model | RMSE | MAE |
|---|---|---:|---:|
| AMZN | ARIMA | 54.410055 | 47.797833 |
| AMZN | LSTM | 6.056306 | 4.710926 |
| `jj` | ARIMA | 3.375728 | 2.690548 |
| `jj` | LSTM | 0.351524 | 0.146346 |

Lower RMSE and MAE indicate smaller prediction errors for the evaluation represented by these results. Because the datasets have different scales, their error values should not be compared directly across datasets.

### Interpretation

- **AMZN:** LSTM recorded lower RMSE and MAE than ARIMA. The report notes that ARIMA's reconstructed long-horizon forecasts were affected by error accumulation during reintegration.
- **`jj`:** LSTM recorded lower values in the reported metrics table. The report also describes ARIMA as performing comparably on this more regular, seasonal series.
- **Overall:** The case study illustrates that model performance depends on the characteristics of the series. LSTM can model complex non-linear patterns, while ARIMA remains a useful statistical baseline for more regular time series.

These results reflect the specific evaluation described in the case study and do not guarantee similar performance on other data.

## ARIMA vs. LSTM

| Aspect | ARIMA | LSTM |
|---|---|---|
| Model category | Statistical time-series model | Recurrent neural network |
| Input preparation | First-order differencing performed manually | Series scaled with `MinMaxScaler` |
| Sequence window | Not applicable | 60 time steps |
| Forecast reconstruction | Cumulative sum of differenced forecasts plus the last observed original value | Inverse scaling and recursive prediction |
| Consideration | Reintegration can accumulate forecast errors over long horizons | Requires sequence preparation and neural-network training |

## Repository Structure

The project can be organized as follows. The structure intentionally keeps the README, data, notebook, and generated forecast plots together without a separate requirements file or a copy of the report document.

```text
time-series-forecasting/
├── README.md
├── data/
│   ├── amzn.csv
│   └── jj.csv
├── notebooks/
│   └── time_series_forecasting.ipynb
└── outputs/
    └── forecast_plots/
```

## Running the Analysis

1. Open the forecasting notebook in Jupyter Notebook, JupyterLab, or Google Colab.
2. Place the AMZN and `jj` datasets in the `data/` directory.
3. Run the notebook cells in order, from data loading and preprocessing through model training, evaluation, and future forecasting.
4. Review the evaluation metrics and forecast plots produced by the notebook.

The analysis uses Python time-series and machine-learning tools, including Pandas, NumPy, Matplotlib, Statsmodels, Scikit-learn, and TensorFlow/Keras. The notebook's imports define the exact packages used by the implementation.

## Evaluation Metrics

- **RMSE (Root Mean Squared Error):** The square root of the mean of squared prediction errors. Larger errors receive a greater penalty.
- **MAE (Mean Absolute Error):** The mean of the absolute differences between predicted and actual values.

Both metrics are expressed in the target series' units. Consider the scale and characteristics of each dataset when interpreting the values.

## Limitations

- Forecast uncertainty generally increases farther into the future.
- Recursive LSTM forecasts can propagate prediction errors across future steps.
- ARIMA forecasts reconstructed from differenced predictions can accumulate errors during reintegration.
- Results depend on data quality, preprocessing, the train/test split, model configuration, and evaluation procedure.
- Stock-price forecasts should not be treated as financial advice or used as the sole basis for investment decisions.

## References

1. Box, G. E. P., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015). *Time Series Analysis: Forecasting and Control* (5th ed.). Wiley.
2. Hyndman, R. J., & Athanasopoulos, G. (2018). *Forecasting: Principles and Practice* (2nd ed.). OTexts.
3. Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
4. Brownlee, J. (2018). *Deep Learning for Time Series Forecasting*. Machine Learning Mastery.

---

**Project focus:** Time-series forecasting and comparative modeling  
**Models:** ARIMA and LSTM  
**Evaluation metrics:** RMSE and MAE
