import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, shapiro, normaltest
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# --- Load the datasets using st.cache_data ---
@st.cache_data
def load_data():
    URL_DATA_1 = 'https://storage.dosm.gov.my/sdg/sdg_03-3-1.parquet'
    df1 = pd.read_parquet(URL_DATA_1)
    df1['date'] = pd.to_datetime(df1['date'])

    URL_DATA_2 = 'https://storage.data.gov.my/healthcare/std_state.parquet'
    df2 = pd.read_parquet(URL_DATA_2)
    df2['date'] = pd.to_datetime(df2['date'])

    URL_DATA_3 = 'https://storage.dosm.gov.my/hies/hh_income_state.parquet'
    df3 = pd.read_parquet(URL_DATA_3)
    df3['date'] = pd.to_datetime(df3['date'])
    return df1, df2, df3

df1, df2, df3 = load_data()

# --- State Name Standardization ---
def standardize_state_names(df, state_col='state'):
    if state_col in df.columns:
        df[state_col] = df[state_col].str.lower().str.replace(r'[^\w\s]', '', regex=True)
        df[state_col] = df[state_col].str.replace('n. sembilan', 'negeri sembilan', regex=False)
        df[state_col] = df[state_col].str.replace('w. persekutuan kuala lumpur', 'kuala lumpur', regex=False)
        df[state_col] = df[state_col].str.replace('w. persekutuan labuan', 'labuan', regex=False)
        df[state_col] = df[state_col].str.replace('w. persekutuan putrajaya', 'putrajaya', regex=False)
    return df

df2 = standardize_state_names(df2)
df3 = standardize_state_names(df3)

# --- Filter data from 2020 onwards ---
df1 = df1[df1['date'].dt.year >= 2020]
df2 = df2[df2['date'].dt.year >= 2020]
df3 = df3[df3['date'].dt.year >= 2020]

# --- Ensure 'state' exists in df1 by merging with df3 ---
df1 = df1.merge(df3[['state', 'date']], on='date', how='left')

# --- Aggregate yearly data ---
df1_both = df1[df1['sex'] == 'both'].copy()
df1_both = df1_both.rename(columns={'incidence': 'hiv_incidence'})
df1_yearly = df1_both.groupby(['date', 'state'], as_index=False)['hiv_incidence'].mean()
df3_yearly = df3.groupby(['date', 'state'], as_index=False)['income_mean'].mean()

# --- Merge datasets ---
df_combined = pd.merge(df1_yearly, df3_yearly, on=['date', 'state'], how='inner')

# --- Streamlit App ---
st.title("Malaysian Incidence Analysis")

# ... (EDA and Visualizations -  All the code you provided for visualizations goes here) ...

# ... (Heatmap, Bubble Chart, Dual-Axis - All the code you provided for these goes here) ...


# ---  Corrected and Improved Forecasting ---
st.subheader("3. Dual-Axis Line Chart with Future Projections")
if not df1_yearly.empty and not df3_yearly.empty:
    def forecast_series(df, value_col, periods=10):
        # Use a more robust forecasting method (e.g., ARIMA) if data allows
        try:
            model = ExponentialSmoothing(df[value_col], trend='add', seasonal=None, damped_trend=True).fit()
            future_years = list(range(df['date'].max().year + 1, df['date'].max().year + 1 + periods))
            forecast_values = model.forecast(periods)
            return pd.DataFrame({'year': future_years, value_col: forecast_values})
        except Exception as e: # Handle potential errors during model fitting
            st.error(f"Error during forecasting: {e}")
            return pd.DataFrame() # Return empty DataFrame to avoid further errors

    # Project HIV incidence and income_mean
    hiv_forecast = forecast_series(df1_yearly.groupby('date', as_index=False)['hiv_incidence'].mean(), 'hiv_incidence')
    income_forecast = forecast_series(df3_yearly.groupby('date', as_index=False)['income_mean'].mean(), 'income_mean')

    if not hiv_forecast.empty and not income_forecast.empty: # Check if forecasting was successful
        df1_full = pd.concat([df1_yearly.groupby('date', as_index=False)['hiv_incidence'].mean(), hiv_forecast])
        df3_full = pd.concat([df3_yearly.groupby('date', as_index=False)['income_mean'].mean(), income_forecast])

        fig = px.line(df3_full, x='year', y='income_mean', title='Projected Household Income vs HIV Incidence (2020-2035)')
        fig.add_scatter(x=df1_full['year'], y=df1_full['hiv_incidence'], name='HIV Incidence', mode='lines', yaxis='y2')
        fig.add_scatter(x=hiv_forecast['year'], y=hiv_forecast['hiv_incidence'], mode='lines', name='Projected HIV', line=dict(dash='dot'))
        fig.update_layout(yaxis2={'title': 'HIV Incidence', 'overlaying': 'y', 'side': 'right'})
        st.plotly_chart(fig)
    else:
        st.warning("Forecasting failed.  Could not generate dual-axis chart.")
else:
    st.warning("df1_yearly or df3_yearly is empty. Cannot create dual-axis chart.")
