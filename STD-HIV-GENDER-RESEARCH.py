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

# --- EDA and Visualizations for df1 ---
st.subheader("EDA for df1 (sdg_03-3-1) - Gender Disparity Focus")
st.write(df1.head())

sns.set(style="whitegrid")

#... (All the EDA code for df1 you provided goes here)...
# Line plot with Seaborn
st.subheader("Incidence Over Time by Sex (Seaborn)")
fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x='date', y='incidence', hue='sex', data=df1, palette='viridis', linewidth=2.5, ax=ax)
ax.set_title('Incidence Over Time by Sex', fontsize=16)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Incidence', fontsize=14)
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Sex', title_fontsize='13', fontsize='11')
fig.tight_layout()
st.pyplot(fig)

# Interactive line plot with Plotly
st.subheader("Interactive Incidence Over Time by Sex (Plotly)")
fig = px.line(df1, x='date', y='incidence', color='sex', title='Interactive Incidence Over Time by Sex',
             labels={'date': 'Date', 'incidence': 'Incidence', 'sex': 'Sex'},
             template='plotly_dark')
fig.update_layout(title_font_size=20, legend_title_font_size=15)
st.plotly_chart(fig)

# Box plot with Seaborn
st.subheader("Incidence Distribution by Sex (Seaborn)")
fig, ax = plt.subplots(figsize=(10, 8))
sns.boxplot(x='sex', y='incidence', data=df1, palette='viridis', ax=ax)
ax.set_title('Incidence Distribution by Sex', fontsize=16)
ax.set_xlabel('Sex', fontsize=14)
ax.set_ylabel('Incidence', fontsize=14)
fig.tight_layout()
st.pyplot(fig)

# Statistical Tests
st.subheader("Statistical Tests")
male_incidence = df1[df1['sex'] == 'male']['incidence']
female_incidence = df1[df1['sex'] == 'female']['incidence']

t_statistic, p_value = ttest_ind(male_incidence, female_incidence, equal_var=False)
st.write(f"T-test: t-statistic = {t_statistic}, p-value = {p_value}")

u_statistic, p_value = mannwhitneyu(male_incidence, female_incidence)
st.write(f"Mann-Whitney U test: u-statistic = {u_statistic}, p-value = {p_value}")

# Male to Female Ratio
st.subheader("Male to Female Incidence Ratio Over Time")
df1_pivot = df1.pivot_table(index='date', columns='sex', values='incidence', aggfunc='mean')
df1_pivot['male_female_ratio'] = df1_pivot['male'] / df1_pivot['female']

fig, ax = plt.subplots(figsize=(12, 8))
sns.lineplot(x=df1_pivot.index, y='male_female_ratio', data=df1_pivot, color='purple', linewidth=2.5, ax=ax)
ax.set_title('Male to Female Incidence Ratio Over Time', fontsize=16)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Male to Female Ratio', fontsize=14)
ax.tick_params(axis='x', rotation=45)
fig.tight_layout()
st.pyplot(fig)

# Incidence Rate by Sex and Year (Plotly)
st.subheader("Incidence Rate by Sex and Year (Plotly)")
df1['year'] = df1['date'].dt.year
incidence_by_sex_year = df1.groupby(['year', 'sex'])['incidence'].mean().reset_index()

fig = px.bar(incidence_by_sex_year, x='year', y='incidence', color='sex', title='Incidence Rate by Sex and Year',
             labels={'year': 'Year', 'incidence': 'Incidence', 'sex': 'Sex'},
             template='plotly_dark')
fig.update_layout(title_font_size=20, legend_title_font_size=15)
st.plotly_chart(fig)
# --- Additional Visualizations (Heatmap, Bubble Chart, Dual-Axis) ---

# --- 1. Heatmap for HIV Incidence vs. Household Income (State-wise) ---
st.subheader("1. Heatmap for HIV Incidence vs. Household Income (State-wise)")
if not df_combined.empty:
    plt.figure(figsize=(12, 6))
    pivot_table = df_combined.pivot(index='state', columns='date', values='hiv_incidence')
    sns.heatmap(pivot_table, cmap='coolwarm', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('HIV Incidence by State (2020 Onward)')
    plt.xlabel('Year')
    plt.ylabel('State')
    st.pyplot(plt)
else:
    st.warning("df_combined is empty. Check your data and merge operations.")

# --- 2. Animated Bubble Chart (State-wise Trends) ---
st.subheader("2. Animated Bubble Chart (State-wise Trends)")
if not df_combined.empty:
    fig = px.scatter(df_combined, x='income_mean', y='hiv_incidence', size='hiv_incidence', color='state',
                     animation_frame='date', animation_group='state',
                     title='HIV Incidence vs Household Income (2020-2035)',
                     labels={'income_mean': 'Household Income', 'hiv_incidence': 'HIV Incidence'})
    st.plotly_chart(fig)
else:
    st.warning("df_combined is empty. Cannot create bubble chart.")

# --- 3. Dual-Axis Line Chart with Future Projections ---
st.subheader("3. Dual-Axis Line Chart with Future Projections")
if not df1_yearly.empty and not df3_yearly.empty:
    def forecast_series(df, value_col, periods=10):
        model = ExponentialSmoothing(df[value_col], trend='add', seasonal=None, damped_trend=True).fit()
        future_years = list(range(df['date'].max().year + 1, df['date'].max().year + 1 + periods))
        forecast_values = model.forecast(periods)
        return pd.DataFrame({'year': future_years, value_col: forecast_values})

    # Project HIV incidence and income_mean
    hiv_forecast = forecast_series(df1_yearly.groupby('date', as_index=False)['hiv_incidence'].mean(), 'hiv_incidence')
    income_forecast = forecast_series(df3_yearly.groupby('date', as_index=False)['income_mean'].mean(), 'income_mean')

    df1_full = pd.concat([df1_yearly.groupby('date', as_index=False)['hiv_incidence'].mean(), hiv_forecast])
    df3_full = pd.concat([df3_yearly.groupby('date', as_index=False)['income_mean'].mean(), income_forecast])

    fig = px.line(df3_full, x='year', y='income_mean', title='Projected Household Income vs HIV Incidence (2020-2035)')
    fig.add_scatter(x=df1_full['year'], y=df1_full['hiv_incidence'], name='HIV Incidence', mode='lines', yaxis='y2')
    fig.add_scatter(x=hiv_forecast['year'], y=hiv_forecast['hiv_incidence'], mode='lines', name='Projected HIV', line=dict(dash='dot'))
    fig.update_layout(yaxis2={'title': 'HIV Incidence', 'overlaying': 'y', 'side': 'right'})
    st.plotly_chart(fig)
else:
    st.warning("df1_yearly or df3_yearly is empty. Cannot create dual-axis chart.")
