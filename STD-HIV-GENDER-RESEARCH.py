import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Load the dataset (using st.cache_data)
@st.cache_data  # Cache the data loading
def load_data():
    URL_DATA_1 = 'https://storage.dosm.gov.my/sdg/sdg_03-3-1.parquet'
    df1 = pd.read_parquet(URL_DATA_1)
    df1['date'] = pd.to_datetime(df1['date'])
    return df1

df1 = load_data()

# Remove "Both" category (do this *after* loading/caching)
df1 = df1[df1['sex']!= 'Both']

# --- Streamlit App ---
st.title("Malaysian Incidence Analysis (Gender Disparity Focus)")

# --- EDA for df1 ---
st.subheader("EDA for df1 (sdg_03-3-1) - Gender Disparity Focus")
st.write(df1.head())  # Use st.write for displaying DataFrames

# Set Seaborn style (do this once at the beginning)
sns.set(style="whitegrid")

# Line plot with Seaborn
st.subheader("Incidence Over Time by Sex (Seaborn)")
fig, ax = plt.subplots(figsize=(12, 8)) # Create figure and axes
sns.lineplot(x='date', y='incidence', hue='sex', data=df1, palette='viridis', linewidth=2.5, ax=ax) # Pass the axes to seaborn
ax.set_title('Incidence Over Time by Sex', fontsize=16)
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Incidence', fontsize=14)
ax.tick_params(axis='x', rotation=45)
ax.legend(title='Sex', title_fontsize='13', fontsize='11')
fig.tight_layout() # use the figure to adjust the layout
st.pyplot(fig) # use st.pyplot to display the figure

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
