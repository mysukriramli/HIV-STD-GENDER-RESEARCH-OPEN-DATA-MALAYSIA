import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu

# Load the dataset
URL_DATA_1 = 'https://storage.dosm.gov.my/sdg/sdg_03-3-1.parquet'
df1 = pd.read_parquet(URL_DATA_1)
df1['date'] = pd.to_datetime(df1['date'])

# Remove "Both" category
df1 = df1[df1['sex'] != 'Both']

# --- EDA for df1 (sdg_03-3-1) - Gender Disparity Focus ---
st.write("\n--- EDA for df1 (sdg_03-3-1) - Gender Disparity Focus ---")
st.write(df1.head())

# Set Seaborn style
sns.set(style="whitegrid")

# Line plot with Seaborn
plt.figure(figsize=(12, 8))
sns.lineplot(x='date', y='incidence', hue='sex', data=df1, palette='viridis', linewidth=2.5)
plt.title('Incidence Over Time by Sex', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Incidence', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Sex', title_fontsize='13', fontsize='11')
plt.tight_layout()
st.pyplot(plt)

# Interactive line plot with Plotly
fig = px.line(df1, x='date', y='incidence', color='sex', title='Interactive Incidence Over Time by Sex',
              labels={'date': 'Date', 'incidence': 'Incidence', 'sex': 'Sex'},
              template='plotly_dark')
fig.update_layout(title_font_size=20, legend_title_font_size=15)
st.plotly_chart(fig)

# Box plot with Seaborn
plt.figure(figsize=(10, 8))
sns.boxplot(x='sex', y='incidence', data=df1, palette='viridis')
plt.title('Incidence Distribution by Sex', fontsize=16)
plt.xlabel('Sex', fontsize=14)
plt.ylabel('Incidence', fontsize=14)
plt.tight_layout()
st.pyplot(plt)

male_incidence = df1[df1['sex'] == 'male']['incidence']
female_incidence = df1[df1['sex'] == 'female']['incidence']

t_statistic, p_value = ttest_ind(male_incidence, female_incidence, equal_var=False)
st.write(f"\nT-test: t-statistic = {t_statistic}, p-value = {p_value}")

u_statistic, p_value = mannwhitneyu(male_incidence, female_incidence)
st.write(f"\nMann-Whitney U test: u-statistic = {u_statistic}, p-value = {p_value}")

# Calculate male to female ratio
df1_pivot = df1.pivot_table(index='date', columns='sex', values='incidence', aggfunc='mean')
df1_pivot['male_female_ratio'] = df1_pivot['male'] / df1_pivot['female']

# Line plot for male to female ratio
plt.figure(figsize=(12, 8))
sns.lineplot(x=df1_pivot.index, y='male_female_ratio', data=df1_pivot, color='purple', linewidth=2.5)
plt.title('Male to Female Incidence Ratio Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Male to Female Ratio', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
st.pyplot(plt)

df1['year'] = df1['date'].dt.year
incidence_by_sex_year = df1.groupby(['year', 'sex'])['incidence'].mean().reset_index()

# Bar plot with Plotly
fig = px.bar(incidence_by_sex_year, x='year', y='incidence', color='sex', title='Incidence Rate by Sex and Year',
             labels={'year': 'Year', 'incidence': 'Incidence', 'sex': 'Sex'},
             template='plotly_dark')
fig.update_layout(title_font_size=20, legend_title_font_size=15)
st.plotly_chart(fig)
