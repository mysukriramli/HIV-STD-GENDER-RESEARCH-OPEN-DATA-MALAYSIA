# HIV-STD-GENDER-RESEARCH-OPEN-DATA-MALAYSIA
HIV AND STD GENDER RESEARCH STUDY OPEN DATA MALAYSIA
# HIV-STD-Gender-Research-Malaysia-Open-Data

This project analyzes the gender disparity in HIV and STD incidence rates in Malaysia using open data. The analysis includes visualizations and statistical tests to understand the trends and differences between male and female incidence rates over time.

## Streamlit App

You can view the interactive visualizations and analysis on the Streamlit app: HIV-STD-Gender-Research-Malaysia-Open-Data
https://std-hiv-gender-researchpy-ecbsz3y4esut5hweabyspv.streamlit.app/

## Datasets

The following datasets are used in this project:
- **HIV Incidence Data**: sdg_03-3-1.parquet
- **STD State Data**: std_state.parquet
- **Household Income State Data**: hh_income_state.parquet

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/HIV-STD-Gender-Research-Malaysia-Open-Data.git
   cd HIV-STD-Gender-Research-Malaysia-Open-Data
Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install the required libraries:

pip install -r requirements.txt
Run the Streamlit app:

streamlit run HIV-STD-GENDER-RESEARCH-MALAYSIA-OPEN-DATA.py
Analysis
Gender Disparity in HIV Incidence
The analysis focuses on the gender disparity in HIV incidence rates over time. Key observations include:

Higher Incidence in Males: Males consistently show higher incidence rates compared to females.
Statistical Significance: The difference in incidence rates between males and females is statistically significant (p-value < 0.05).
Stable Ratio: The ratio of male to female incidence has remained relatively stable over the observed period.
Visualizations
The project includes various visualizations to illustrate the findings:

Incidence Over Time by Sex: Line plots showing the trends in incidence rates for males and females.
Incidence Distribution by Sex: Box plots comparing the distribution of incidence rates between males and females.
Male to Female Incidence Ratio: Line plots showing the ratio of male to female incidence rates over time.
Future Work
Further analysis can be conducted to explore:

External Factors: Investigate potential external factors influencing incidence rates, such as public health programs and changes in reporting methods.
Seasonality Patterns: Explore seasonality patterns in the data to understand fluctuations better.
Advanced Time Series Models: Use more advanced time series models or incorporate external regressors for improved forecasting accuracy.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

License
This project is licensed under the MIT License. See the LICENSE file for details.


### Discussion on the Topic

#### Gender Disparity in HIV and STD Incidence

The analysis of HIV and STD incidence rates in Malaysia reveals significant gender disparities. Males consistently show higher incidence rates compared to females. This disparity could be attributed to various factors, including differences in risk behaviors, access to healthcare, and social stigma.

#### Key Findings

1. **Higher Incidence in Males**: The data shows that males have a higher incidence rate of HIV and STDs compared to females. This could be due to higher engagement in high-risk behaviors such as unprotected sex and intravenous drug use.
2. **Statistical Significance**: The difference in incidence rates between males and females is statistically significant, indicating that the observed disparity is not due to random chance.
3. **Stable Ratio**: The ratio of male to female incidence has remained relatively stable over the observed period, suggesting consistent gender differences in exposure and vulnerability to HIV and STDs.

#### Implications

Understanding the gender disparity in HIV and STD incidence is crucial for developing targeted public health interventions. Efforts should focus on:
- **Education and Awareness**: Increasing awareness about safe sex practices and reducing stigma associated with HIV and STDs.
- **Access to Healthcare**: Improving access to testing, treatment, and preventive measures for both males and females.
- **Behavioral Interventions**: Addressing high-risk behaviors through targeted interventions and support programs.

#### Future Research

Future research should explore the underlying factors contributing to gender disparities in HIV and STD incidence. This includes examining the impact of socioeconomic factors, cultural norms, and public health policies. Additionally, advanced time series models and external regressors can be used to improve forecasting accuracy and understand the dynamics of incidence rates over time.
