import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

def binomial_z_score(p1, p2, n1, n2):
    "Returns the z-score for a two-(binomial) sample test of population means."
    p = ((n1*p1) + (n2*p2))/(n1+n2)
    z = (p1 - p2) / np.sqrt((p*(1-p))*(1/n1+1/n2))
    return z

st.write('# AB Test Sample Size Calculator')
st.write('Calculate the necessary sample size to detect a given change in conversion rate with a given confidence level.')
st.write('### Inputs')

z_lookup = {
    0.8: 1.28,
    0.9: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}

sample_size_input = st.number_input(label='Test Size', min_value=1, step=1, value=1000)
pop_size_input = st.number_input(label='Control Size', min_value=1, step=1, value=2000)
target_metric_input = st.slider(label='Control Conversion Rate', min_value=0.01, max_value=100., step=0.01, value=5., format="%.2f%%",)
# target_metric_input = st.number_input(label='Population target metric (%)', min_value=0.01, max_value=100., step=0.01, value=5., format="%.2f")

confidence_intervals = st.multiselect(
    label='Confidence Intervals', 
    options=[0.8, 0.9, 0.95, 0.99], 
    default=[0.95], 
    format_func=lambda x: f'{x * 100}%', 
)

pop_size = pop_size_input
sample_size = sample_size_input
target_metric = target_metric_input / 100

df = pd.DataFrame({
    1 + 0.02 * x: binomial_z_score(
        target_metric * (1 + 0.02 * x),
        target_metric,
        sample_size, 
        pop_size, 
    ) for x in range(1, 101)
}, index=['TestZScore']).T

df['TestConversionRate'] = target_metric * (df.index)
df['TestConversionRateXAxis'] = (df['TestConversionRate'] * 100)#.round(2)

for ci in confidence_intervals:
    df[f'{ci * 100:.0f}% Confidence Interval'] = z_lookup[ci]

chg_lookup = {ci: df[df['TestZScore'] >= z]['TestConversionRate'].min() for ci, z in z_lookup.items() if ci in confidence_intervals}
s = ''
for k, v in chg_lookup.items():
    s += f"\n- {k*100:.0f}% CI: {v*100:.2f}%"

st.write('### Results')
st.markdown(f'For a Test group of {sample_size} and a Control group of {pop_size} with base conversion rate of {round(target_metric * 100,2)}%, the required change to conversion rate is: {s}')
st.write('### Chart')
st.write('Y-axis: z-score. X-axis: test conversion rate')

st.line_chart(
    data=df.set_index('TestConversionRateXAxis')[['TestZScore'] + [x for x in df.columns if 'Confidence Interval' in x]],
    height=600,
)