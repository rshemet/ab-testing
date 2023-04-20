import streamlit as st
import pandas as pd
import numpy as np
# import scipy
from scipy.stats import norm
from matplotlib import pyplot as plt

# st.set_page_config(layout="wide")

tab1, tab2 = st.tabs(["Test planninng", "Significance calculator"])



def binomial_z_score(p1, p2, n1, n2):
    "Returns the z-score for a two-(binomial) sample test of population means."
    p = ((n1*p1) + (n2*p2))/(n1+n2)
    z = (p1 - p2) / np.sqrt((p*(1-p))*(1/n1+1/n2))
    return z

z_lookup = {
    0.8: 1.28,
    0.9: 1.645,
    0.95: 1.96,
    0.99: 2.576,
}

with tab1:

    st.write('# AB Test Sample Size Calculator')
    st.write('Calculate the necessary sample size to detect a given change in conversion rate with a given confidence level.')
    st.write('### Inputs')

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
        1 + 0.002 * x: binomial_z_score(
            target_metric * (1 + 0.002 * x),
            target_metric,
            sample_size, 
            pop_size, 
        ) for x in range(1, 1001)
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


with tab2:

    st.write('# AB Test Results Significance Calculator')
    st.write('Calculate the significance of a given change in conversion rate with a given confidence level.')
    st.write('### Inputs')

    test_size_input_2 = st.number_input(label='Test Sample Size', min_value=1, step=1, value=1000)
    control_size_input_2 = st.number_input(label='Control Sample Size', min_value=1, step=1, value=2000)
    test_metric_input_2 = st.number_input(label='Test conversion rate (%)', min_value=0.01, max_value=100., step=0.01, value=5., format="%.2f")
    control_metric_input_2 = st.number_input(label='Control conversion rate (%)', min_value=0.01, max_value=100., step=0.01, value=5., format="%.2f")
    # tails = st.selectbox('Test type', options=('one-tailed', 'two-tailed'))

    # factor = 2 if tails == 'one-tailed' else 1

    z_score = binomial_z_score(
        p1=test_metric_input_2 / 100, 
        p2=control_metric_input_2 / 100, 
        n1=test_size_input_2, 
        n2=control_size_input_2,
    )

    p_value = (1 - (norm.cdf(z_score))) * 2

    s = ''
    for ci, z in z_lookup.items():
        s += f'\n- {ci*100:.0f}% CI: {"🟢" if abs(z_score) > z else "❌"}'

    st.write('### Results')
    st.markdown(f'For the set of inputs above, the z-score is: {z_score:.2f} (p={p_value:.3f}). The significance thresholds are: {s}')

    shade_percent = 1 - p_value
    fig, ax = plt.subplots()

    lower_bound = -4
    upper_bound = 4

    x = np.linspace(lower_bound, upper_bound, 1000)
    y = norm.pdf(x, 0, 1)
    cv = norm.ppf(1 - (1 - shade_percent) / 2)

    ax.plot(x, y, color='black', linewidth=0.5)
    ax.fill_between(x, y, where=(x >= cv) & (x <= upper_bound), color='gray', alpha=0.3)
    ax.fill_between(x, y, where=(x >= lower_bound) & (x <= -cv), color='gray', alpha=0.3)

    ax.set_xlabel('X')
    ax.set_ylabel('PDF')
    ax.set_title(f'Tail probability: {1 - shade_percent:.2%}')

    st.pyplot(fig)