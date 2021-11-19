import streamlit as st
import numpy as np
import pandas as pd
from pages import utils
import matplotlib.pyplot as plt
import seaborn as sns
import os
from statistics import mean, median, stdev, mode

def app():
    if 'main_data.csv' not in os.listdir('data'):
        st.markdown("Please upload data through `Upload Data` page!")
    else:
        df_analysis = pd.read_csv('data/main_data.csv')

        st.markdown("## Univariate Variable Analysis")

        var_analysis_option = st.selectbox(
            'Pilih Variabel',
            df_analysis.columns)
        
        univariate_variable_analysis(df_analysis, var_analysis_option)
        

def univariate_variable_analysis(df, var_analysis_option):

	colb1, colb2 = st.columns([3,1])

	with colb1:
		# Age distribution
		plt.figure(figsize = (9,5))
		if ((df.columns[0] == var_analysis_option) or 
				(df.columns[4] == var_analysis_option)):
			sns.distplot(df[var_analysis_option])
		elif ((df.columns[1] == var_analysis_option) or 
				(df.columns[2] == var_analysis_option) or
				(df.columns[3] == var_analysis_option) or
				(df.columns[5] == var_analysis_option)):
			sns.countplot(df[var_analysis_option])
		st.pyplot(plt)

	with colb2:
		if ((df.columns[0] == var_analysis_option) or 
				(df.columns[4] == var_analysis_option)):
			# st.text(mean(df[var_analysis_option]))
			var_analysis_option_df = pd.Series(
				[df[var_analysis_option].max(), df[var_analysis_option].min(),
				median(df[var_analysis_option]), mode(df[var_analysis_option]),
				mean(df[var_analysis_option]), stdev(df[var_analysis_option])],
				index=['max', 'min', 'median', 'mode', 'mean', 'stdev'])
			st.dataframe(var_analysis_option_df)
		elif ((df.columns[1] == var_analysis_option) or 
				(df.columns[2] == var_analysis_option) or
				(df.columns[3] == var_analysis_option) or
				(df.columns[5] == var_analysis_option)):
			st.table(df[var_analysis_option].value_counts())