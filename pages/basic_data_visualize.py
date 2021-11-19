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

		st.markdown("## Basic Data Analysis")
		col1, col2 = st.columns(2)

		x_var = col1.radio("Pilih variabel x", options=df_analysis.columns)       

		with col2:
			y_var = st.selectbox(
				'Pilih Variabel y (label)',
				df_analysis.columns)
		
		if x_var and x_var != df_analysis.columns[-1]:
			if y_var == df_analysis.columns[-1]:
				# st.text(x_var, )
				# st.text(type(x_var))
				# st.text(y_var, )
				# st.text(type(y_var))
				xy_data_analysis(df_analysis, y_var, x_var)


def xy_data_analysis(df, valX, valY):
	if ((df.columns[0] == valY) or 
			(df.columns[4] == valY)):
		plt.figure(figsize=(9,5))
		sns.swarmplot(x = valX, y = valY, data = df)
		plt.legend(df[valX].value_counts().index)
		plt.title(f"{valY} -- {valX}")
		st.pyplot(plt)
	elif ((df.columns[1] == valY) or 
			(df.columns[2] == valY) or
			(df.columns[3] == valY) or
			(df.columns[5] == valY)):
		df_combine = df.groupby([valX, valY]).size().reset_index(name = "Count")
		plt.figure(figsize = (9,5))
		sns.barplot(x = valX,y= "Count", hue = valY,data = df_combine)
		plt.title(f"{valY} -- {valX}")
		st.pyplot(plt)