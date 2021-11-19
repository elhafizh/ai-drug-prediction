# Import necessary libraries
import json
import joblib

import numpy as np
import pandas as pd
import streamlit as st

# Machine Learning 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

# Custom classes 
from .utils import isNumerical
import os

def app():
	"""This application helps in running machine learning models without having to write explicit code 
	by the user. It runs some basic models and let's the user select the X and y variables. 
	"""
	
	# Load the data 
	if 'main_data.csv' not in os.listdir('data'):
		st.markdown("Please upload data through `Upload Data` page!")
	else:
		df = pd.read_csv('data/main_data.csv')
		st.dataframe(df)
		st.write(f"**Variabel yang akan diprediksi :** {df.columns[-1]}")
		st.write(f"**Variabel yang akan digunakan untuk melakukan prediksi:** \
			 {list(df.columns[:-1])}")
		
		# Label Encoding
		label_list = ["Sex","BP","Cholesterol","Na_to_K","Drug"]
		for l in label_list:
			label_encoder(df, l)
		
		model_comparison = {
			"knn" : False,
			"svm" : False,
			"rf" : False
		}


		# Perform train test splits 
		st.markdown("#### Train Test Splitting")
		size = st.slider("Percentage of value division",
							min_value=0.1, 
							max_value=0.9, 
							step = 0.1, 
							value=0.8, 
							help="This is the value which will be used to divide the data for training and testing. Default = 80%")
		
		x = df.drop(["Drug"],axis=1)
		y = df.Drug

		x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state = 42, shuffle = True)

		y_train = y_train.values.reshape(-1,1)
		y_test = y_test.values.reshape(-1,1)

		x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=size, random_state=42)
		st.write("Jumlah training samples:", x_train.shape[0])
		st.write("Jumlah testing samples:", x_test.shape[0])

		# To store results of models
		result_dict_train = {}
		result_dict_test = {}

		x_var = st.radio("Pilih algoritma yang akan dipakai :",options=('KNN', 'SVM', 'Random Forest'))       

		"""KNN Classifier & GridSearchCV"""

		if x_var == "KNN":
			if st.button(f"Jalankan Algoritma {x_var}"):
				with st.spinner("Proses Training..."):
					grid = {'n_neighbors':np.arange(1,120),
							'p':np.arange(1,3),
							'weights':['uniform','distance']
						}

					knn = KNeighborsClassifier(algorithm = "auto")
					knn_cv = GridSearchCV(knn,grid,cv=5)
					knn_cv.fit(x_train,y_train)
				st.write("Hyperparameters:",knn_cv.best_params_)
				st.write("Train Score:",knn_cv.best_score_)
				st.write("Test Score:",knn_cv.score(x_test,y_test))
				result_dict_train["KNN GridSearch Train Score"] = knn_cv.best_score_
				result_dict_test["KNN GridSearch Test Score"] = knn_cv.score(x_test,y_test)
				model_comparison['knn'] = True
		if x_var == "Random Forest":
			if st.button(f"Jalankan Algoritma {x_var}"):
				with st.spinner("Proses Training..."):
					grid = {'n_estimators':np.arange(100,1000,100),
							'criterion':['gini','entropy']
						}

					rf = RandomForestClassifier(random_state = 42)
					rf_cv = GridSearchCV(rf,grid,cv=5)
					rf_cv.fit(x_train,y_train)
				st.write("Hyperparameters:",rf_cv.best_params_)
				st.write("Train Score:",rf_cv.best_score_)
				st.write("Test Score:",rf_cv.score(x_test,y_test))
				result_dict_train["Random Forest GridSearch Train Score"] = rf_cv.best_score_
				result_dict_test["Random Forest GridSearch Test Score"] = rf_cv.score(x_test,y_test)
				model_comparison['rf'] = True
		if x_var == "SVM":
			if st.button(f"Jalankan Algoritma {x_var}"):
				with st.spinner("Proses Training..."):
					grid = {
						'C':[0.01,0.1,1,10],
						'kernel' : ["linear","poly","rbf","sigmoid"],
						'degree' : [1,3,5,7],
						'gamma' : [0.01,1]
					}

					svm  = SVC ()
					svm_cv = GridSearchCV(svm, grid, cv = 5)
					svm_cv.fit(x_train,y_train)
				st.write("Hyperparameters:",svm_cv.best_params_)
				st.write("Train Score:",svm_cv.best_score_)
				st.write("Test Score:",svm_cv.score(x_test,y_test))
				result_dict_train["Random Forest GridSearch Train Score"] = svm_cv.best_score_
				result_dict_test["Random Forest GridSearch Test Score"] = svm_cv.score(x_test,y_test)
				model_comparison['svm'] = True
		
		if model_comparison["knn"] and model_comparison["svm"] and \
			model_comparison["rf"]:
			pass
				

def label_encoder(df, y):
    le = LabelEncoder()
    df[y] = le.fit_transform(df[y])