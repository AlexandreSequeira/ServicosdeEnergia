# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 13:47:33 2023

@author: Alexandre Sequeira
"""

import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Load data
df = pd.read_csv('test_data_2019.csv')
df['Date'] = pd.to_datetime (df['Date']) # create a new column 'data time' of datetime type
#df = df.set_index('Date') # make 'datetime' into index

df2=df.iloc[:,2:5]
X2=df2.values
fig = px.line(df, x="Date", y=df.columns[1:5])

df_real=df.drop(['hour','Hollk','Power-1'],axis=1)
y2=df_real['Power_kW'].values

#Load and run models


#Random Forest
with open('Models/RF_model.pkl','rb') as file:
    RF_model2=pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

###Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)

#Bootstrapping

with open('Models/BT_model.pkl','rb') as file:
    BT_model=pickle.load(file)

y2_pred_BT = BT_model.predict(X2)

###Evaluate errors
MAE_BT=metrics.mean_absolute_error(y2,y2_pred_BT) 
MSE_BT=metrics.mean_squared_error(y2,y2_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y2)

#Decision Tree

with open('Models/DT_regr_model.pkl','rb') as file:
    DT_model=pickle.load(file)

y2_pred_DT = DT_model.predict(X2)

###Evaluate errors
MAE_DT=metrics.mean_absolute_error(y2,y2_pred_DT) 
MSE_DT=metrics.mean_squared_error(y2,y2_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y2,y2_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y2)


#Gradient Boosting

with open('Models/GB_model.pkl','rb') as file:
    GB_model=pickle.load(file)

y2_pred_GB = GB_model.predict(X2)

###Evaluate errors
MAE_GB=metrics.mean_absolute_error(y2,y2_pred_GB) 
MSE_GB=metrics.mean_squared_error(y2,y2_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y2)

#Extreme Gradient Boosting

with open('Models/XGB_model.pkl','rb') as file:
    XGB_model=pickle.load(file)

y2_pred_XGB = XGB_model.predict(X2)

###Evaluate errors
MAE_XGB=metrics.mean_absolute_error(y2,y2_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y2,y2_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y2,y2_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y2)

#Neural Networks

with open('Models/NN_model.pkl','rb') as file:
    NN_model=pickle.load(file)

y2_pred_NN = NN_model.predict(X2)

###Evaluate errors
MAE_NN=metrics.mean_absolute_error(y2,y2_pred_NN) 
MSE_NN=metrics.mean_squared_error(y2,y2_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y2,y2_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y2)

#Linear Regression

with open('Models/LR_model.pkl','rb') as file:
    LR_model=pickle.load(file)

y2_pred_LR = LR_model.predict(X2)

###Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)


d = {'Methods': ['Linear Regression','Random Forest','Bootstrapping','Neural Networks','Gradient Boosting','Extreme Gradient Boosting','Decision Tree'], 'MAE': [MAE_LR, MAE_RF,MAE_BT,MAE_NN,MAE_GB,MAE_XGB,MAE_DT], 'MSE': [MSE_LR, MSE_RF,MSE_BT,MSE_NN,MSE_GB,MSE_XGB,MSE_DT], 'RMSE': [RMSE_LR, RMSE_RF,RMSE_BT,RMSE_NN,RMSE_GB,RMSE_XGB,RMSE_DT],'cvRMSE': [cvRMSE_LR, cvRMSE_RF,cvRMSE_BT,cvRMSE_NN,cvRMSE_GB,cvRMSE_XGB,cvRMSE_DT]}
df_metrics = pd.DataFrame(data=d)
d={'Date':df_real['Date'].values, 'Linear Regression': y2_pred_LR,'Random Forest': y2_pred_RF, 'Bootstrapping': y2_pred_BT, 'Neural Networks': y2_pred_BT, 'Gradient Boosting': y2_pred_GB, 'Extreme Gradient Boosting': y2_pred_XGB, 'Decision Tree': y2_pred_DT}
df_forecast=pd.DataFrame(data=d)
df_results=pd.merge(df_real,df_forecast, on='Date')



fig2 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:9])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.H2('IST Energy Forecast tool (kWh)'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Model Errors', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H3('IST Raw Data'),
            dcc.Graph(
                id='yearly-data',
                figure=fig,
            ),
            html.H6("""The meaning of the different values are:
                    Power_kW: average consumption of energy during that hour;
                    Power-1: average consumption in the previous hour;
                    Hollk:if August,Saturday,Sunday or holiday 1, else 0;
                    Hour: hour of the day;
                    """),
            
        ])
    if tab == 'tab-2':
        return html.Div([
            html.H3('IST Electricity Forecast (kWh)'),
            dcc.Graph(
                id='yearly-data',
                figure=fig2,
                ),
            
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Methods Errors'),
            generate_table(df_metrics)
        ])


if __name__ == '__main__':
    app.run_server(debug=False)








