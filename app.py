#Import packages
import numpy as np
import pandas as pd
import streamlit as st

import plotly.graph_objs as go
from plotly.subplots import make_subplots

from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics


import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from mlforecast import MLForecast
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean

from neuralforecast import NeuralForecast
from neuralforecast.models import NBEATS, NHITS


import itertools

st.set_page_config(
    page_title="Trafic Dashboard",
    page_icon="✅",
    layout="wide",
)

#Import the data
df = pd.read_parquet('data/traffic_10lines.parquet')
df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.date

df1 = df.drop_duplicates(subset=['home_airport', 'paired_airport'])
df2 = df.copy()
df2.set_index('date', inplace=True)

#Liste home airport dans la base de données
list_home_air = list(df1['home_airport'].value_counts().index)

#Liste paired airport dans la base de donnée
list_paired_air = list(df1['paired_airport'].value_counts().index)

combinations = list(itertools.product(list_home_air, list_paired_air))

#%%
#FUNCTION


def draw_ts_multiple(df: pd.DataFrame, v1: str, v2: str=None, prediction: str=None, date: str='date'or 'ds',
              secondary_y=True, covid_zone=False, display=True):
    """
    Draw times series possibly on two y axis.
    Args:
    - df (pd.DataFrame): time series dataframe (one line per date, series in columns)
    - v1 (str | list[str]): name or list of names of the series to plot on the first x axis
    - v2 (str): name of the serie to plot on the second y axis (default: None)
    - prediction (str): name of v1 hat (prediction) displayed with a dotted line (default: None)
    - date (str): name of date column for time (default: 'date')
    - secondary_y (bool): use a secondary y axis if v2 is used (default: True)
    - covid_zone (bool): highlight COVID-19 period with a grayed rectangle (default: False)
    - display (bool): display figure otherwise just return the figure (default: True)

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Plotly figure generated

    Notes:
    Make sure to use the semi-colon trick if you don't want to have the figure displayed twice.
    Or use `display=False`.
    """
    if isinstance(v1, str):
        variables = [(v1, 'V1')]
    else:
        variables = [(v, 'V1.{}'.format(i)) for i, v in enumerate(v1)]
    title = '<br>'.join([n + ': '+ v for v, n in variables]) + ('<br>V2: ' + v2) if v2 else '<br>'.join([v + ': '+ n for v, n in variables])
    layout = dict(
    title=title,
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label='1m',
                     step='month',
                     stepmode='backward'),
                dict(count=6,
                     label='6m',
                     step='month',
                     stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(
            visible = True
        ),
        type='date'
    ), width=1000, height=600
  )
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.update_layout(layout)
    for v, name in variables:
        fig.add_trace(go.Scatter(x=df[date], y=df[v], name=name), secondary_y=False)
        if v2:
            fig.add_trace(go.Scatter(x=df[date], y=df[v2], name='V2'), secondary_y=secondary_y)
            fig['layout']['yaxis2']['showgrid'] = False
            fig.update_yaxes(rangemode='tozero')
            fig.update_layout(margin=dict(t=125 + 30 * (len(variables) - 1)))
        if prediction:
            fig.add_trace(go.Scatter(x=df[date], y=df[prediction], name='^V1', line={'dash': 'dot'}), secondary_y=False)

        if covid_zone:
            fig.add_vrect(
                x0=pd.Timestamp("2020-03-01"), x1=pd.Timestamp("2022-01-01"),
                fillcolor="Gray", opacity=0.7,
                layer="below", line_width=0.9,
            )
        if display:
            st.plotly_chart(fig, theme="streamlit", use_container_width=True)
    return fig


#Define a function to display a trafic graph
def air_flux(home_airport : str, paired_airport: str, input_data: pd.DataFrame):
    
    """
    Display a number of passenger per date
    
    Parameters:
    - homeAirport (str): IATA Code for home airport
    - pairedAirport (str): IATA Code for paired airport

    Returns:
    - Ploly Graph: aggregated daily PAX traffic on route (home-paired)
    """
    
    plot = draw_ts_multiple(
        (input_data
         .query('home_airport == "{}" and paired_airport == "{}"'.format(home_airport, paired_airport))
         .groupby(['home_airport', 'paired_airport', 'date'])
         .agg(pax_total=('pax', 'sum'))
         .reset_index()
        ),
        'pax_total',
        covid_zone=True,
    )
    
    return plot

#Define a function 

def generate_route_df(traffic_df: pd.DataFrame, homeAirport: str, pairedAirport: str) -> pd.DataFrame:
    
    """
    Extract route dataframe from traffic dataframe for route from home airport to paired airport

    Args:
    - traffic_df (pd.DataFrame): traffic dataframe
    - homeAirport (str): IATA Code for home airport
    - pairedAirport (str): IATA Code for paired airport

    Returns:
    - pd.DataFrame: aggregated daily PAX traffic on route (home-paired)
    """
    _df = (traffic_df
         .query('home_airport == "{home}" and paired_airport == "{paired}"'.format(home=homeAirport, paired=pairedAirport))
         .groupby(['home_airport', 'paired_airport', 'date'])
         .agg(pax_total=('pax', 'sum'))
         .reset_index()
         )
    return _df

def forecast_plot(df_forecast, true_data ):
   
    
    trace_open = go.Scatter(
    x = df_forecast["ds"],
    y = df_forecast["yhat"],
    mode = 'lines',
    line = {"color": "green"},
    name="Forecast")

    trace_close = go.Scatter(
        x = true_data["ds"],
        y = true_data["y"],
        mode ="lines",
        line = {"color": "blue"},
        name="Data values"
    )
    
    data_for = [trace_open, trace_close]

    layout = go.Layout(title="Passenger Flux Forecast",xaxis_rangeslider_visible=True, width=1200, height=600)
    
    fig = go.Figure(data=data_for,layout=layout)
    
    return fig

#%%

st.markdown('<h1 class="font"> <strong>  Welcome to the Airport Passenger Traffic Analysis</strong></h1>', unsafe_allow_html=True)

#Define a selection menu
with st.sidebar:
    
    st.markdown('<h1 class="font"> <strong> Selection of route </strong></h1>', unsafe_allow_html=True)
    home_airport = st.selectbox(
        '**Home Airport**', list_home_air)
    paired_airport = st.selectbox(
        '**Paired Airport**', list_paired_air)
    
    st.markdown('<h1 class="font"> <strong> Selection of dates </strong></h1>', unsafe_allow_html=True)
    start_date = st.date_input('**Select the start date**:', value = df2.index.min())
    end_date = st.date_input('**Select the end date**:', value=df2.index.max())
    
    
    st.markdown('<h1 class="font"> <strong> Forecast Input </strong></h1>', unsafe_allow_html=True)
    
    nb_days = st.slider('**Days of forecast**', 7, 366, 50)
    model_select = st.selectbox("**Select Model for forcasting**", ["Prophet", "Nixtla", "Neural Forecast"])
    
    if model_select == "Nixtla": 
        display_model = st.selectbox("**Choose the model to display**", ["XGBRegressor", "LGBMRegressor", "RandomForestRegressor"])
    elif model_select =="Neural Forecast":
        
        display_nl = st.selectbox("**Choose the model to display**", ["NBEATS", "NHITS"])
    
    run_forecast = st.button('**Forecast**')
    
      
# create 5 columns
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
# fill in those 5 columns with respective metrics or KPIs
kpi1.metric( 
    label="**Home Airport:**", 
    value= home_airport)

kpi2.metric( 
    label="**Paired Airport:**", 
    value= paired_airport)

kpi3.metric( 
    label="**Days of forecast:**", 
    value= nb_days)

kpi4.metric( 
    label="**Start date**", 
    value= str(start_date))

kpi5.metric( 
    label="**End date**", 
    value= str(end_date))


#Define a color of slidebar
ColorMinMax = st.markdown(''' <style> div.stSlider > div[data-baseweb = "slider"] > div[data-testid="stTickBar"] > div {
    background: rgb(1 1 1 / 0%); } </style>''', unsafe_allow_html = True)


Slider_Cursor = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div[role="slider"]{
    background-color: rgb(14, 38, 74); box-shadow: rgb(14 38 74 / 20%) 0px 0px 0px 0.2rem;} </style>''', unsafe_allow_html = True)

    
Slider_Number = st.markdown(''' <style> div.stSlider > div[data-baseweb="slider"] > div > div > div > div
                                { color: rgb(14, 38, 74); } </style>''', unsafe_allow_html = True)
    

col = f''' <style> div.stSlider > div[data-baseweb = "slider"] > div > div {{
    background: linear-gradient(to right, rgb(1, 183, 158) 0%, 
                                rgb(1, 183, 158) {nb_days}%, 
                                rgba(151, 166, 195, 0.25) {nb_days}%, 
                                rgba(151, 166, 195, 0.25) 100%); }} </style>'''

ColorSlider = st.markdown(col, unsafe_allow_html = True)

#%%
# Affichage de la table
#st.dataframe(data=df, width=1000, height=700)

#filter data
filtered_data = df[(df['home_airport'] == home_airport) & (df['paired_airport'] == paired_airport)]
filtered_data.reset_index(drop=True, inplace=True)

#Date selection
if start_date < end_date:
    pass
else:
    st.error('Error : Date de fin doit être choisi après la dete de début.')


#greater than the start date and smaller than the end date
mask = (filtered_data['date'] >= start_date) & (filtered_data['date'] <= end_date)
filtered_data = filtered_data.loc[mask]
# And display the result!
st.dataframe(filtered_data)
#%%
#Data Visusalisation

st.markdown("## Passenger trafic for the home airport {} and paired airport {}".format(home_airport, paired_airport))

if len(filtered_data) == 0: 
    st.write("**No data available for this selection. Try another selection**")
else: 
    air_flux(home_airport, paired_airport, input_data = filtered_data)
    
        
#%%
#Define a design of kpi
st.markdown("""
<style>
div[data-testid="metric-container"] {
   background-color: rgba(28, 131, 225, 0.1);
   border: 1px solid rgba(28, 131, 225, 0.1);
   padding: 5% 5% 5% 10%;
   border-radius: 25px;
   color: rgb(30, 103, 119);
   overflow-wrap: break-word;
}

/* breakline for metric text         */
div[data-testid="metric-container"] > label[data-testid="stMetricLabel"] > div {
   overflow-wrap: break-word;
   white-space: break-spaces;
   color: blue;
}
</style>
"""
, unsafe_allow_html=True)


st.markdown("## Prediction of the number of passengers on {} days and along the route {} -  {}".format(nb_days, home_airport, paired_airport))


data = generate_route_df(filtered_data, home_airport, paired_airport).rename(columns={'date': 'ds', 'pax_total': 'y'})

if run_forecast :
    
    if len(data) == 0:
        
        st.write("**No data available for this selection. Try another selection**")
        
    else :
        
        if model_select == "Prophet":
            
            model = Prophet()
            model.fit(data) 
            
            # Prepare to predict 
            future_df = model.make_future_dataframe(periods=nb_days) 
            forecast = model.predict(future_df)
            df_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            
            
            
            #Performance of Model 
            eval_df = cross_validation(model, initial='366 days', period='{} days'.format(nb_days), horizon='{} days'.format(nb_days))
            perfor = performance_metrics(eval_df)
            
            # create three columns
            kpi6, kpi7, kpi8, kpi9,kpi10 = st.columns(5)
            # fill in those three columns with respective metrics or KPIs
            kpi6.metric(
                label="**MSE**",
                value=round(np.mean(perfor["mse"]),2))
            
            kpi7.metric(
                label="**RMSE**",
                value=round(np.mean(perfor["rmse"]),2))
            
            kpi8.metric(
                label="**MAE**",
                value=round(np.mean(perfor["mae"]), 2))
            
            kpi9.metric(
                label="**MDAPE**",
                value=round(np.mean(perfor["mdape"]),2))
            
            kpi10.metric(
                label="**SMAPE**",
                value=round(np.mean(perfor["smape"]),2))
            
            
            
            #Plot the prediction
            st.plotly_chart(forecast_plot(df_forecast, data))
            
        elif model_select == "Nixtla":
            
            #List of model
            tested_models = [
                lgb.LGBMRegressor(),
                xgb.XGBRegressor(),
                RandomForestRegressor(random_state=0),
            ]
            
            @njit
            def rolling_mean_28(x):
                return rolling_mean(x, window_size=28)
            
            
            fcst = MLForecast(
                models=tested_models,
                freq='D',
                lags=[7, 14, 21, 28],
                lag_transforms={
                    1: [expanding_mean],
                    7: [rolling_mean_28]
                },
                date_features=['dayofweek'],
                differences=[1])
            
            #Prediction
            filtered_data["date"] = pd.to_datetime(filtered_data["date"])
            nixtla_model = fcst.fit(generate_route_df(filtered_data, home_airport, paired_airport).drop(columns=['paired_airport']),
                                    id_col='home_airport', time_col='date', target_col='pax_total')
            predict_df = nixtla_model.predict(nb_days)
            
            #Performance of model
            crossvalidation_df = fcst.cross_validation(
                data=generate_route_df(filtered_data, home_airport, paired_airport).drop(columns=['paired_airport']),
                window_size=90,
                n_windows=5,
                id_col='home_airport',
                time_col='date',
                target_col='pax_total'
            )
            
            def mse(y, y_hat): 
                delta_y = np.square(y - y_hat) 
                return np.nanmean(delta_y)

            def rmse(y, y_hat): 
                delta_y = np.square(y - y_hat) 
                return np.sqrt(mse(y, y_hat))
          
            def mae(y, y_hat): 
                delta_y = np.abs(y - y_hat)
                
                return np.nanmean(delta_y)
          
          
            cv_mse = crossvalidation_df.groupby(['home_airport', 'cutoff']).apply(lambda df: mse(df['pax_total'], df[display_model])).mean()
          
            cv_rmse = crossvalidation_df.groupby(['home_airport', 'cutoff']).apply(lambda df: rmse(df['pax_total'], df[display_model])).mean()
            
            cv_mae = crossvalidation_df.groupby(['home_airport', 'cutoff']).apply(lambda df: mae(df['pax_total'], df[display_model])).mean()
            
            # create three columns
            kpi11, kpi12, kpi13 = st.columns(3)
            # fill in those three columns with respective metrics or KPIs
            kpi11.metric(
                label="**MSE**",
                value=round(cv_mse,2))
            
            kpi12.metric(
                label="**RMSE**",
                value=round(cv_rmse ,2))
            
            kpi13.metric(
                label="**MAE**",
                value=round(cv_mae, 2))
            
            
            #Display the forecast graph
            draw_ts_multiple((pd.concat([generate_route_df(filtered_data, home_airport, paired_airport).drop(columns=['paired_airport']),
                         predict_df])), v1='pax_total', v2= display_model)
            
            
        elif model_select == "Neural Forecast":
            
            horizon = nb_days
            models = [NBEATS(input_size=2 * horizon, h=horizon, max_epochs=50),
            NHITS(input_size=2 * horizon, h=horizon, max_epochs=50)]
            
            nforecast = NeuralForecast(models=models, freq='D')
            nforecast.fit(df=generate_route_df(filtered_data, home_airport, paired_airport).drop(columns=['paired_airport']).rename(columns={'home_airport': 'unique_id',
                                                                                                      'date': 'ds',
                                                                                                      'pax_total': 'y'}))
            
            
            #Plot the graph
            draw_ts_multiple((pd.concat([generate_route_df(filtered_data, home_airport, paired_airport).drop(columns=['paired_airport']).rename(columns={'home_airport': 'unique_id',
                                                                                                      'date': 'ds',
                                                                                                      'pax_total': 'y'}),
                         nforecast.predict().reset_index()]).rename(columns = {'unique_id': 'home_airport', 'ds': 'date', 'y': 'pax_total' })),
             v1='pax_total', v2= display_nl)
            
            
    
                
            
            
            
            
            
    
    
    
    
    


    
