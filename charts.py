"""Plotly chart builders used by the Dash application."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

if __package__:
    from .data_utils import DAYS, SEASONS
else:
    from data_utils import DAYS, SEASONS

COLORS = ['#227c9d', '#dc724c', '#648b46', '#8c6caf']
WEATHER_LABELS = dict(temperature_c='Temperature (°C)', precipitation_mm='Precipitation (mm/day)', wind_speed_ms='Wind speed (m/s)')
MODEL_LABELS = dict(calendar_only='Calendar only', calendar_weather='Calendar + weather', previous_week='Previous week')


def style(fig, title, y=None):
    fig.update_layout(template='plotly_white', title=dict(text=title, font=dict(size=18)),
                      font=dict(family='Arial, sans-serif', color='#233746'),
                      margin=dict(l=55,r=25,t=65,b=55), height=390,
                      paper_bgcolor='white', plot_bgcolor='white', legend_title_text='',
                      colorway=COLORS, hovermode='closest')
    if y:
        fig.update_yaxes(title=y)
    return fig


def empty(title, message='No days match these filters. Adjust the date range or selection.'):
    fig = style(go.Figure(), title)
    fig.add_annotation(text=message, x=.5, y=.5, xref='paper', yref='paper', showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False); fig.update_yaxes(visible=False)
    return fig


def demand_trend(data):
    if data.empty:
        return empty('Daily demand')
    fig = go.Figure(go.Scatter(x=data.date, y=data.daily_hires, mode='markers',
                              name='Daily hires', marker=dict(size=4,color=COLORS[0],opacity=.6)))
    # Break the moving average at every omitted calendar date. Never smooth
    # across winter/weekday gaps introduced by the controls.
    series = data.set_index('date').daily_hires.reindex(pd.date_range(data.date.min(), data.date.max()))
    smoothed = series.rolling(7,min_periods=7).mean()
    fig.add_trace(go.Scatter(x=smoothed.index,y=smoothed,mode='lines',name='7 consecutive day mean',
                            line=dict(color=COLORS[1],width=2),connectgaps=False))
    fig.update_xaxes(title='Date')
    return style(fig, f'Daily hires · {len(data):,} selected days', 'Bicycle hires per day')


def calendar_heatmap(data):
    if data.empty:
        return empty('Calendar patterns')
    mean = data.pivot_table(index='month',columns='day_of_week',values='daily_hires',aggfunc='mean').reindex(index=range(1,13),columns=range(7))
    count = data.pivot_table(index='month',columns='day_of_week',values='daily_hires',aggfunc='size').reindex(index=range(1,13),columns=range(7)).fillna(0)
    fig = go.Figure(go.Heatmap(z=mean.to_numpy(),x=DAYS,y=list(range(1,13)),customdata=count.to_numpy(),
                               colorscale='Teal',colorbar=dict(title='Mean hires'),hoverongaps=False,
                               hovertemplate='%{x} · month %{y}<br>Mean %{z:,.0f}<br>Days %{customdata:.0f}<extra></extra>'))
    fig.update_yaxes(title='Month',dtick=1,autorange='reversed')
    return style(fig,'Mean daily hires by month and weekday')


def season_comparison(data):
    if data.empty:
        return empty('Season and day type')
    grouped = data.groupby(['season','day_type'],as_index=False).daily_hires.agg(mean_hires='mean',days='size')
    fig = px.bar(grouped,x='season',y='mean_hires',color='day_type',barmode='group',text='days',
                 category_orders={'season':SEASONS,'day_type':['Weekday','Weekend']},color_discrete_sequence=COLORS,
                 labels={'season':'Season','mean_hires':'Mean daily hires','day_type':'Day type','days':'Days'})
    fig.update_traces(texttemplate='n=%{text}',textposition='outside',cliponaxis=False)
    return style(fig,'Season and day type · labels show observation counts','Mean daily bicycle hires')


def weather_scatter(data, variable='temperature_c', grouping='day_type'):
    if data.empty:
        return empty('Weather and demand')
    grouping = grouping if grouping in ['day_type','season'] else 'day_type'
    variable = variable if variable in WEATHER_LABELS else 'temperature_c'
    order = SEASONS if grouping=='season' else ['Weekday','Weekend']
    fig = px.scatter(data,x=variable,y='daily_hires',color=grouping,hover_data=['date'],opacity=.6,
                     category_orders={grouping:order},color_discrete_sequence=COLORS,
                     labels={variable:WEATHER_LABELS[variable],'daily_hires':'Daily bicycle hires',grouping:'Group'})
    for i,label in enumerate(order):
        frame = data.loc[data[grouping].eq(label)]
        if len(frame) >= 3 and frame[variable].nunique() >= 2:
            coeff = np.polyfit(frame[variable],frame.daily_hires,1)
            x = np.array([frame[variable].min(),frame[variable].max()])
            fig.add_trace(go.Scatter(x=x,y=np.polyval(coeff,x),mode='lines',line=dict(color=COLORS[i],width=2),
                                     name=f'{label} trend (n={len(frame)})',hoverinfo='skip'))
    return style(fig,'Weather associations · lines are descriptive fits','Daily bicycle hires')


def rain_comparison(data, threshold=1.):
    if data.empty:
        return empty('Rainfall groups')
    frame = data.assign(rain=np.where(data.precipitation_mm>=threshold,f'Rain ≥{threshold:g} mm',f'Below {threshold:g} mm'))
    grouped=frame.groupby(['rain','day_type'],as_index=False).daily_hires.agg(mean_hires='mean',days='size')
    fig=px.bar(grouped,x='rain',y='mean_hires',color='day_type',text='days',barmode='group',color_discrete_sequence=COLORS,
               labels={'rain':'Daily rainfall group','day_type':'Day type','mean_hires':'Mean daily hires'})
    fig.update_traces(texttemplate='n=%{text}',textposition='outside',cliponaxis=False)
    return style(fig,'Rainfall and demand · descriptive group means','Mean daily bicycle hires')


def performance_chart(metrics):
    frame = metrics.copy(); frame['name']=frame.model.map(MODEL_LABELS)
    fig=px.bar(frame,x='name',y='mae',color='name',text_auto=',.0f',color_discrete_sequence=COLORS,
               labels={'name':'Model','mae':'MAE in daily hires'})
    fig.update_layout(showlegend=False)
    return style(fig,'Fixed holdout · all 366 days of 2016','Mean absolute error in hires')


def predictions_chart(predictions):
    if predictions.empty:
        return empty('Observed and predicted hires','No held-out 2016 days match the selection. Select dates in 2016.')
    fig=go.Figure()
    for name,color in [('daily_hires','#233746'),('calendar_only','#648b46'),('calendar_weather','#227c9d')]:
        fig.add_trace(go.Scatter(x=predictions.date,y=predictions[name],mode='markers',marker=dict(size=4,color=color,opacity=.65),
                                 name='Observed' if name=='daily_hires' else MODEL_LABELS[name]))
    fig.update_xaxes(title='Held-out date in 2016')
    return style(fig,f'Observed and predicted · {len(predictions)} held-out days','Daily bicycle hires')


def residual_chart(predictions):
    if predictions.empty:
        return empty('Prediction errors','No held-out 2016 days match the selection.')
    frame=predictions.assign(error=predictions.daily_hires-predictions.calendar_weather)
    fig=px.scatter(frame,x='calendar_weather',y='error',color='day_type',hover_data=['date'],opacity=.6,
                   color_discrete_sequence=COLORS,labels={'calendar_weather':'Calendar + weather prediction','error':'Observed minus predicted hires'})
    fig.add_hline(y=0,line_dash='dash',line_color='#82909b')
    return style(fig,'Held-out errors · calendar plus weather','Observed minus predicted hires')
