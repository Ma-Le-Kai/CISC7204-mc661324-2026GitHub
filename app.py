"""Standalone local Dash application using the bundled data snapshot."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from dash import Dash, Input, Output, State, dcc, html

if __package__:
    from .data_utils import ROOT, SEASONS, WEATHER, filter_data, validate_daily
    from .charts import (WEATHER_LABELS, MODEL_LABELS, demand_trend, calendar_heatmap,
        season_comparison, weather_scatter, rain_comparison, performance_chart, predictions_chart, residual_chart)
else:
    from data_utils import ROOT, SEASONS, WEATHER, filter_data, validate_daily
    from charts import (WEATHER_LABELS, MODEL_LABELS, demand_trend, calendar_heatmap,
        season_comparison, weather_scatter, rain_comparison, performance_chart, predictions_chart, residual_chart)


def table(frame, columns):
    return html.Div(html.Table([
        html.Thead(html.Tr([html.Th(label) for _,label in columns])),
        html.Tbody([html.Tr([html.Td(f'{row[key]:,.0f}' if isinstance(row[key],(int,float,np.number)) else row[key])
                             for key,_ in columns]) for _,row in frame.iterrows()])
    ]),className='table-scroll')


def read_results(root=ROOT):
    root=Path(root)
    required=['data/london_daily.csv','data/test_predictions.csv','data/model_metrics.csv','data/summary.json']
    if any(not (root/name).exists() for name in required):
        raise FileNotFoundError('Bundled data snapshot is incomplete. Restore the four files in this dashboard folder\'s data directory.')
    data=pd.read_csv(root/required[0],parse_dates=['date'])
    validate_daily(data,'Dashboard input',['daily_hires']+WEATHER)
    return (data,pd.read_csv(root/required[1],parse_dates=['date']),pd.read_csv(root/required[2]),
            json.loads((root/required[3]).read_text(encoding='utf8')))


def create_app(root=ROOT):
    data,predictions,metrics,summary=read_results(root)
    app=Dash(__name__,title='London Bike Sharing | Weather and Calendar',assets_folder=str(Path(__file__).resolve().parent/'assets'))
    graph_config=dict(displaylogo=False,toImageButtonOptions=dict(format='png',scale=2))
    def graph(identifier):
        return dcc.Graph(id=identifier,config=graph_config)
    def panel(identifier,title,description,content):
        return html.Section([html.Div([html.H2(title),html.P(description)],className='section-intro'),content],id=identifier)
    display_metrics=metrics.assign(model=metrics.model.map(MODEL_LABELS))
    change=summary['weather_mae_improvement_pct']
    app.layout=html.Div([
        html.Header([
            html.Div([html.Span('CISC7204 / PAIR 01',className='eyebrow'),html.Span('2015—2016 · LONDON',className='period')],className='topline'),
            html.H1(['A city on two wheels',html.Span('Weather, calendars and daily demand')]),
            html.P('Explore 731 days of Santander Cycle hires. Start with the calendar, examine weather relationships, then test what generalises to a later year.'),
            html.Div([html.Span('731 validated daily records'),html.Span('TfL + NASA POWER'),html.Span('Aggregate open data')],className='badges')
        ],className='hero'),
        html.Main([
            html.Section([
                html.Div([html.Label('Date range'),dcc.DatePickerRange(id='dates',min_date_allowed='2015-01-01',max_date_allowed='2016-12-31',
                         start_date='2015-01-01',end_date='2016-12-31',display_format='DD MMM YYYY')],className='control date-control'),
                html.Div([html.Label('Day type'),dcc.Dropdown(id='day-type',options=['All','Weekday','Weekend'],value='All',clearable=False)],className='control'),
                html.Div([html.Label('Seasons'),dcc.Dropdown(id='seasons',options=SEASONS,value=SEASONS,multi=True,placeholder='Select at least one season')],className='control seasons-control'),
                html.Button('Download selected CSV',id='download-button',className='download-button'),dcc.Download(id='download-data'),
            ],className='filter-bar',**{'aria-label':'Dashboard filters'}),
            html.Div(id='selection-label',className='selection-label',role='status',**{'aria-live':'polite'}),
            html.Div(id='kpis',className='kpi-grid'),
            dcc.Tabs(id='view',value='overview',children=[dcc.Tab(label='01  Demand overview',value='overview'),
                     dcc.Tab(label='02  Weather relationships',value='weather'),dcc.Tab(label='03  Held-out evaluation',value='models')]),
            panel('overview-panel','When does London ride?',
                  'Compare daily demand with recurring weekday and seasonal patterns. Counts refer to days, not individual trips.',
                  html.Div([html.Div(graph('trend'),className='chart wide'),html.Div(graph('calendar'),className='chart'),
                            html.Div(graph('season-chart'),className='chart')],className='chart-grid')),
            panel('weather-panel','How does weather relate to demand?',
                  'Use the same date and calendar filters to compare like groups. Lines and group means describe associations; they do not identify causal effects.',
                  html.Div([
                      html.Div([html.Div([html.Label('Weather measure'),dcc.Dropdown(id='weather-variable',options=[{'label':v,'value':k} for k,v in WEATHER_LABELS.items()],value='temperature_c',clearable=False)],className='control'),
                                html.Div([html.Label('Compare groups'),dcc.Dropdown(id='weather-group',options=[{'label':'Weekday and weekend','value':'day_type'},{'label':'Seasons','value':'season'}],value='day_type',clearable=False)],className='control'),
                                html.Div([html.Label('Rainfall threshold'),dcc.Dropdown(id='rain-threshold',options=[{'label':f'{v:g} mm/day','value':v} for v in [.5,1,2]],value=1,clearable=False)],className='control')],className='weather-controls'),
                      html.Div([html.Div(graph('weather-scatter'),className='chart'),html.Div(graph('rain-chart'),className='chart')],className='chart-grid'),
                      html.P(id='weather-story',className='story',role='status')
                  ])),
            panel('models-panel','Does weather add information beyond the calendar?',
                  'Fit and tune on 2015; evaluate on 2016. Test weather is realised weather, so this is a retrospective benchmark rather than an operational forecast.',
                  html.Div([
                      html.P(f"Across the fixed 2016 test set, adding weather {'reduces' if change>=0 else 'increases'} MAE by {abs(change):.1f}% against the calendar-only model.",className='story'),
                      html.P('The score table and MAE bars below always use all 366 test days. The observed-versus-predicted and residual charts follow your filters within 2016.',className='scope-note'),
                      table(display_metrics,[('model','Model'),('days','Test days'),('mae','MAE in hires'),('rmse','RMSE in hires')]),
                      html.Div([html.Div(dcc.Graph(figure=performance_chart(metrics),config=graph_config),className='chart wide'),
                                html.Div(graph('predictions-chart'),className='chart'),html.Div(graph('residuals-chart'),className='chart')],className='chart-grid'),
                      html.P('The previous-week benchmark uses the observed count seven days earlier for each day. Earlier 2016 observations are available in this rolling one-day comparison. It is not a forecast of the whole year made on 1 January.',className='scope-note')
                  ])),
            html.Footer([
                html.Strong('Read the evidence within its limits.'),
                html.P('Two historical years, citywide hire totals and one NASA weather grid point. Holidays, events and scheme changes are not fully captured. Calendar controls and statistical intervals do not establish causation.'),
                html.Div([html.A('TfL daily hire data',href='https://data.london.gov.uk/dataset/number-of-bicycle-hires-2r84d',target='_blank',rel='noreferrer'),
                          html.A('NASA POWER documentation',href='https://power.larc.nasa.gov/docs/services/api/temporal/daily/',target='_blank',rel='noreferrer')],className='source-links'),
                html.Small('CISC7204 Assignment 02')
            ])
        ])
    ],className='app')

    @app.callback(Output('overview-panel','style'),Output('weather-panel','style'),Output('models-panel','style'),Input('view','value'))
    def change_view(view):
        return tuple({'display':'block' if view==name else 'none'} for name in ['overview','weather','models'])

    outputs=[Output(name,'figure') for name in ['trend','calendar','season-chart','weather-scatter','rain-chart','predictions-chart','residuals-chart']]
    @app.callback(*outputs,Output('selection-label','children'),Output('kpis','children'),Output('weather-story','children'),
                  Input('dates','start_date'),Input('dates','end_date'),Input('day-type','value'),Input('seasons','value'),
                  Input('weather-variable','value'),Input('weather-group','value'),Input('rain-threshold','value'))
    def update(start,end,day_type,seasons,variable,group,threshold):
        return dashboard_payload(data,predictions,start,end,day_type,seasons,variable,group,threshold)

    @app.callback(Output('download-data','data'),Input('download-button','n_clicks'),
                  State('dates','start_date'),State('dates','end_date'),State('day-type','value'),State('seasons','value'),prevent_initial_call=True)
    def download(clicks,start,end,day_type,seasons):
        frame=filter_data(data,start,end,day_type,seasons or [])
        return dcc.send_data_frame(frame.to_csv,'london_selected_days.csv',index=False,date_format='%Y-%m-%d')

    # Pure analysis callback is also exposed for focused integration tests.
    app.update_dashboard=update
    return app


def dashboard_payload(data,predictions,start,end,day_type,seasons,variable,group,threshold):
    selected=filter_data(data,start,end,day_type,seasons or [])
    test=filter_data(predictions,start,end,day_type,seasons or [])
    threshold=threshold if threshold in [.5,1,2] else 1
    figures=[demand_trend(selected),calendar_heatmap(selected),season_comparison(selected),
             weather_scatter(selected,variable,group),rain_comparison(selected,threshold),predictions_chart(test),residual_chart(test)]
    scope=f'{len(selected):,} / 731 days selected · {start or "start"} to {end or "end"} · {day_type} · {", ".join(seasons or []) or "no seasons"}'
    mean=f'{selected.daily_hires.mean():,.0f}' if len(selected) else '—'
    temp=f'{selected.temperature_c.mean():.1f} °C' if len(selected) else '—'
    rainy=f'{(selected.precipitation_mm>=threshold).mean():.0%}' if len(selected) else '—'
    cards=[html.Div([html.Span(label),html.Strong(value),html.Small(note)],className='kpi') for label,value,note in [
        ('Selected days',f'{len(selected):,}','Unit of analysis: one day'),('Average daily hires',mean,'Citywide Santander Cycle hires'),
        ('Mean temperature',temp,'NASA POWER London grid point'),('Days above rain threshold',rainy,f'At least {threshold:g} mm/day')]]
    if len(selected)<3:
        story='Too few selected days for a useful correlation. Broaden the filters; a fitted relationship is not reliable with this selection.'
    else:
        col=variable if variable in WEATHER else WEATHER[0]
        r=selected[col].corr(selected.daily_hires)
        story=(f'For these {len(selected)} days, the unadjusted Pearson correlation between {WEATHER_LABELS[col].lower()} and hires is {r:.3f}. '
               'Group lines are descriptive; changing the calendar selection may change the relationship.') if np.isfinite(r) else 'The selected weather values do not vary enough to calculate a correlation.'
    return (*figures,scope,cards,story)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8060)
    args=parser.parse_args()
    create_app().run(host='127.0.0.1',port=args.port,debug=False)
