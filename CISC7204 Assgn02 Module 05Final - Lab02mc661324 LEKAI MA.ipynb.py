# Import required libraries
import pandas as pd
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.express as px

# 1. 直接读取你刚刚下载到本地的 CSV 文件（避免网络 404 问题）
data = pd.read_csv('historical_automobile_sales.csv')

# Create a dash application
app = dash.Dash(__name__)

# Get the list of years from the dataset
year_list = [i for i in range(1980, 2024, 1)]

# TASK 2.1: Title
app.layout = html.Div(children=[
    html.H1("Automobile Sales Statistics Dashboard",
            style={'textAlign': 'center', 'color': '#503D36', 'font-size': 24}),

    html.Div([
        # TASK 2.2: Report Type Dropdown
        dcc.Dropdown(id='dropdown-statistics',
                     options=[
                         {'label': 'Yearly Statistics', 'value': 'Yearly Statistics'},
                         {'label': 'Recession Period Statistics', 'value': 'Recession Period Statistics'}
                     ],
                     placeholder='Select a report type',
                     value='Select Statistics',
                     style={'width': '80%', 'padding': '3px', 'font-size': '20px', 'text-align-last': 'center'}),

        # TASK 2.2: Year Dropdown
        dcc.Dropdown(id='select-year',
                     options=[{'label': i, 'value': i} for i in year_list],
                     placeholder='Select-year',
                     value='Select-year',
                     style={'width': '80%', 'padding': '3px', 'font-size': '20px', 'text-align-last': 'center'})
    ]),

    # TASK 2.3: Output Container
    html.Div([
        html.Div(id='output-container', className='chart-grid', style={'display': 'flex', 'flex-direction': 'column'}),
    ])
])


# TASK 2.4: Enable/disable year dropdown
@app.callback(
    Output(component_id='select-year', component_property='disabled'),
    Input(component_id='dropdown-statistics', component_property='value'))
def update_input_container(selected_statistics):
    if selected_statistics == 'Yearly Statistics':
        return False
    else:
        return True


# TASK 2.4, 2.5, 2.6: Update graphs
@app.callback(
    Output(component_id='output-container', component_property='children'),
    [Input(component_id='dropdown-statistics', component_property='value'),
     Input(component_id='select-year', component_property='value')])
def update_output_container(selected_statistics, input_year):
    # TASK 2.5: Recession Period Statistics
    if selected_statistics == 'Recession Period Statistics':
        recession_data = data[data['Recession'] == 1]

        yearly_rec = recession_data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        R_chart1 = dcc.Graph(
            figure=px.line(yearly_rec, x='Year', y='Automobile_Sales',
                           title="Average Automobile Sales Fluctuation over Recession Period"))

        average_sales = recession_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        R_chart2 = dcc.Graph(
            figure=px.bar(average_sales, x='Vehicle_Type', y='Automobile_Sales',
                          title="Average Number of Vehicles Sold by Vehicle Type during Recession"))

        exp_rec = recession_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        R_chart3 = dcc.Graph(
            figure=px.pie(exp_rec, values='Advertising_Expenditure', names='Vehicle_Type',
                          title="Total Expenditure Share by Vehicle Type During Recessions"))

        unemp_data = recession_data.groupby(['unemployment_rate', 'Vehicle_Type'])[
            'Automobile_Sales'].mean().reset_index()
        R_chart4 = dcc.Graph(
            figure=px.bar(unemp_data, x='unemployment_rate', y='Automobile_Sales', color='Vehicle_Type',
                          labels={'unemployment_rate': 'Unemployment Rate',
                                  'Automobile_Sales': 'Average Automobile Sales'},
                          title='Effect of Unemployment Rate on Vehicle Type and Sales'))

        return [
            html.Div(className='chart-item', children=[html.Div(children=R_chart1), html.Div(children=R_chart2)],
                     style={'display': 'flex'}),
            html.Div(className='chart-item', children=[html.Div(children=R_chart3), html.Div(children=R_chart4)],
                     style={'display': 'flex'})
        ]

    # TASK 2.6: Yearly Statistics
    elif (input_year and selected_statistics == 'Yearly Statistics'):
        try:
            input_year = int(input_year)
        except ValueError:
            return None

        yearly_data = data[data['Year'] == input_year]

        yas = data.groupby('Year')['Automobile_Sales'].mean().reset_index()
        Y_chart1 = dcc.Graph(
            figure=px.line(yas, x='Year', y='Automobile_Sales',
                           title='Yearly Automobile Sales for the Whole Period'))

        mas = yearly_data.groupby('Month')['Automobile_Sales'].sum().reset_index()
        Y_chart2 = dcc.Graph(
            figure=px.line(mas, x='Month', y='Automobile_Sales',
                           title='Total Monthly Automobile Sales in {}'.format(input_year)))

        avr_vdata = yearly_data.groupby('Vehicle_Type')['Automobile_Sales'].mean().reset_index()
        Y_chart3 = dcc.Graph(
            figure=px.bar(avr_vdata, x='Vehicle_Type', y='Automobile_Sales',
                          title='Average Vehicles Sold by Vehicle Type in the year {}'.format(input_year)))

        exp_data = yearly_data.groupby('Vehicle_Type')['Advertising_Expenditure'].sum().reset_index()
        Y_chart4 = dcc.Graph(
            figure=px.pie(exp_data, values='Advertising_Expenditure', names='Vehicle_Type',
                          title='Total Advertisement Expenditure for Each Vehicle in {}'.format(input_year)))

        return [
            html.Div(className='chart-item', children=[html.Div(children=Y_chart1), html.Div(children=Y_chart2)],
                     style={'display': 'flex'}),
            html.Div(className='chart-item', children=[html.Div(children=Y_chart3), html.Div(children=Y_chart4)],
                     style={'display': 'flex'})
        ]

    else:
        return None


# Run the app
if __name__ == '__main__':
    app.run(debug=True)