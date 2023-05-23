"""
This is a boilerplate pipeline 'visualize'
generated using Kedro 0.18.7
"""
from dash import Dash, dcc, dash_table, html, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import webbrowser
import numpy as np
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def dashboard(cardio_db,port):
    cell_lines = cardio_db["cell_line"].unique()
    col_simple = ['cell_line','compound','note','file_path_full','time']

    # df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')

    # app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    # app.layout = html.Div([
    #     html.H1(children='CardioMEA Dashboard', style={'textAlign':'center'}),
    #     dcc.Dropdown(df.country.unique(), 'Canada', id='dropdown-selection'),
    #     dcc.Graph(id='graph-content')
    # ])

    # @callback(
    #     Output('graph-content', 'figure'),
    #     Input('dropdown-selection', 'value')
    # )
    # def update_graph(value):
    #     dff = df[df.country==value]
    #     return px.line(dff, x='year', y='pop')
    
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    checklist = html.Div(
        [
            dbc.Label("Choose cell lines"),
            dbc.Checklist(
                options=[{"label": c, "value": c} for c in cell_lines],
                value=[],
                id="checklist",
                inline=True,
            ),
        ]
    )

    columns = [{'name': i, 'id': i} for i in col_simple]

    # Define data table with checkbox column
    table = dash_table.DataTable(
        id='datatable',
        columns=columns,
        data=[],
        editable=False,
        row_selectable="multi",
        selected_rows=[],
        style_table={'overflowX': 'scroll'},
        style_cell={'textAlign': 'left', 'minWidth': '150px'},
    )

    @app.callback(
        Output("datatable", "data"),
        Input("checklist", "value")
    )
    def update_table(selected_values):
        # create a list of options for the dropdown based on selected values
        df = cardio_db.loc[cardio_db["cell_line"].isin(selected_values), col_simple]
        table_data = [{c['id']: df[c['name']].iloc[i] for c in columns} for i in range(len(df))]

        return table_data

    # Define callback to update selected rows
    @app.callback(
        Output('selected_data', 'children'),
        [Input('datatable', 'selected_rows')],
        [State('datatable', 'data')]
    )
    def update_selected_rows(selected_rows, data):
        if not selected_rows:
            return ''
        selected_data = [data[i] for i in selected_rows]
        return html.Div([
            html.Div('Selected Rows:'),
            html.Ul([
                html.Li(f"{row['cell_line']} ({row['time']})")
                for row in selected_data
            ])
        ])

    example_df = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/gapminder_unfiltered.csv')
    @app.callback(
        Output('graph-content', 'figure'),
        Input('dropdown-selection', 'value')
    )
    def update_graph(value):
        dff = example_df[example_df.country==value]
        return px.line(dff, x='year', y='pop')

    app.layout = html.Div([
        html.H1(children='CardioMEA Dashboard', style={'textAlign':'center'}),
        dbc.Row([          
            html.Div([
                # check list to choose cell lines    
                dbc.Form([checklist]),
                html.H4("List of files"),
                # table to show shorlisted files
                table,
                html.Div(id='selected_data'),      
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    dcc.Dropdown(example_df.country.unique(), 'Canada', id='dropdown-selection'),
                    dcc.Graph(id='graph-content'),
                ])  
            ], width=4),
        ])
    ])

    # open the URL with the default web browser of the user’s computer
    print("Ctrl + C to exit.")
    webbrowser.open_new(f"http://127.0.0.1:{port}/")
    
    app.run(debug=False, port=port)

    
