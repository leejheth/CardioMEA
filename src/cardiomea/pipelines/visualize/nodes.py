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
import itertools
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def dashboard(cardio_db,port):
    cell_lines = cardio_db["cell_line"].unique()
    col_simple = ['cell_line','compound','file_path_full','time','note']
    
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
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
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

    
    # R amplitude plot
    @app.callback(
        Output('r_amp_graph', 'figure'),
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def r_amp_plot(selected_rows, data):
        if not selected_rows:
            return px.strip()
        selected_data = [data[i] for i in selected_rows]
        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        # filter only rows that are selected
        df_selected = cardio_db[cardio_db['file_path_full'].isin(data_df['file_path_full'])].reset_index(drop=True)
        # convert string of r_amplitudes to list of int
        r_amp_list = list(df_selected['r_amplitudes_str'].apply(lambda x: list(map(int, x.split(' ')))))

        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(r_amp_list[i])
        r_amp_flattened = list(itertools.chain(*r_amp_list))

        return px.strip(x=xlabels, y=r_amp_flattened)

    # R width plot
    @app.callback(
        Output('r_width_graph', 'figure'),
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def r_amp_plot(selected_rows, data):
        if not selected_rows:
            return px.strip()
        selected_data = [data[i] for i in selected_rows]
        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        # filter only rows that are selected
        df_selected = cardio_db[cardio_db['file_path_full'].isin(data_df['file_path_full'])].reset_index(drop=True)
        # convert string of r_amplitudes to list of int
        r_width_list = list(df_selected['r_widths_str'].apply(lambda x: list(map(float, x.split(' ')))))

        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(r_width_list[i])
        r_width_flattened = list(itertools.chain(*r_width_list))

        return px.strip(x=xlabels, y=r_width_flattened)
    
    # FPD plot
    @app.callback(
        Output('fpd_graph', 'figure'),
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def fpd_plot(selected_rows, data):
        if not selected_rows:
            return px.strip()
        selected_data = [data[i] for i in selected_rows]
        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        # filter only rows that are selected
        df_selected = cardio_db[cardio_db['file_path_full'].isin(data_df['file_path_full'])].reset_index(drop=True)
        # convert string of r_amplitudes to list of int
        fpd_list = list(df_selected['fpds_str'].apply(lambda x: list(map(float, x.split(' ')))))

        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(fpd_list[i])
        fpd_flattened = list(itertools.chain(*fpd_list))

        return px.strip(x=xlabels, y=fpd_flattened)
    
    # conduction speed plot
    @app.callback(
        Output('conduction_graph', 'figure'),
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def conduction_plot(selected_rows, data):
        if not selected_rows:
            return px.strip()
        selected_data = [data[i] for i in selected_rows]
        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        # filter only rows that are selected
        df_selected = cardio_db[cardio_db['file_path_full'].isin(data_df['file_path_full'])].reset_index(drop=True)
        # convert string of r_amplitudes to list of int
        speed_list = list(df_selected['conduction_speed_str'].apply(lambda x: list(map(float, x.split(' ')))))

        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(speed_list[i])
        speed_flattened = list(itertools.chain(*speed_list))

        return px.strip(x=xlabels, y=speed_flattened)

    app.layout = html.Div([
        html.H1(children='CardioMEA Dashboard', style={'textAlign':'center'}),
        dbc.Row([          
            html.Div([
                # check list to choose cell lines    
                dbc.Form([checklist]),
                html.H4("List of files"),
                # table to show shorlisted files
                table,
                html.Div(id='selected_data',children=[]),      
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("R amplitudes", style={'textAlign':'center'}),
                    dcc.Graph(id='r_amp_graph'),
                ])  
            ], width=5),
            dbc.Col([
                html.Div([
                    html.H4("R widths", style={'textAlign':'center'}),
                    dcc.Graph(id='r_width_graph'),
                ])  
            ], width=5),
        ]),
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H4("FPDs", style={'textAlign':'center'}),
                    dcc.Graph(id='fpd_graph'),
                ])  
            ], width=5),
            dbc.Col([
                html.Div([
                    html.H4("Conduction speed", style={'textAlign':'center'}),
                    dcc.Graph(id='conduction_graph'),
                ])  
            ], width=5),
        ])
    ])

    # open the URL with the default web browser of the user’s computer
    print("Ctrl + C to exit.")
    # webbrowser.open_new(f"http://127.0.0.1:{port}/")
    
    app.run(debug=True, port=port)

    
