"""
This is a boilerplate pipeline 'visualize'
generated using Kedro 0.18.7
"""
from dash import Dash, dcc, dash_table, html, Output, Input, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import webbrowser
import itertools
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def dashboard(cardio_db,port,base_directory):
    cell_lines = cardio_db["cell_line"].unique()
    cardio_db["file_path"] = cardio_db["file_path_full"].apply(lambda x: x.removeprefix(base_directory))
    col_simple = ['cell_line','compound','file_path','time','note']
    columns = [{'name': i, 'id': i} for i in col_simple]
    
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    checklist = html.Div(
        [
            html.H4("Choose cell lines"),
            dbc.Checklist(
                options=[{"label": c, "value": c} for c in cell_lines],
                value=[],
                id="checklist",
                inline=True,
            ),
        ]
    )

    switch = html.Div(
        [
            dbc.Switch(
                label="Show the latest data only",
                value=True,
                id="latest_only",
            ),
        ]
    )

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

    # Reset selections if table is changed by the user
    @app.callback(
        Output("datatable", "selected_rows"),
        [
            Input("checklist", "value"),
            Input("latest_only", "value"),
        ],
    )
    def reset_selected_rows(checklist, switch):
        return []

    @app.callback(
        Output("datatable", "data"),
        [
            Input("checklist", "value"),
            Input("latest_only", "value"),
        ],
    )
    def update_table(selected_values, switch_value):
        if switch_value:
            df = cardio_db.sort_values('time').groupby('file_path').tail(1).reset_index(drop=True)
        else:
            df = cardio_db.copy()
        # filter only cell lines that are selected 
        df_selected = df.loc[df["cell_line"].isin(selected_values), col_simple]
        table_data = [{c['id']: df_selected[c['name']].iloc[i] for c in columns} for i in range(len(df_selected))]

        return table_data

    # Define callback to display selected rows
    @app.callback(
        Output('selected_data', 'children'),
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def display_selected_rows(selected_rows, data):
        if not selected_rows:
            return ''
        selected_data = [data[i] for i in selected_rows]
        return html.Div([
            html.H4('Selected Files in Order:'),
            html.Ul([
                html.Li(f"file_{cnt+1}: {row['cell_line']} ({row['file_path']})")
                for cnt, row in enumerate(selected_data)
            ])
        ])

    
    # # R amplitude plot
    # @app.callback(
    #     Output('r_amp_graph', 'figure'),
    #     Input('datatable', 'selected_rows'),
    #     State('datatable', 'data')
    # )
    # def r_amp_plot(selected_rows, data):
    #     if not selected_rows:
    #         return px.strip()
    #     selected_data = [data[i] for i in selected_rows]
    #     # convert list of dict to dataframe
    #     data_df = pd.DataFrame(selected_data)
    #     data_df["time"] = pd.to_datetime(data_df["time"])
    #     # filter only rows that are selected and preserve the selection order
    #     df_selected = pd.merge(data_df["time"],cardio_db, how="left", on="time", sort=False)
    #     # convert string of r_amplitudes to list of int
    #     r_amp_list = list(df_selected['r_amplitudes_str'].apply(lambda x: list(map(int, x.split(' ')))))

    #     xlabels=[]
    #     for i in range(len(df_selected)):
    #         xlabels += [f'file_{i+1}']*len(r_amp_list[i])
    #     r_amp_flattened = list(itertools.chain(*r_amp_list))
    #     fig = px.strip(x=xlabels, y=r_amp_flattened, labels={'x':'', 'y':'Micro Volt'}).update_traces(jitter=1,opacity=0.6,marker_size=5)
    #     fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['r_amplitudes_mean'].to_list(),
    #                      error_y_array=df_selected['r_amplitudes_std'].to_list(),
    #                      mode='markers',
    #                      marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=30, line=dict(width=2)),
    #                      showlegend=False))
    #     return fig

    # # R width plot
    # @app.callback(
    #     Output('r_width_graph', 'figure'),
    #     Input('datatable', 'selected_rows'),
    #     State('datatable', 'data')
    # )
    # def r_amp_plot(selected_rows, data):
    #     if not selected_rows:
    #         return px.strip()
    #     selected_data = [data[i] for i in selected_rows]
    #     # convert list of dict to dataframe
    #     data_df = pd.DataFrame(selected_data)
    #     data_df["time"] = pd.to_datetime(data_df["time"])
    #     # filter only rows that are selected and preserve the selection order
    #     df_selected = pd.merge(data_df["time"],cardio_db, how="left", on="time", sort=False)
    #     # convert string of r_amplitudes to list of int
    #     r_width_list = list(df_selected['r_widths_str'].apply(lambda x: list(map(float, x.split(' ')))))

    #     xlabels=[]
    #     for i in range(len(df_selected)):
    #         xlabels += [f'file_{i+1}']*len(r_width_list[i])
    #     r_width_flattened = list(itertools.chain(*r_width_list))
    #     return px.strip(x=xlabels, y=r_width_flattened, labels={'x':'', 'y':'Milli Second'})
    
    # # FPD plot
    # @app.callback(
    #     Output('fpd_graph', 'figure'),
    #     Input('datatable', 'selected_rows'),
    #     State('datatable', 'data')
    # )
    # def fpd_plot(selected_rows, data):
    #     if not selected_rows:
    #         return px.strip()
    #     selected_data = [data[i] for i in selected_rows]
    #     # convert list of dict to dataframe
    #     data_df = pd.DataFrame(selected_data)
    #     data_df["time"] = pd.to_datetime(data_df["time"])
    #     # filter only rows that are selected and preserve the selection order
    #     df_selected = pd.merge(data_df["time"],cardio_db, how="left", on="time", sort=False)
    #     # convert string of r_amplitudes to list of int
    #     fpd_list = list(df_selected['fpds_str'].apply(lambda x: list(map(float, x.split(' ')))))

    #     xlabels=[]
    #     for i in range(len(df_selected)):
    #         xlabels += [f'file_{i+1}']*len(fpd_list[i])
    #     fpd_flattened = list(itertools.chain(*fpd_list))
    #     return px.strip(x=xlabels, y=fpd_flattened, labels={'x':'', 'y':'Milli Second'})
    
    # # conduction speed plot
    # @app.callback(
    #     Output('conduction_graph', 'figure'),
    #     Input('datatable', 'selected_rows'),
    #     State('datatable', 'data')
    # )
    # def conduction_plot(selected_rows, data):
    #     if not selected_rows:
    #         return px.strip()
    #     selected_data = [data[i] for i in selected_rows]
    #     # convert list of dict to dataframe
    #     data_df = pd.DataFrame(selected_data)
    #     data_df["time"] = pd.to_datetime(data_df["time"])
    #     # filter only rows that are selected and preserve the selection order
    #     df_selected = pd.merge(data_df["time"],cardio_db, how="left", on="time", sort=False)
    #     # convert string of r_amplitudes to list of int
    #     speed_list = list(df_selected['conduction_speed_str'].apply(lambda x: list(map(float, x.split(' ')))))

    #     xlabels=[]
    #     for i in range(len(df_selected)):
    #         xlabels += [f'file_{i+1}']*len(speed_list[i])
    #     speed_flattened = list(itertools.chain(*speed_list))
    #     return px.strip(x=xlabels, y=speed_flattened, labels={'x':'', 'y':'cm/s'})

    @app.callback(
        Output('tab1_graphs', 'figure'),
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def tab1_graphs(selected_rows, data):
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=("R amplitude", "R width", "FPD", "Conduction speed")
        )
        fig.update_layout(height=900)
        if not selected_rows:
            fig.add_trace(go.Scatter(), row=1, col=1)
            fig.add_trace(go.Scatter(), row=1, col=2)
            fig.add_trace(go.Scatter(), row=2, col=1)
            fig.add_trace(go.Scatter(), row=2, col=2)
            return fig

        selected_data = [data[i] for i in selected_rows]
        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        data_df["time"] = pd.to_datetime(data_df["time"])
        # filter only rows that are selected and preserve the selection order
        df_selected = pd.merge(data_df["time"],cardio_db, how="left", on="time", sort=False)
        
        # convert string of r_amplitudes to list of int
        r_amp_list = list(df_selected['r_amplitudes_str'].apply(lambda x: list(map(int, x.split(' ')))))
        r_amp_flattened = list(itertools.chain(*r_amp_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(r_amp_list[i])
        fig.add_trace(px.strip(x=xlabels, y=r_amp_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=1, col=1)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['r_amplitudes_mean'].to_list(),
                         error_y_array=df_selected['r_amplitudes_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=30, line=dict(width=2)),
                         showlegend=False),
                         row=1, col=1)
        
        # convert string of r_widths to list of int
        r_width_list = list(df_selected['r_widths_str'].apply(lambda x: list(map(float, x.split(' ')))))
        r_width_flattened = list(itertools.chain(*r_width_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(r_width_list[i])
        fig.add_trace(px.strip(x=xlabels, y=r_width_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=1, col=2)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['r_widths_mean'].to_list(),
                         error_y_array=df_selected['r_widths_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=30, line=dict(width=2)),
                         showlegend=False),
                         row=1, col=2)

        # convert string of FPDs to list of int
        fpd_list = list(df_selected['fpds_str'].apply(lambda x: list(map(float, x.split(' ')))))
        fpd_flattened = list(itertools.chain(*fpd_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(fpd_list[i])
        fig.add_trace(px.strip(x=xlabels, y=fpd_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=2, col=1)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['fpds_mean'].to_list(),
                         error_y_array=df_selected['fpds_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=30, line=dict(width=2)),
                         showlegend=False),
                         row=2, col=1)
        
        # convert string of conduction speed to list of int
        speed_list = list(df_selected['conduction_speed_str'].apply(lambda x: list(map(float, x.split(' ')))))
        speed_flattened = list(itertools.chain(*speed_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(speed_list[i])
        fig.add_trace(px.strip(x=xlabels, y=speed_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=2, col=2)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['conduction_speed_mean'].to_list(),
                         error_y_array=df_selected['conduction_speed_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=30, line=dict(width=2)),
                         showlegend=False),
                         row=2, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="Micro Volt", showgrid=False, row=1, col=1)
        fig.update_yaxes(title_text="Milli Second", showgrid=False, row=1, col=2)
        fig.update_yaxes(title_text="Milli Second", showgrid=False, row=2, col=1)
        fig.update_yaxes(title_text="cm/s", showgrid=False, row=2, col=2)
        return fig

    app.layout = html.Div([
        html.H1(children='CardioMEA Dashboard', style={'textAlign':'center'}),
        dbc.Row([          
            dbc.Card(
                html.Div([
                    # check list to choose cell lines and a switch to choose between single (latest) or multiple files per recording     
                    checklist,
                    html.Br(),
                    switch,
                    html.Br(),
                    html.H4("List of processed files"),
                    # table to show shorlisted files
                    table,
                    html.Br(),
                    html.Div(id='selected_data',children=[]),      
                ]), color="light", style={"width": "95%"},
            ),
        ], justify="center"),
        html.Br(),
        # dbc.Row(
        #     dbc.CardGroup([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("R amplitudes", style={'textAlign':'center'}),
        #                 dcc.Graph(id='r_amp_graph'),
        #             ]),
        #         ]),
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("R widths", style={'textAlign':'center'}),
        #                 dcc.Graph(id='r_width_graph'),
        #             ]),
        #         ]),
        #     ], style={"width": "95%"}),
        #     justify="center",
        # ),
        # dbc.Row(
        #     dbc.CardGroup([
        #         dbc.Card([
        #             dbc.CardBody([
        #                 html.H4("FPDs", style={'textAlign':'center'}),
        #                 dcc.Graph(id='fpd_graph'),
        #             ]),
        #         ]),
        #         dbc.Card([
        #             dbc.CardBody([
        #             html.H4("Conduction speed", style={'textAlign':'center'}),
        #             dcc.Graph(id='conduction_graph'),
        #             ]),
        #         ]),
        #     ], style={"width": "95%"}),
        #     justify="center",
        # ),
        dbc.Row(
            dbc.CardGroup([
                dbc.Card(
                    dbc.Tabs(
                        [
                            dbc.Tab(dcc.Graph(id='tab1_graphs'), label="Data distribution"),
                            dbc.Tab(html.H4('Here comes recording info'), label="Recording info"),
                            dbc.Tab(html.H4('Here comes feature analysis'), label="Feature analysis"),
                        ], 
                    ),
                ),
            ], style={"width": "97%"}),
            justify="center",
        ),
    ])

    # open the URL with the default web browser of the user’s computer
    print("Ctrl + C to exit.")
    # webbrowser.open_new(f"http://127.0.0.1:{port}/")
    
    app.run(debug=True, port=port)

    
