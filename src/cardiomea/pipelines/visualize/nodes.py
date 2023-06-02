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
        page_size=10,    
        page_current=0, 
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
            Input("reset_button", "n_clicks")
        ],
    )
    def reset_selected_rows(checklist, switch, reset):
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
    
    header = ['gain','active_area_in_percent','rec_duration','rec_proc_duration','n_beats','mean_nni','sdnn','sdsd','nni_50','pnni_50','nni_20','pnni_20','rmssd','median_nni','range_nni','cvsd','cvnni','mean_hr','max_hr','min_hr','std_hr','triangular_index','tinn','lf','hf','lf_hf_ratio','lfnu','hfnu','total_power','vlf','csi','cvi','modified_csi','sd1','sd2','ratio_sd2_sd1']

    @app.callback(
        Output('feature_table', 'children'),
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def feature_table(selected_rows, data):
        if not selected_rows:
            return ''
        selected_data = [data[i] for i in selected_rows]
        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        data_df["time"] = pd.to_datetime(data_df["time"])
        # filter only rows that are selected and preserve the selection order
        df_selected = pd.merge(data_df["time"], cardio_db, how="left", on="time", sort=False)
        df = df_selected[header].applymap(lambda x: round(x,1) if x is not None else None)
        df.insert(0,'file',[f'file_{i+1}' for i in range(len(df_selected))])
        df_T = df.T
        df_T.rename(columns=df_T.iloc[0], inplace=True)
        df_T.drop(df_T.index[0], inplace=True)  
        df_T.reset_index(drop=False, inplace=True)  

        return dbc.Table.from_dataframe(df_T, id='feature_table', striped=True, bordered=True, hover=True)

    app.layout = html.Div([
        html.H1(children='CardioMEA Dashboard', style={'textAlign':'center'}),
        dbc.Row([          
            dbc.Card(
                html.Div([
                    # check list to choose cell lines     
                    checklist,
                    html.Br(),
                    # switch to choose between single (latest) or multiple files per recording 
                    switch,
                    html.Br(),
                    html.H4("List of processed files"),
                    # table to show shorlisted files
                    table,
                    dbc.Button("Reset selections", id="reset_button", color="primary", n_clicks=0),
                    html.Br(),
                    html.Div(id='selected_data',children=[]),      
                ]), color="light", style={"width": "95%"}
            ),
        ], justify="center"),
        html.Br(),
        dbc.Row(
            dbc.CardGroup([
                dbc.Card(
                    dbc.Tabs([                    
                        dbc.Tab(dcc.Graph(id='tab1_graphs'), label="Data distribution"),
                        dbc.Tab([
                            html.Div(id='feature_table',children=[],style={"overflow": "scroll"}), 
                            html.H5('Click on the link below to see documentations of the HRV features.'),
                            dcc.Link('Link to the HRV documentation', href="https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html"),
                        ], label="Recording info"),
                        dbc.Tab(html.H4('Here comes feature analysis'), label="Feature analysis"),
                    ]),
                ),
            ], style={"width": "97%"}),
            justify="center",
        ),
    ])

    # open the URL with the default web browser of the user’s computer
    print("Ctrl + C to exit.")
    # webbrowser.open_new(f"http://127.0.0.1:{port}/")
    
    app.run(debug=True, port=port)

    
