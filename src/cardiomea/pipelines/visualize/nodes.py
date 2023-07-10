"""
This is a boilerplate pipeline 'visualize'
generated using Kedro 0.18.7
"""
from dash import Dash, dcc, dash_table, html, Output, Input, State, ctx
from dash.exceptions import PreventUpdate
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from optimalflow.autoFS import dynaFS_clf
from plotly.subplots import make_subplots
from sklearn.inspection import permutation_importance
from autosklearn.classification import AutoSklearnClassifier
from sklearn.model_selection import StratifiedKFold
import dash_bootstrap_components as dbc
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import base64
import dash_bio
import webbrowser
import itertools

def dashboard(cardio_db_FP_all,cardio_db_AP,port,base_directory):
    bel_logo = 'data/01_raw/bel_ohne_schrift.jpg'
    logo_base64 = base64.b64encode(open(bel_logo, 'rb').read()).decode('ascii')
    
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    ##### Extracellular recordings #####
    # df = cardio_db_FP_all.sort_values('time').groupby('file_path_full').tail(1).reset_index(drop=True)
    # df_SQT5_base = df[df['file_path_full'].str.contains('base')]
    # df_SQT5_base = df_SQT5_base[df_SQT5_base['cell_line'].str.contains('SQT5')]

    # df_SQT5 = df[df['file_path_full'].str.contains('highamp_1.raw')]
    # df_SQT5 = df_SQT5[df_SQT5['cell_line'].str.contains('SQT5')]

    # df_BlHi = df[df['file_path_full'].str.contains('highamp_1.raw')]
    # df_BlHi = df_BlHi[df_BlHi['cell_line'].str.contains('BlHi')]

    # df_D5 = df[df['file_path_full'].str.contains('highamp_1.raw')]
    # df_D5 = df_D5[df_D5['cell_line'].str.contains('D5')]

    # cardio_db_FP = pd.concat([df_SQT5_base, df_SQT5, df_BlHi, df_D5], axis=0).reset_index(drop=True)

    df = cardio_db_FP_all.copy()
    df = df[~df['cell_line'].str.contains('PKP2-KO|PKP2-WT')]
    cardio_db_FP = df[~df['file_path_full'].str.contains('overall')]

    # cardio_db_FP = cardio_db_FP_all.copy()

    cell_lines = cardio_db_FP["cell_line"].unique()
    cardio_db_FP["file_path"] = cardio_db_FP["file_path_full"].apply(lambda x: x.removeprefix(base_directory))
    col_simple = ['cell_line','compound','file_path','time','note']
    columns = [{'name': i, 'id': i} for i in col_simple]

    checklist = html.Div(
        [
            html.H4("Choose cell lines"),
            dbc.Checklist(
                options=[{"label": c, "value": c} for c in cell_lines],
                value=[],
                id="checklist_cell_lines",
                inline=True,
            ),
            html.Br(),
            html.H4("Choose experiment compounds"),
            dbc.Checklist(
                options=[],
                # select all by default
                value=[],
                id="checklist_compounds",
                inline=True,
            ),
        ]
    )

    @app.callback(
        Output("checklist_compounds", "options"),
        Output("checklist_compounds", "value"),
        Input("checklist_cell_lines", "value"),
    )
    def update_compound_list(checklist_cell_lines):
        compound_list = cardio_db_FP["compound"].loc[cardio_db_FP["cell_line"].isin(checklist_cell_lines)].unique()
        options = [{'label': c, 'value': c} for c in compound_list]
        return options, compound_list

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
        columns=[{"name": i, "id": i} for i in ['cell_line','compound','file_path','time_processed','note']],
        data=[],
        page_size=10,    
        page_current=0, 
        editable=False,
        row_selectable="multi",
        selected_rows=[],
        style_table={'overflowX': 'scroll'},
        style_cell={'textAlign': 'left', 'minWidth': '150px'},
    )

    @app.callback(
        Output("datatable", "data"),
        [
            Input("checklist_cell_lines", "value"),
            Input("checklist_compounds", "value"),
            Input("latest_only", "value"),
        ],
    )
    def update_table(checklist_cell_lines, checklist_compounds, switch_value):
        # show only the latest data if switch is on
        if switch_value:
            df = cardio_db_FP.sort_values('time').groupby('file_path').tail(1).reset_index(drop=True)
        else:
            df = cardio_db_FP.copy()
        # filter only cell lines that are selected 
        df_selected = df.loc[df["cell_line"].isin(checklist_cell_lines) & df["compound"].isin(checklist_compounds), col_simple]
        # add a column with time in string format for display
        df_selected["time_processed"] = df_selected["time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        table_data = [{c['id']: df_selected[c['name']].iloc[i] for c in columns+[{'name': 'time_processed', 'id': 'time_processed'}]} for i in range(len(df_selected))]

        return table_data

    # Reset selections if table is changed by the user
    @app.callback(
        Output("datatable", "selected_rows"),
        [
            Input("checklist_cell_lines", "value"),
            Input("checklist_compounds", "value"),
            Input("latest_only", "value"),
            Input("reset_button", "n_clicks"),
            Input("select_all_button", "n_clicks"),
            Input("datatable", "data"),
        ],
    )
    def reset_selected_rows(checklist_cell_lines, checklist_compounds, switch, reset, select_all, data):
        updated_input = ctx.triggered_id
        if updated_input == "select_all_button":
            return [i for i in range(len(data))]
        else:
            return []

    header = ['gain','n_electrodes_sync','active_area_in_percent','rec_duration','rec_proc_duration','n_beats','mean_nni','sdnn','sdsd','nni_50','pnni_50','nni_20','pnni_20','rmssd','median_nni','range_nni','cvsd','cvnni','mean_hr','max_hr','min_hr','std_hr']

    @app.callback(
        [
            Output('selected_data', 'children'),
            Output('tab1_graphs', 'figure'),
            Output('feature_table', 'children'),
        ],
        Input('datatable', 'selected_rows'),
        State('datatable', 'data')
    )
    def filelist_graphs_table(selected_rows, data):
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=("R amplitude", "R width", "FPD", "Conduction speed"), horizontal_spacing=0.1, vertical_spacing=0.1
        )
        if not selected_rows:
            fig.add_trace(go.Scatter(), row=1, col=1)
            fig.add_trace(go.Scatter(), row=1, col=2)
            fig.add_trace(go.Scatter(), row=2, col=1)
            fig.add_trace(go.Scatter(), row=2, col=2)
            fig.for_each_xaxis(lambda x: x.update(showgrid=False, zeroline=False))
            fig.for_each_yaxis(lambda x: x.update(showgrid=False, zeroline=False))
            return '', fig, ''

        selected_data = [data[i] for i in selected_rows]
        selected_file_list = html.Div([
            html.H4('Selected Files in Order:'),
            html.Ul([
                html.Li(f"file_{cnt+1}: {row['cell_line']} ({row['file_path']})")
                for cnt, row in enumerate(selected_data)
            ])
        ])

        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        data_df["time"] = pd.to_datetime(data_df["time"])
        # filter only rows that are selected and preserve the selection order
        df_selected = pd.merge(data_df["time"], cardio_db_FP, how="left", on="time", sort=False)
        
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
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
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
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
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
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
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
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
                         showlegend=False),
                         row=2, col=2)

        # Update yaxis properties
        fig.update_yaxes(title_text="Micro Volt", showgrid=False, row=1, col=1)
        fig.update_yaxes(title_text="Milli Second", showgrid=False, row=1, col=2)
        fig.update_yaxes(title_text="Milli Second", showgrid=False, row=2, col=1)
        fig.update_yaxes(title_text="cm/s", showgrid=False, row=2, col=2)
        fig.update_layout(height=900)

        # create feature table for tab 2
        df = df_selected[header].applymap(lambda x: round(x,1) if x is not None else None)
        df.insert(0,'file',[f'file_{i+1}' for i in range(len(df_selected))])
        df_T = df.T
        df_T.rename(columns=df_T.iloc[0], inplace=True)
        df_T.drop(df_T.index[0], inplace=True)  
        df_T.reset_index(drop=False, inplace=True)
        feat_table = dbc.Table.from_dataframe(df_T, id='feature_table', striped=True, bordered=True, hover=True)

        return selected_file_list, fig, feat_table
    
    @app.callback(
        Output('download_output', 'children'),
        Input('download_button', 'n_clicks'),
        State('tab1_graphs', 'figure'),
    )
    def download_output(n_clicks, fig):
        if n_clicks==0:
            return ''
        elif fig is not None:
            save_path = "data/08_reporting/data_distribution.pdf"
            pio.write_image(fig, save_path, format="pdf", engine="kaleido", width=1200, height=900)
            return 'Figure is downloaded to '+save_path
        else:
            return 'No figure to download'
    
    df_columns = cardio_db_FP.columns
    rm_columns = ['time','cell_line','compound','note','file_path_full','file_path','gain','rec_duration','rec_proc_duration','n_electrodes_sync','r_amplitudes_str','r_amplitudes_std','r_widths_str','r_widths_std','fpds_str','fpds_std','conduction_speed_str','conduction_speed_std']
    feature_columns = [f for f in df_columns if f not in rm_columns]
    
    feature_analysis = html.Div([
        html.H4('Select features'),
        dbc.Checklist(
            options=[{"label": c, "value": c} for c in feature_columns],
            value=[c for c in feature_columns],
            id="checklist_features",
            inline=True,
        ),
    ])

    @app.callback(
        [
            Output('feat_dependency_graphs','figure'),
            Output('dendrogram','figure'),
        ],
        [
            Input('checklist_features','value'),
            Input('datatable', 'selected_rows'),
        ],
        State('datatable', 'data')
    )
    def feat_dependency_graphs(features, selected_rows, data):
        fig1 = make_subplots(rows=1, cols=2, subplot_titles=["Correlation","Multicollinearity"])
        fig2 = make_subplots(rows=1, cols=1, subplot_titles=["Similarity"])
        if len(features)<2 or len(selected_rows)<2:
            fig1.add_trace(go.Scatter(), row=1, col=1)
            fig1.add_trace(go.Scatter(), row=1, col=2)
            fig1.for_each_xaxis(lambda x: x.update(showgrid=False, zeroline=False))
            fig1.for_each_yaxis(lambda x: x.update(showgrid=False, zeroline=False))
            fig2.add_trace(go.Scatter(), row=1, col=1)
            fig2.for_each_xaxis(lambda x: x.update(showgrid=False, zeroline=False))
            fig2.for_each_yaxis(lambda x: x.update(showgrid=False, zeroline=False))
            return fig1, fig2
        
        selected_data = [data[i] for i in selected_rows]
        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        data_df["time"] = pd.to_datetime(data_df["time"])
        # filter only rows that are selected and preserve the selection order
        df_selected = pd.merge(data_df["time"], cardio_db_FP, how="left", on="time", sort=False)
        # keep only selected features
        df_filtered = df_selected[features]

        # plot correlation map using plotly
        corr = df_filtered.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        half_corr=corr.mask(mask)
        
        fig1.add_trace(go.Heatmap(z=half_corr, x=corr.columns, y=corr.columns, colorscale='RdBu', zmin=-1, zmax=1, colorbar_thickness=10, colorbar_x=0.45), row=1, col=1)
        fig1.update_xaxes(side="bottom", tickangle=45, tickvals = np.arange(len(corr)-1), row=1, col=1)
        fig1.update_yaxes(autorange='reversed', tickvals = np.arange(1,len(corr)), row=1, col=1)
        fig1.update_layout({'plot_bgcolor':'rgba(0,0,0,0)'})

        # plot multicollinearity using VIF    
        # calculate VIF only if number of selected features is less than number of selected files    
        if len(selected_rows) > len(features):
            df = df_filtered.copy()
            # fill missing values with mean value of each feature
            df.fillna(value=df.mean(), inplace=True)
            df['intercept'] = 1
            # calculating VIF for each feature
            vif = pd.DataFrame()
            vif["feature"] = df.columns
            vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
            vif = vif[vif['feature']!='intercept']

            fig1.add_trace(go.Bar(x=vif['feature'], y=vif['VIF'], marker_color='rgb(158,202,225)'), row=1, col=2)
            fig1.update_xaxes(tickangle=45, row=1, col=2)
            fig1.update_yaxes(title_text="Variance inflation factor (VIF)", row=1, col=2)
            fig1.update_layout(height=700)

        # plot similarity map
        df = df_filtered.copy()
        # fill missing values with mean value of each feature
        df.fillna(value=df.mean(), inplace=True)
        df_norm = (df-df.min())/(df.max()-df.min())
        # fill missing values with 0, where 0 means data points are all the same in that feature
        df_norm.fillna(value=0, inplace=True)
        clustergram = dash_bio.Clustergram(
            data=df_norm.values.T,
            row_labels=df_norm.columns.to_list(),
            hidden_labels='column',
            width=1500,
            height=700,
            cluster='row',
            line_width=4,
            center_values=False, 
            color_map= [[0.0, '#71AFD9'],[1.0, '#71D993']],
            color_list={
                'row': ['#636EFA', '#00CC96', '#19D3F3'],
                'bg': '#506784'
            },
        )

        return fig1, clustergram
        
    # automated feature selection using Optimal Flow
    auto_feat_selection = dbc.Card([
        html.Div([
            html.H4('Automated feature selection'),
            dbc.Row([
                dbc.Col([
                    html.H5('How many features to select?'),
                    dcc.Slider(
                        id='slider_feat_selection',
                        min=2,
                        max=len(feature_columns),
                        step=1,
                        value=2,
                    ),
                ]),
                dbc.Col(html.Div(dbc.Button("Run", id="run_optflow", color="primary", n_clicks=0), className="d-grid col-6 mx-auto"), width=2),
                dbc.Col([
                    html.H5('Best features calculated by Optimal Flow'),
                    html.Div(id='best_features'),
                ], width=3),
            ]),
        ], style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '5px', 'margin-bottom': '10px'}),
    ])

    @app.callback(
        Output('slider_feat_selection','max'),
        Input('checklist_features','value'),
    )
    def set_slider_range(features):
        if len(features)<2:
            return 2
        else:
            return len(features) 
    
    @app.callback(
        Output('best_features','children'),
        [
            Input('slider_feat_selection','value'),
            Input('run_optflow','n_clicks'),
            Input('checklist_features','value'),
            Input('datatable', 'selected_rows'),
        ],
        State('datatable', 'data')
    )
    def auto_feature_selection(n_desired, n_clicks, features, selected_rows, data):
        updated_input = ctx.triggered_id
        if updated_input=='run_optflow': # if Run button is pressed
            if len(features)<2 or len(selected_rows)<2:
                return 'Select more features and/or more data rows.'
            selected_data = [data[i] for i in selected_rows]
            # convert list of dict to dataframe
            data_df = pd.DataFrame(selected_data)
            data_df["time"] = pd.to_datetime(data_df["time"])
            # filter only rows that are selected and preserve the selection order
            df_selected = pd.merge(data_df["time"], cardio_db_FP, how="left", on="time", sort=False)
            # keep only selected features
            df_filtered = df_selected[features+['cell_line']]

            # define dataset
            X = df_filtered.drop('cell_line',axis=1,inplace=False)
            # remove rows which contain NaN
            y = df_filtered.loc[X.notna().all(axis=1),'cell_line']
            X = X.loc[X.notna().all(axis=1)]

            clf_fs = dynaFS_clf(fs_num=n_desired, cv=3, input_from_file=True)
            feat_list = clf_fs.fit(X,y)

            return html.Div([
                html.Ul([html.Li(f) for f in feat_list[1]])
            ])
        else:
            return 'Click Run'

    autoML = html.Div([
        dbc.Card([
            html.Div([
                html.H4('Automated machine learning'),
                dbc.Row([
                    dbc.Col([
                        html.H5("Missing data"),
                        dbc.RadioItems(
                            options=[
                                {"label": "Drop missing data", "value": 'drop'},
                                {"label": "Impute missing data", "value": 'impute'},
                            ],
                            value='drop',
                            id="impute_input",
                        ),
                    ]),
                    dbc.Col([
                        html.H5("Cross validation folds"),
                        dbc.Input(id='cv', type="number", min=2, max=10, step=1, value=5),
                    ]),
                    dbc.Col([
                        html.H5("Time limit per fold (min)"),
                        dbc.Input(id='time_limit', type="number", min=1, max=30, step=1, value=3),
                    ]),
                    dbc.Col([
                        html.H5("Permutation repeats"),
                        dbc.Input(id='perm_repeats', type="number", min=1, max=20, step=1, value=10),
                    ]),
                ]),
                html.Br(),
                dbc.Button("Run AutoML", id="run_automl", className="mb-3", color="primary", n_clicks=0),
                dcc.Interval(id="progress_interval", n_intervals=0, interval=500, disabled=True),
                dbc.Collapse(dbc.Progress(id="progress"), id="collapse", is_open=False),
            ], style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '5px', 'margin-bottom': '10px'}),
        ]),
        html.Br(),
        dbc.Card([
            html.Div([
                dbc.Row([
                    dbc.Col(html.Div(id='automl_scores', style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '10px', 'margin-bottom': '10px'}), width=3),
                    dbc.Col(html.Div(
                            [
                                html.H5("Feature importance"), 
                                dbc.Label("Select dataset"),
                                dbc.RadioItems(
                                    options=[
                                        {"label": "Training data", "value": 'train_data'},
                                        {"label": "Test data", "value": 'test_data'},
                                    ],
                                    value='test_data', id="dataset_input", inline=True,
                                ),
                                html.Div(dcc.Graph(id='automl_perm_train'), hidden=True), # hidden graph to store data
                                html.Div(dcc.Graph(id='automl_perm_test'), hidden=True), # hidden graph to store data
                                dcc.Graph(id='automl_perm'),
                            ], style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '10px'}
                    )),
                ]),
            ], style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '5px', 'margin-bottom': '10px'}),
        ]),
    ])

    @app.callback(
        Output("progress_interval", "max_intervals"),
        Input("progress", "value"), 
    )
    def stop_progress_bar(prog):
        if prog==100:
            return 0
        else:
            return -1
    
    @app.callback(
        [
            Output("progress_interval", "n_intervals"),
            Output("progress_interval", "disabled"),
            Output("collapse", "is_open"),
        ],
        [
            Input("run_automl", "n_clicks"),
            Input("impute_input", "value"),
            Input("cv", "value"),
            Input("time_limit", "value"),
            Input("perm_repeats", "value"),
        ],
    )
    def reset_progress_bar(n_clicks, impute, cv, time_limit, perm_repeats):
        updated_input = ctx.triggered_id
        if updated_input=='run_automl': # if Run button is pressed
            return 0, False, True
        else: # if configuration is changed
            return 0, True, False

    @app.callback(
        [
            Output("progress", "value"), 
            Output("progress", "label"),
        ],
        [
            Input("progress_interval", "n_intervals"),
            Input("progress_interval", "interval"),
            Input("time_limit", "value"),
            Input("cv", "value"),
            Input("run_automl", "n_clicks"),
        ],
    )
    def update_progress_bar(n, interval, time_limit, cv, n_clicks):
        if n_clicks:
            time_limit_ms = time_limit * 60 * 1000 * cv 
            progress = round(100 * n * interval / time_limit_ms)
            return progress, f"{progress} %" if progress >= 5 else "" # add text only after 5% progress 
        else:
            return 0, ""
       
    @app.callback(
        [
            Output("automl_scores", "children"),
            Output("automl_perm_train", "figure"),
            Output("automl_perm_test", "figure"),
        ],
        [
            Input("run_automl", "n_clicks"),
            Input("impute_input", "value"),
            Input("cv", "value"),
            Input("time_limit", "value"),
            Input("perm_repeats", "value"),
            Input('checklist_features','value'),
            Input('datatable', 'selected_rows'),
        ],
        State('datatable', 'data'),
    )
    def automl(n_clicks, impute, cv, time_limit, perm_repeats, features, selected_rows, data):
        updated_input = ctx.triggered_id
        if updated_input=='run_automl': # if Run button is pressed
            selected_data = [data[i] for i in selected_rows]
            # convert list of dict to dataframe
            data_df = pd.DataFrame(selected_data)
            # check if at least 2 cell lines are selected
            if data_df['cell_line'].unique().size < 2:
                return 'Please select at least 2 cell lines.', go.Figure(), True, go.Figure(), False
            else:
                data_df["time"] = pd.to_datetime(data_df["time"])
                # filter only rows that are selected and preserve the selection order
                df_selected = pd.merge(data_df["time"], cardio_db_FP, how="left", on="time", sort=False)
                # keep only selected features
                df_filtered = df_selected[features+['cell_line']]

                if impute == 'drop':
                    ## drop NaN entries
                    y = df_filtered.loc[df_filtered.notna().all(axis=1),'cell_line'].to_numpy()
                    X = df_filtered.loc[df_filtered.notna().all(axis=1)].drop('cell_line',axis=1,inplace=False).values
                else:
                    ## keep NaN entries (will be later imputed by mean values)
                    X = df_filtered.drop('cell_line',axis=1,inplace=False).values
                    y = df_filtered['cell_line'].to_numpy()

                train_scores=[]
                test_scores=[]
                feat_imp_train = pd.DataFrame(columns=features)
                feat_imp_test = pd.DataFrame(columns=features)
                skf = StratifiedKFold(n_splits=cv)
                for train_index, test_index in skf.split(X, y):
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]

                    # run AutoML
                    automl = AutoSklearnClassifier(
                        time_left_for_this_task = time_limit * 60,
                        per_run_time_limit = 30,
                        # do not create an ensemble of top-performin models to prevent overfitting.
                        ensemble_kwargs = {'ensemble_size': 1},
                        initial_configurations_via_metalearning = 0,
                        n_jobs = 4,
                    )
                    automl.fit(X_train, y_train)

                    # get performance scores (accuracy)   
                    train_scores.append(automl.score(X_train, y_train))
                    test_scores.append(automl.score(X_test, y_test))

                    # get feature importances (permutation analysis)
                    r = permutation_importance(automl, X_train, y_train, n_repeats=perm_repeats)
                    sort_idx = r.importances_mean.argsort()[::-1]
                    feat_imp_train = pd.concat([feat_imp_train, pd.DataFrame(r.importances[sort_idx].T, columns=features)], ignore_index=True)

                    r = permutation_importance(automl, X_test, y_test, n_repeats=perm_repeats)
                    sort_idx = r.importances_mean.argsort()[::-1]
                    feat_imp_test = pd.concat([feat_imp_test, pd.DataFrame(r.importances[sort_idx].T, columns=features)], ignore_index=True)
                
                output = html.Div([
                    html.H5("AutoML scores"),
                    html.Ul([
                        html.Li(f"Training data score: {np.mean(train_scores):.3f} +/- {np.std(train_scores):.3f}"),
                        html.Li(f"Test data score: {np.mean(test_scores):.3f} +/- {np.std(test_scores):.3f}"),
                    ]),
                    html.Br(),
                    html.H5("Best ML model"),
                    html.P(str(list(automl.show_models().values())[0]['sklearn_classifier'])),
                ])

                fig1 = go.Figure()
                feat_imp_train.loc['mean'] = feat_imp_train.mean()
                feat_imp_train.sort_values(by='mean', axis=1, ascending=False, inplace=True)
                for f in feat_imp_train.columns:
                    fig1.add_trace(go.Box(y=feat_imp_train[f], name=f, showlegend=False))
                fig1.update_layout(xaxis_title="Features", yaxis_title="Feature importance")
                fig1.update_xaxes(tickangle=45)
                
                fig2 = go.Figure()
                feat_imp_test.loc['mean'] = feat_imp_test.mean()
                feat_imp_test.sort_values(by='mean', axis=1, ascending=False, inplace=True)
                for f in feat_imp_test.columns:
                    fig2.add_trace(go.Box(y=feat_imp_test[f], name=f, showlegend=False))                                
                fig2.update_layout(xaxis_title="Features", yaxis_title="Feature importance")
                fig2.update_xaxes(tickangle=45)

                return output, fig1, fig2

        else: 
            output = html.Div([
                html.H5("AutoML scores"),
                html.P("Click Run AutoML to start AutoML training."),
            ])
            return output, go.Figure(), go.Figure()

    @app.callback(
        Output("automl_perm", "figure"),
        [
            Input("dataset_input", "value"),
            Input("automl_perm_train", "figure"),
            Input("automl_perm_test", "figure"),
        ]
    )
    def update_automl_perm(dataset, fig1, fig2):
        if dataset == 'train_data':
            return fig1
        else:
            return fig2

    ##### Intracellular recordings #####
    cell_lines_AP = cardio_db_AP["cell_line"].unique()
    cardio_db_AP["file_path"] = cardio_db_AP["file_path_full"].apply(lambda x: x.removeprefix(base_directory))

    checklist_AP = html.Div(
        [
            html.H4("Choose cell lines"),
            dbc.Checklist(
                options=[{"label": c, "value": c} for c in cell_lines_AP],
                value=[],
                id="checklist_cell_lines_AP",
                inline=True,
            ),
            html.Br(),
            html.H4("Choose experiment compounds"),
            dbc.Checklist(
                options=[],
                # select all by default
                value=[],
                id="checklist_compounds_AP",
                inline=True,
            ),
        ]
    )
    @app.callback(
        Output("checklist_compounds_AP", "options"),
        Output("checklist_compounds_AP", "value"),
        Input("checklist_cell_lines_AP", "value"),
    )
    def update_compound_list_AP(checklist_cell_lines):
        compound_list = cardio_db_AP["compound"].loc[cardio_db_AP["cell_line"].isin(checklist_cell_lines)].unique()
        options = [{'label': c, 'value': c} for c in compound_list]
        return options, compound_list

    switch_AP = html.Div(
        [
            dbc.Switch(
                label="Show the latest data only",
                value=True,
                id="latest_only_AP",
            ),
        ]
    )

    # Define data table with checkbox column
    table_AP = dash_table.DataTable(
        id='datatable_AP',
        columns=[{"name": i, "id": i} for i in ['cell_line','compound','file_path','time_processed','note']],
        data=[],
        page_size=10,    
        page_current=0, 
        editable=False,
        row_selectable="multi",
        selected_rows=[],
        style_table={'overflowX': 'scroll'},
        style_cell={'textAlign': 'left', 'minWidth': '150px'},
    )

    @app.callback(
        Output("datatable_AP", "data"),
        [
            Input("checklist_cell_lines_AP", "value"),
            Input("checklist_compounds_AP", "value"),
            Input("latest_only_AP", "value"),
        ],
    )
    def update_table_AP(checklist_cell_lines, checklist_compounds, switch_value):
        # show only the latest data if switch is on
        if switch_value:
            df = cardio_db_AP.sort_values('time').groupby('file_path').tail(1).reset_index(drop=True)
        else:
            df = cardio_db_AP.copy()
        # filter only cell lines that are selected 
        df_selected = df.loc[df["cell_line"].isin(checklist_cell_lines) & df["compound"].isin(checklist_compounds), col_simple]
        # add a column with time in string format for display
        df_selected["time_processed"] = df_selected["time"].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        table_data = [{c['id']: df_selected[c['name']].iloc[i] for c in columns+[{'name': 'time_processed', 'id': 'time_processed'}]} for i in range(len(df_selected))]

        return table_data

    # Reset selections if table is changed by the user
    @app.callback(
        Output("datatable_AP", "selected_rows"),
        [
            Input("checklist_cell_lines_AP", "value"),
            Input("checklist_compounds_AP", "value"),
            Input("latest_only_AP", "value"),
            Input("reset_button_AP", "n_clicks")
        ],
    )
    def reset_selected_rows_AP(checklist_cell_lines, checklist_compounds, switch, reset):
        return []
    
    header_AP = ['gain','rec_duration','rec_proc_duration','electroporation_yield','n_electrodes']
    
    @app.callback(
        [
            Output('selected_data_AP', 'children'),
            Output('tab1_graphs_AP', 'figure'),
            Output('feature_table_AP', 'children'),
        ],
        Input('datatable_AP', 'selected_rows'),
        State('datatable_AP', 'data')
    )
    def filelist_graphs_table_AP(selected_rows, data):
        fig = make_subplots(
            rows=2, cols=2, subplot_titles=("AP amplitude", "Depolarization time", "APD50", "APD90"), horizontal_spacing=0.1, vertical_spacing=0.1
        )
        if not selected_rows:
            fig.add_trace(go.Scatter(), row=1, col=1)
            fig.add_trace(go.Scatter(), row=1, col=2)
            fig.add_trace(go.Scatter(), row=2, col=1)
            fig.add_trace(go.Scatter(), row=2, col=2)
            fig.for_each_xaxis(lambda x: x.update(showgrid=False, zeroline=False))
            fig.for_each_yaxis(lambda x: x.update(showgrid=False, zeroline=False))
            return '', fig, ''
        
        selected_data = [data[i] for i in selected_rows]
        selected_file_list = html.Div([
            html.H4('Selected Files in Order:'),
            html.Ul([
                html.Li(f"file_{cnt+1}: {row['cell_line']} ({row['file_path']})")
                for cnt, row in enumerate(selected_data)
            ])
        ])

        # convert list of dict to dataframe
        data_df = pd.DataFrame(selected_data)
        data_df["time"] = pd.to_datetime(data_df["time"])
        # filter only rows that are selected and preserve the selection order
        df_selected = pd.merge(data_df["time"],cardio_db_AP, how="left", on="time", sort=False)
                
        # convert string of AP amplitudes (peak-to-peak) to list of int
        amp_list = list(df_selected['ap_amplitudes_str'].apply(lambda x: list(map(int, x.split(' ')))))
        amp_flattened = list(itertools.chain(*amp_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(amp_list[i])
        fig.add_trace(px.strip(x=xlabels, y=amp_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=1, col=1)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['ap_amplitudes_mean'].to_list(),
                         error_y_array=df_selected['ap_amplitudes_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
                         showlegend=False),
                         row=1, col=1)
        
        # convert string of depolarization time to list of int
        amp_list = list(df_selected['depolarization_time_str'].apply(lambda x: list(map(int, x.split(' ')))))
        amp_flattened = list(itertools.chain(*amp_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(amp_list[i])
        fig.add_trace(px.strip(x=xlabels, y=amp_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=1, col=2)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['depolarization_time_mean'].to_list(),
                         error_y_array=df_selected['depolarization_time_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
                         showlegend=False),
                         row=1, col=2)
        
        # convert string of APD50 to list of int
        amp_list = list(df_selected['apd50_str'].apply(lambda x: list(map(int, x.split(' ')))))
        amp_flattened = list(itertools.chain(*amp_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(amp_list[i])
        fig.add_trace(px.strip(x=xlabels, y=amp_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=2, col=1)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['apd50_mean'].to_list(),
                         error_y_array=df_selected['apd50_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
                         showlegend=False),
                         row=2, col=1)
        
        # convert string of APD90 to list of int
        amp_list = list(df_selected['apd90_str'].apply(lambda x: list(map(int, x.split(' ')))))
        amp_flattened = list(itertools.chain(*amp_list))
        xlabels=[]
        for i in range(len(df_selected)):
            xlabels += [f'file_{i+1}']*len(amp_list[i])
        fig.add_trace(px.strip(x=xlabels, y=amp_flattened).update_traces(jitter=1,opacity=0.6,marker_size=5).data[0], row=2, col=2)
        fig.add_trace(go.Scatter(x=[f'file_{i+1}' for i in range(len(df_selected))], y=df_selected['apd90_mean'].to_list(),
                         error_y_array=df_selected['apd90_std'].to_list(),
                         mode='markers',
                         marker=dict(symbol='141', color='rgba(0,0,0,0.6)', size=15, line=dict(width=1)),
                         showlegend=False),
                         row=2, col=2)
        
        # Update yaxis properties
        fig.update_yaxes(title_text="Micro Volt", showgrid=False, row=1, col=1)
        fig.update_yaxes(title_text="Milli Second", showgrid=False, row=1, col=2)
        fig.update_yaxes(title_text="Milli Second", showgrid=False, row=2, col=1)
        fig.update_yaxes(title_text="Milli Second", showgrid=False, row=2, col=2)
        fig.update_layout(height=900)

        # create feature table for tab 2
        df = df_selected[header_AP].applymap(lambda x: round(x,1) if x is not None else None)
        df.insert(0,'file',[f'file_{i+1}' for i in range(len(df_selected))])
        df_T = df.T
        df_T.rename(columns=df_T.iloc[0], inplace=True)
        df_T.drop(df_T.index[0], inplace=True)  
        df_T.reset_index(drop=False, inplace=True)  
        feat_table = dbc.Table.from_dataframe(df_T, id='feature_table_AP', striped=True, bordered=True, hover=True)
        
        return selected_file_list, fig, feat_table

    @app.callback(
        Output('download_output_AP', 'children'),
        Input('download_button_AP', 'n_clicks'),
        State('tab1_graphs_AP', 'figure'),
    )
    def download_output(n_clicks, fig):
        if n_clicks==0:
            return ''
        elif fig is not None:
            save_path = "data/08_reporting/data_distribution_AP.pdf"
            pio.write_image(fig, save_path, format="pdf", engine="kaleido", width=1200, height=900)
            return 'Figure is downloaded to '+save_path
        else:
            return 'No figure to download'

    app.layout = html.Div([
        dbc.Row([
            dbc.Col(
                html.A(html.Img(src=f'data:image/jpg;base64,{logo_base64}', height="35px"), href="https://bsse.ethz.ch/bel", target="_blank"), 
                width=2,
                style={'margin-left':'10px','margin-top':'8px'},
            ),
            dbc.Col(html.H1(children='CardioMEA Dashboard', style={'textAlign':'center'})),
            dbc.Col(width=2),
        ]),    
        dcc.Tabs([
            dcc.Tab([       
                html.Br(),
                dbc.Row([   
                    dbc.Col([
                        dbc.Card(
                            html.Div([
                                # check list to choose cell lines     
                                checklist,
                                html.Br(),
                                # switch to choose between single (latest) or multiple files per recording 
                                switch,
                                html.Br(),
                                html.H4("List of processed files"),
                                # table to show data
                                table,
                                html.Div([
                                    dbc.Button("Select all", id="select_all_button", color="primary", outline=True, size="sm", n_clicks=0),
                                    dbc.Button("Reset selections", id="reset_button", color="primary", outline=True, size="sm", n_clicks=0),
                                ]),
                                html.Br(),
                                html.Div(id='selected_data',children=[]),      
                            ], style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '5px', 'margin-bottom': '10px'}), 
                        color="#E4E6D8"),
                    ], style={'margin-left': '10px', 'margin-right': '10px'}),
                ], justify="center"),
                html.Br(),
                dbc.Row([
                    dbc.Col([
                        dbc.Tabs([                    
                            dbc.Tab([
                                dcc.Graph(id='tab1_graphs'), 
                                dbc.Button("Download figures as PDF", id="download_button", color="primary", outline=True, size="lg", n_clicks=0),
                                html.Span(id="download_output", style={"verticalAlign": "middle"}),
                            ], label="Data distribution", activeTabClassName="fw-bold", tab_id="tab1-1"),
                            dbc.Tab([
                                html.Div(id='feature_table',children=[],style={"overflow": "scroll"}), 
                                html.H5('Click on the link below to see documentations of the HRV features.'),
                                dcc.Link('Link to the HRV documentation', href="https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html"),
                            ], label="Recording info", activeTabClassName="fw-bold", tab_id="tab1-2"),
                            dbc.Tab([
                                html.Div([
                                    feature_analysis,
                                    html.Br(),
                                    dbc.Card(
                                        html.Div([
                                            html.H4('Feature dependency graphs'),
                                            dcc.Graph(id='feat_dependency_graphs'),
                                            html.H5('Similarity'),
                                            dcc.Graph(id='dendrogram'),
                                        ], style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '5px'}),
                                    ),
                                    html.Br(),
                                    auto_feat_selection,
                                    html.Br(),
                                    autoML,
                                ], style={'margin-left': '15px', 'margin-right': '15px', 'margin-top': '10px'}),
                            ], label="Feature analysis", activeTabClassName="fw-bold", tab_id="tab1-3"),
                        ], active_tab="tab1-1"), 
                    ], style={'margin-left': '15px', 'margin-right': '15px'}),
                ], justify="center"), 
            ], label='Extracellular Analysis', style={'borderBottom':'1px solid #d6d6d6','padding':'6px','fontWeight':'bold'}, selected_style={'borderTop':'1px solid #d6d6d6','borderBottom':'1px solid #d6d6d6','backgroundColor':'#C0D2EA','padding':'6px','fontWeight':'bold'}),
            dcc.Tab([
                html.Br(),
                dbc.Row([   
                    dbc.Col([
                        dbc.Card(
                            html.Div([
                                # check list to choose cell lines     
                                checklist_AP,
                                html.Br(),
                                # switch to choose between single (latest) or multiple files per recording 
                                switch_AP,
                                html.Br(),
                                html.H4("List of processed files"),
                                # table to show data
                                table_AP,
                                dbc.Button("Reset selections", id="reset_button_AP", color="primary", outline=True, size="sm", n_clicks=0),
                                html.Br(),
                                html.Div(id='selected_data_AP',children=[]),      
                            ], style={'margin-left': '10px', 'margin-right': '10px', 'margin-top': '5px', 'margin-bottom': '10px'}), 
                        color="#E4E6D8"),
                    ], style={'margin-left': '10px', 'margin-right': '10px'}),
                ], justify="center"),
                html.Br(), 
                dbc.Row([
                    dbc.Col([
                        dbc.Tabs([                    
                            dbc.Tab([
                                dcc.Graph(id='tab1_graphs_AP'), 
                                dbc.Button("Download figures as PDF", id="download_button_AP", color="primary", outline=True, size="lg", n_clicks=0),
                                html.Span(id="download_output_AP", style={"verticalAlign": "middle"}),
                                ], label="Data distribution", activeTabClassName="fw-bold", tab_id="tab2-1"),
                            dbc.Tab([
                                html.Div(id='feature_table_AP',children=[],style={"overflow": "scroll"}), 
                                ], label="Recording info", activeTabClassName="fw-bold", tab_id="tab2-2"),
                        ], active_tab="tab2-1"), 
                    ], style={'margin-left': '15px', 'margin-right': '15px'}),
                ], justify="center"), 
            ], label='Intracellular Analysis', style={'borderBottom':'1px solid #d6d6d6','padding':'6px','fontWeight':'bold'}, selected_style={'borderTop':'1px solid #d6d6d6','borderBottom':'1px solid #d6d6d6','backgroundColor':'#C0D2EA','padding':'6px','fontWeight':'bold'}),
        ], style={'height': '44px', 'margin-left': '10px', 'margin-right': '10px'}),
    ])

    # open the URL with the default web browser of the user’s computer
    print("Ctrl + C to exit.")
    webbrowser.open_new(f"http://127.0.0.1:{port}/")
    
    app.run(debug=False, port=port)

    
