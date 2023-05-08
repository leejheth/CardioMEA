"""
This is a boilerplate pipeline 'visualize'
generated using Kedro 0.18.7
"""
from dash import Dash, dash_table, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc
# from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import webbrowser
import numpy as np

def dashboard(cardio_db,port):
    print(cardio_db.shape)
    cell_lines = cardio_db["cell_line"].unique()
    col_simple = ["cell_line","compound","time"]

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

    @app.callback(
        Output("checklist_output", "children"),
        [
            Input("checklist", "value"),
        ],
    )
    def on_form_change(checklist_value):
        output_string = f"{len(checklist_value)} checklist items selected. Selected items are {checklist_value}"

        return output_string

    dropdown_options = [
        {"label": "label_op1_1", "value": "111"},
        {"label": "Option 2", "value": "opt2"},
        {"label": "Option 3", "value": "opt3"},
        {"label": "Option 4", "value": "opt4"},
        {"label": "Option 5", "value": "opt5"},
    ]

    @app.callback(Output('dropdown_output', 'children'),
                Input('dropdown', 'value'))
    def update_output(selected_items):
        return html.Ul([html.Li(item) for item in selected_items])
    
    @app.callback(
        Output("dropdown", "options"),
        Input("checklist", "value")
    )
    def update_dropdown(selected_values):
        # create a list of options for the dropdown based on selected values
        df = cardio_db.loc[cardio_db["cell_line"].isin(selected_values), col_simple]
        dropdown_options=[]
        for i in range(len(df)):
            contents = df.iloc[i].values.tolist()
            dropdown_options.append(', '.join([f"{col}: {content}" for col, content in zip(col_simple, contents)]))

        return dropdown_options

    table_header = [
        html.Thead(html.Tr([html.Th(c) for c in col_simple]))
    ]

    row1 = html.Tr([html.Td("Arthur"), html.Td("Dent")])
    row2 = html.Tr([html.Td("Ford"), html.Td("Prefect")])
    row3 = html.Tr([html.Td("Zaphod"), html.Td("Beeblebrox")])
    row4 = html.Tr([html.Td("Trillian"), html.Td("Astra")])

    table_body = [html.Tbody([row1, row2, row3, row4])]


    ###
    # Create sample data
    df2 = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie', 'Dave'],
        'Age': [25, 30, 35, 40],
        'Gender': ['Female', 'Male', 'Male', 'Male']
    })

    # Define table columns and data
    columns = [{'name': i, 'id': i} for i in df2.columns]
    data = df2.to_dict('records')

    # Define data table with checkbox column
    table = dash_table.DataTable(
        id='datatable',
        columns=columns,
        # columns=[
        #     {"name": "Select", "id": "select", "type": "text"},
        #     *columns  # add other columns
        # ],
        data=[
            {c['id']: df2[c['name']].iloc[i] for c in columns} for i in range(len(data))
        ],
        editable=False,
        row_selectable="multi",
        selected_rows=[],
        style_table={'overflowX': 'scroll'},
        style_cell={'textAlign': 'left', 'minWidth': '150px'},
    )

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
                html.Li(f'{row["Name"]} ({row["Age"]})')
                for row in selected_data
            ])
        ])
    ###

    app.layout = html.Div([
        html.H1(children='CardioMEA Dashboard', style={'textAlign':'center'}),
        dbc.Row([
            dbc.Col(
                html.Div(
                    [
                        # check list to choose cell lines    
                        dbc.Form([checklist]),
                        html.P(id="checklist_output"),
                        # dropdown menu to choose files 
                        dcc.Dropdown(
                            id='dropdown',
                            # options=dropdown_options,
                            options=[],
                            multi=True,
                            value=[],
                        ),
                        html.P(id='dropdown_output'),
                    ]
                ), width=4,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H4("List of selected files"),
                        # table to show the selected files
                        # dbc.Table(table_header + table_body, bordered=True),
                        table,
                        html.Div(id='selected_data')
                    ]
                ), width=6,
            ), 
        ]),
    ])



    # open the URL with the default web browser of the user’s computer
    print("Ctrl + C to exit.")
    # webbrowser.open_new(f"http://127.0.0.1:{port}/")

    app.run(debug=True, port=port)

    

