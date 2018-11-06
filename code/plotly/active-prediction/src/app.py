import dash
import dash_core_components as dcc
import dash_html_components as html
import datetime
from dash.dependencies import Input, Output, State
# from load_file_csv import parse_contents
from plot_chart import *
# from load_file_csv import *
import dash_table_experiments as dte
import pandas as pd
import numpy as np
import base64
import io
import json
from datetime import datetime as dt
import datetime
from read_configure import *
import constant
import dash_auth
from flask import Flask

# init config and add assets
flask_server = Flask(__name__)
external_stylesheets = [
    'https://codepen.io/chriddyp/pen/bWLwgP.assets', 'assets/cdr_style.assets']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, server=flask_server)
auth = dash_auth.BasicAuth(
    app,
    constant.VALID_USERNAME_PASSWORD_PAIRS
)

# configure = get_configure()
# config = html.Div(id='config', children=[
#     html.Div(children=[
#         html.Div(key+': '),
#         dcc.Input(
#             id=configure[key],
#             type='text',
#             value=configure[key]),
#         html.Hr(),
#     ], style={
#         'width': '100%',
#         'margin': 'auto',
#     })
#     for key in configure
# ], style={
#     'width': '100%',
#     'margin': 'auto',
# })

# LEFT PANEL SHOW INFOR ABOUT CDRS
leftNagivation = html.Div(
    children=[html.H1('Configure')], className='leftNagivation')

# settings allow choose period to show in the graph
periods = ['day', 'week', 'month']
chart_types = ['bar', 'line']
periods_component = html.Div([html.Div(html.Span('Period: '), style={
    'float': 'left',
    'margin-top': '30px',
}),
    dcc.Dropdown(
    id='periods_id',
    options=[{'label': i, 'value': i}
             for i in periods],
    value=periods[0],
    className='periods',
    searchable=False,
    clearable=False,
)], className='date_range'
)

# start date
start_date = html.Span(className='date_range', children=[
    'Start date: ', dcc.Input(
        id='start-date-input',
        type='Date',
        value=datetime.date.today() - datetime.timedelta(days=6))]
)

# end date
end_date = html.Span(className='date_range', children=[
    'End date: ', dcc.Input(
        id='end-date-input',
        type='Date',
        value=datetime.date.today())]
    #  - datetime.timedelta(days=60)
)

chart_type_component = html.Div([html.Div(html.Span('Chart type:'), style={
    'float': 'left',
    'margin-top': '30px',
}),
    dcc.Dropdown(
    id='chart_type_id',
    options=[{'label': i, 'value': i}
             for i in chart_types],
    value=chart_types[0],
    searchable=False,
    clearable=False,
    className='periods',
)], className='date_range')


duration_date = html.Div(id='duration_date', children=[
    start_date,
    end_date,
])

settings = html.Div(id='settings', children=[
                    duration_date, html.Div(children=[periods_component, chart_type_component])], className='show')

refresh_data = html.Button(id='refresh_data', children=['Refresh Data'], style={
                           'padding-left': '15px', 'margin': '15px', 'float': 'right', 'background-color': 'red', 'color': 'white'})

header = html.Div(
    children=[html.Div(html.H2('Header'))], className='header')

# listAttrs allow choose attributes user want to show in the graph
listAttrs = html.Div([dcc.Checklist(
    id='list_attrs',
    options=[],
    values=[],
    labelStyle={'display': 'inline-block',
                'margin-left': '20px',
                }
)], style={
    'float': 'left',
    'width': '48%',
})

# upload csv file
upload_component = html.Div([
    dcc.Upload(id='upload-file', children=['Data file: ', html.Button('Upload'), html.Div(className='clear'),
                                           html.Span(id='file_name_uploaded', style={
                                               'color': 'blue',
                                               'margin-left': '50px',
                                               'margin-top': '-10px',
                                           }), ], style={
        'width': '30%',
        'float': 'left',
    },
        multiple=True,
    ),
    dcc.Upload(id='affected-file', children=['Affected users: ', html.Button('Upload'), html.Div(className='clear'),
                                             html.Span(id='file_name_affected_users_uploaded', style={
                                                 'color': 'blue',
                                                 'margin-left': '50px',
                                                 'margin-top': '-10px',
                                             }), ], style={
        'width': '30%',
        'float': 'left',
    },
        multiple=True,
    ),
    dcc.Upload(id='unaffected-file', children=['Unaffected users: ', html.Button('Upload'), html.Div(className='clear'),
                                               html.Span(id='file_name_unaffected_users_uploaded', style={
                                                   'color': 'blue',
                                                   'margin-left': '50px',
                                                   'margin-top': '-10px',
                                               }), ], style={
        'width': '30%',
        'float': 'left',
    },
        multiple=True,
    ),
], className='upload')

# graph area
graph_with_all_component = html.Div(children=[html.H2('Graph',
                                                      style={'padding-left': '15px', 'margin-left': '15px', 'float': 'left'}), refresh_data,
                                              upload_component,
                                              html.Div(
    id='output-file-upload'),
    html.Div(id='hidden_df_from_file', style={
        'display': 'none', }),
    html.Div(id='hidden_affected_users_from_file', style={
        'display': 'none', }),
    html.Div(id='hidden_unaffected_users_from_file', style={
        'display': 'none', }),
    html.Div(id='hidden_attrs', style={
        'display': 'none', }),
    html.Div(children=[html.Hr(
        style={'clear': 'both', }),
        html.Div(children=[listAttrs, settings]),
        html.Div(
        style={'clear': 'both', }),
        html.Div(id='graph_area')]),
], style={
    'margin': 'auto',
})

# right panel contains settings and graph area
rightPanel = html.Div(
    children=[header, graph_with_all_component], className='rightPanel')

# total page
# app.layout = html.Div(children=[leftNagivation, rightPanel])

def serve_layout():
    return html.Div(children=[leftNagivation, rightPanel])

app.layout = serve_layout

def parse_contents_to_df(contents, filename, date, sep='|'):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = None
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep=sep)
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded), sep=sep)

    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    if df is None:
        return None
    return df.to_json()


def read_userids_from_file(file_path):
    ids = []
    with open(file_path, 'r') as file:
        ids = [str(line.strip()) for line in file]
    return ids


@app.callback(Output('hidden_df_from_file', 'children'),
              [Input('upload-file', 'contents'),
               Input('upload-file', 'filename'),
               Input('upload-file', 'last_modified'),
               Input('refresh_data', 'n_clicks')])
def update_dataframe(list_of_contents, list_of_names, list_of_dates, n_clicks):
    # if list_of_names is None:
    #     return None
    # config = get_configure()
    # data_file_path = config[constant.DATA_FILE_PATH]
    # df = pd.read_csv(data_file_path, sep=config[constant.SPLIT_CSV])
    # if df is not None:
    #     return df.to_json()
    if list_of_contents is not None:
        children = [
            parse_contents_to_df(c, n, d, sep=',') for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        if len(children) > 0:
            return children[0]
        else:
            return None
    return None


def parse_contents_to_list_users(contents, filename, date, sep='|'):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    str_list = decoded.decode('utf-8')
    users_list = [user.strip() for user in str_list.split('\n')]
    return json.dumps(users_list)


@app.callback(Output('hidden_affected_users_from_file', 'children'),
              [Input('affected-file', 'contents'),
               Input('affected-file', 'filename'),
               Input('affected-file', 'last_modified'),
               Input('refresh_data', 'n_clicks')])
def update_affected_users(list_of_contents, list_of_names, list_of_dates, n_clicks):
    # if list_of_names is None:
    #     return None
    # config = get_configure()
    # affected_users = read_userids_from_file(
    #     config[constant.AFFECTED_USERS_FILE_PATH])
    # if affected_users is not None:
    #     return json.dumps(affected_users)
    if list_of_contents is not None:
        children = [
            parse_contents_to_list_users(c, n, d, sep=',') for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        if len(children) > 0:
            return children[0]
        else:
            return None
    return None


@app.callback(Output('hidden_unaffected_users_from_file', 'children'),
              [Input('unaffected-file', 'contents'),
               Input('unaffected-file', 'filename'),
               Input('unaffected-file', 'last_modified'),
               Input('refresh_data', 'n_clicks')])
def update_unaffected_users(list_of_contents, list_of_names, list_of_dates, n_clicks):
    # if list_of_names is None:
    #     return None
    # config = get_configure()
    # unaffected_users = read_userids_from_file(
    #     config[constant.UNAFFECTED_USERS_FILE_PATH])
    # if unaffected_users is not None:
    #     return json.dumps(unaffected_users)
    if list_of_contents is not None:
        children = [
            parse_contents_to_list_users(c, n, d, sep=',') for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
        if len(children) > 0:
            return children[0]
        else:
            return None
    return None


@app.callback(Output('graph_area', 'children'),
              [Input('hidden_df_from_file', 'children'),
               Input('hidden_affected_users_from_file', 'children'),
               Input('hidden_unaffected_users_from_file', 'children'),
               Input('list_attrs', 'values'),
               Input('start-date-input', 'value'),
               Input('end-date-input', 'value'),
               Input('chart_type_id', 'value'),
               Input('periods_id', 'value'),
               ])
def update_graph(json_df, affected_users, unaffected_users, attr_values, start_date, end_date, chart_type, period):
    if json_df is None:
        dff = pd.DataFrame()
        # return html.Div('The format must be csv file with columns seperate by "," character', style={
        #     'margin': 'auto',
        #     'text-align': 'center',
        # })
    else:
        dff = pd.read_json(json_df)
    if attr_values is None:
        attr_values = []
    if affected_users is not None:
        affected_users = json.loads(affected_users)
    else:
        affected_users = []
    if unaffected_users is not None:
        unaffected_users = json.loads(unaffected_users)
    else:
        unaffected_users = []
    chart = plot_chart(dff, attr_values,
                       start_date[:10], end_date[:10], affected_users, unaffected_users, chart_type, period)

    return html.Div([
        chart,
    ], style={
        'margin-top': '50px',
        'margin': 'auto',
        'textAlign': 'center',
        'font-weight': 'bold',
        'font-style': 'italic',
    })


@app.callback(Output('file_name_uploaded', 'children'),
              [Input('upload-file', 'filename')])
def update_filename_upload(filenames):
    return html.Div(children=filenames)


@app.callback(Output('list_attrs', 'options'),
              [Input('hidden_df_from_file', 'children')])
def update_attrs(json_df):
    if json_df is None:
        return []

    dff = pd.read_json(json_df)
    columns = dff.columns
    column_types = dff.dtypes
    numeric_columns = [column for (column, column_type) in zip(
        columns, column_types) if column_type == 'float64' or column_type == 'int64']
    return [{'label': attr, 'value': attr} for attr in numeric_columns]


@app.callback(Output('list_attrs', 'values'),
              [Input('hidden_df_from_file', 'children')])
def update_value_attrs(json_df):
    if json_df is None:
        return []

    dff = pd.read_json(json_df)
    columns = dff.columns
    column_types = dff.dtypes
    numeric_columns = [column for (column, column_type) in zip(
        columns, column_types) if column_type == 'float64' or column_type == 'int64']
    if len(numeric_columns) > 1:
        return [numeric_columns[1]]
    return []


# @app.callback(Output('settings', 'className'),
#               [Input('hidden_df_from_file', 'children')])
# def show_date_range(json_df):
#     if json_df is None:
#         return 'hidden'
#     return 'show'


@app.callback(Output('file_name_affected_users_uploaded', 'children'),
              [Input('affected-file', 'filename')])
def update_affected_file_name(filename):
    return html.Div(children=filename)


@app.callback(Output('file_name_unaffected_users_uploaded', 'children'),
              [Input('unaffected-file', 'filename')])
def update_unaffected_file_name(filename):
    return html.Div(children=filename)


if __name__ == '__main__':
    app.run_server(debug=True)
