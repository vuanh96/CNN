import dash
from dash.dependencies import Output, Input, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import pandas as pd
from recommendation import CF
from datetime import datetime
import copy
import plotly.graph_objs as go

app = dash.Dash()

# stylesheets = [
#     'https://cdn.rawgit.com/plotly/dash-app-stylesheets/2d266c578d2a6e8850ebce48fdb52759b2aef506/stylesheet-oil-and-gas.assets']
#
# for stylesheet in stylesheets:
#     app.css.append_css({"external_url": stylesheet})

# Loading data from file csv
users = pd.read_csv('ml-100k/u.user.zipcode', header=0)
columns_used = ['USER ID', 'AGE', 'GENDER', 'OCCUPATION', 'ZIP CODE', 'POST OFFICE CITY']

genres = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama',
          'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance ', 'Sci-Fi', 'Thriller', 'War', 'Western']


def extract_genres(row):
    list_genre = ""
    for g in genres:
        if row[g] == 1:
            list_genre += g + ", "
    return list_genre[:-2]


items = pd.read_csv('ml-100k/u.item', sep='|', header=None,
                    names=['MOVIE ID', 'MOVIE TITLE', 'RELEASE DATE', 'IDMb URL'] + genres)
items['GENRES'] = items.apply(lambda row: extract_genres(row), axis=1)

ratings = pd.read_csv('ml-100k/ub.base', sep='\t', header=None, names=['USER ID', 'MOVIE ID', 'RATING', 'TIME STAMP'])

# Training and recommend
train = ratings[['USER ID', 'MOVIE ID', 'RATING']].values
rs = CF(train, k=30, uuCF=1)
rs.fit()

mapbox_access_token = 'pk.eyJ1IjoiamFja2x1byIsImEiOiJjajNlcnh3MzEwMHZtMzNueGw3NWw5ZXF5In0.fk8k06T96Ml9CLGgKmk81w'
main_layout = dict(
    autosize=True,
    # height=500,
    margin=dict(
        l=30,
        r=30,
        b=0,
        t=30
    ),
    hovermode="closest",
    legend=dict(font=dict(size=10), orientation='h'),
    mapbox=dict(
        accesstoken=mapbox_access_token,
        # center=dict(
        #     lon=-78.05,
        #     lat=42.54
        # ),
        # zoom=3,
    )
)

# Bulding layout for html
map_graph_layout = copy.deepcopy(main_layout)
map_graph_layout.update({'title': 'Users Map'})
map_graph = dcc.Graph(id='map', figure=dict(
    data=[
        dict(
            type='scattermapbox',
            lat=users['LAT'],
            lon=users['LON'],
            text=users['POST OFFICE CITY'],
            marker=dict(
                color=users['AGE'],
                size=8,
                opacity=0.6
            ),
            customdata=users['ZIP CODE']
        ),
    ],
    layout=map_graph_layout
))

user_profile_table = dt.DataTable(
    rows=users[columns_used].to_dict("record"),
    columns=columns_used,
    row_selectable=True,
    filterable=True,
    sortable=True,
    editable=False,
    selected_row_indices=[0],
    id='table-user-profile'
)

movie_recommend_table = dt.DataTable(
    rows=[],
    columns=['MOVIE ID', 'MOVIE TITLE', 'GENRES'],
    row_selectable=False,
    filterable=True,
    sortable=True,
    editable=False,
    selected_row_indices=[],
    id='table-movie-recommend'
)

app.layout = html.Div([
    html.Div([
        html.H1('Movie Recommendation System'),
    ], className='row'),
    html.Div([
        map_graph
    ], className='row'),
    html.Div([
        html.H4('The distribution of users: ', style={'display': 'inline'}),
        html.Div(id='n_users_city', style={'display': 'inline'})
    ], className='row'),
    html.Div([
        html.H3('Users Profile and Top 10 Movie Recommend For User'),
    ], className='row'),
    html.Div([
        html.Div([
            user_profile_table
        ], className='seven columns'),
        html.Div([
            movie_recommend_table
        ], className='five columns')
    ], className='row'),
    html.Div([
        html.Div([
            dcc.Graph(id='line-chart-user-rating')
        ], className='four columns'),
        html.Div([
            dcc.Graph(id='pie-chart-user-rating')
        ], className='eight columns')
    ], className='row')
], className='ten columns offset-by-one')


@app.callback(
    Output('n_users_city', 'children'),
    [Input('map', 'hoverData')])
def update_text(hoverData):
    try:
        s = users[users['ZIP CODE'] == hoverData['points'][0]['customdata']]
        return html.H4(
            "{} : {} users".format(s.iat[0, 7], s.shape[0]),
            style={'color': 'red', 'display': 'inline'}
        )
    except:
        return None


@app.callback(
    Output('table-movie-recommend', 'rows'),
    [Input('table-user-profile', 'selected_row_indices')])
def update_table_recommend(selected_row_indices):
    try:
        idx = selected_row_indices[0]
        user_id = users.loc[idx, 'USER ID']
        id_items_recommend = rs.recommend(user_id)[:10]
        items_recommend = items[items['MOVIE ID'].isin(id_items_recommend)]

        return items_recommend.to_dict("record")
    except:
        return []


@app.callback(
    Output('line-chart-user-rating', 'figure'),
    [Input('table-user-profile', 'selected_row_indices')])
def update_line_chart_user_rating(selected_row_indices):
    def group_ratings_by_month(df, idx, col):
        time = datetime.fromtimestamp(df.loc[idx][col])
        return time.year, time.month

    idx = selected_row_indices[0]
    user_id = users.loc[idx, 'USER ID']
    ratings_by_user = ratings[ratings['USER ID'] == user_id]
    ratings_by_month = \
        ratings_by_user.groupby(lambda idx: group_ratings_by_month(ratings_by_user, idx, 'TIME STAMP'))['RATING'] \
            .count().reset_index()

    data = [
        dict(
            type='scatter',
            mode='lines+markers',
            name='User ' + str(user_id),
            x=["{}-{}".format(y, m) for y, m in ratings_by_month['index']],
            y=ratings_by_month['RATING'],
            line=dict(
                shape="spline",
                smoothing="2",
            )
        ),
        dict(
            type='bar',
            name='User ' + str(user_id),
            x=["{}-{}".format(y, m) for y, m in ratings_by_month['index']],
            y=ratings_by_month['RATING'] * 2,
        )
    ]
    layout = copy.deepcopy(main_layout)
    layout['title'] = "Ratings History Of User " + str(user_id)
    figure = dict(data=data, layout=layout)
    return figure


@app.callback(
    Output('pie-chart-user-rating', 'figure'),
    [Input('table-user-profile', 'selected_row_indices')])
def update_pie_chart_user_rating(selected_row_indices):
    idx = selected_row_indices[0]
    user_id = users.loc[idx]['USER ID']
    ratings_by_user = ratings[ratings['USER ID'] == user_id]
    cnt_star = ratings_by_user.groupby('RATING')['USER ID'].count().reset_index()

    cnt_genres = items[items['MOVIE ID'].isin(ratings_by_user['MOVIE ID'])][genres].sum()
    data = [
        dict(
            type='pie',
            values=cnt_star['USER ID'],
            labels=["{} ★".format(i) for i in cnt_star['RATING']],
            name='Rating Of User',
            hoverinfo='label+percent',
            hole=.5,
            domain={"x": [0, .48]},
        ),
        dict(
            type='pie',
            values=cnt_genres,
            labels=cnt_genres.index,
            name='Rating Of User',
            hoverinfo='label+percent',
            hole=.5,
            domain={"x": [.52, 1]},

        )
    ]
    layout = copy.deepcopy(main_layout)
    layout.update(
        dict(
            title="Percentage Of Ratings By User " + str(user_id),
            annotations=[
                dict(
                    showarrow=False,
                    text="Star",
                    font=dict(
                        size=20
                    ),
                    x=.2,
                    y=.5
                ),
                dict(
                    showarrow=False,
                    text="Genres",
                    font=dict(
                        size=20
                    ),
                    x=.8,
                    y=.5
                )
            ]
        )
    )
    fig = dict(data=data, layout=layout)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
