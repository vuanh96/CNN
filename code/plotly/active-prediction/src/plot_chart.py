import dash_core_components as dcc
import plotly.graph_objs as go
import plotly.plotly as py
import dash_html_components as html
from dateutil import parser
import datetime
from calendar import monthrange
from dateutil import rrule
from datetime import date
import pandas as pd

# return data xaxis and yaxis of chart as period as DAY


def get_chart_data_as_day(affected_filtered_df, unaffected_filtered_df, attr_values, day_column='date'):
    x_axis = [('Day '+str(date)[:10])
              for date in list(affected_filtered_df[(day_column)].unique())]
    affected_y_axises = [affected_filtered_df.groupby(day_column)[attr_value].sum()
                         for attr_value in attr_values]
    unaffected_y_axises = [unaffected_filtered_df.groupby(day_column)[attr_value].sum()
                           for attr_value in attr_values]

    return x_axis, affected_y_axises, unaffected_y_axises

 # return data xaxis and yaxis of chart as period as WEEK


def get_chart_data_as_week(affected_filtered_df, unaffected_filtered_df, attr_values, start_date, duration, day_column='date'):
    x_axis = []
    affected_y_axises = []
    unaffected_y_axises = []
    start_dow = int(parser.parse(start_date).weekday())
    x_axis = [str('Week '+str(i))
              for i in range(int((start_dow + duration) / 7) + 1)]

    for attr_value in attr_values:
        start_date_of_week = start_date
        affected_y_axis = []
        unaffected_y_axis = []
        for num_of_week in x_axis:
            start_dow = int(parser.parse(start_date_of_week).weekday())
            end_date_of_week = str(parser.parse(
                start_date_of_week) + datetime.timedelta(days=6-start_dow))[:10]
            affected_week_data = affected_filtered_df[affected_filtered_df[day_column]
                                                      >= start_date_of_week]
            affected_week_data = affected_week_data[affected_week_data[day_column]
                                                    <= end_date_of_week]
            sum = affected_week_data[attr_value].sum()
            affected_y_axis.append(sum)

            unaffected_week_data = unaffected_filtered_df[unaffected_filtered_df[day_column]
                                                          >= start_date_of_week]
            unaffected_week_data = unaffected_week_data[unaffected_week_data[day_column]
                                                        <= end_date_of_week]
            sum = unaffected_week_data[attr_value].sum()
            unaffected_y_axis.append(sum)

            start_date_of_week = str(parser.parse(
                end_date_of_week) + datetime.timedelta(days=1))[:10]
        affected_y_axises.append(affected_y_axis)
        unaffected_y_axises.append(unaffected_y_axis)
    return x_axis, affected_y_axises, unaffected_y_axises

# return data xaxis and yaxis of chart as period as MONTH


def get_chart_data_as_month(affected_filtered_df, unaffected_filtered_df, attr_values, start_date, end_date, day_column='date'):
    x_axis = []
    affected_y_axises = []
    unaffected_y_axises = []

    # get months x_axis
    x_axis = [str(str(month_year.month)+'/'+str(month_year.year)) for month_year in list(
        rrule.rrule(rrule.MONTHLY, dtstart=parser.parse(start_date), until=parser.parse(end_date)))]

    # get months y_axises
    for attr_value in attr_values:
        start_date_in_month = start_date
        affected_y_axis = []
        unaffected_y_axis = []
        while start_date_in_month < end_date:
            currentdate = parser.parse(start_date_in_month)
            end_date_in_month = str(currentdate.replace(
                day=int(monthrange(int(currentdate.year), int(currentdate.month))[1])))[:10]

            affected_month_data = affected_filtered_df[affected_filtered_df[day_column]
                                                       >= start_date_in_month]
            affected_month_data = affected_month_data[affected_month_data[day_column]
                                                      <= end_date_in_month]
            sum = affected_month_data[attr_value].sum()
            affected_y_axis.append(sum)

            unaffected_month_data = unaffected_filtered_df[unaffected_filtered_df[day_column]
                                                           >= start_date_in_month]
            unaffected_month_data = unaffected_month_data[unaffected_month_data[day_column]
                                                          <= end_date_in_month]
            sum = unaffected_month_data[attr_value].sum()
            unaffected_y_axis.append(sum)

            start_date_in_month = str(parser.parse(
                end_date_in_month) + datetime.timedelta(days=1))[:10]
        unaffected_y_axises.append(unaffected_y_axis)
        affected_y_axises.append(affected_y_axis)

    return x_axis, affected_y_axises, unaffected_y_axises


def plot_bar_chart(x_axis, y_axises, attr_values, title, affected='affected'):
    bar_charts = [go.Bar(
        x=x_axis,
        y=y_axis,
        name=affected + ' '+attr_values[index],
        marker=dict(
            line=dict(
                color='rgb(8,48,107)',
            ),
        ),
    ) for index, y_axis in enumerate(y_axises)]

    return bar_charts


def plot_line_chart(x_axis, y_axises, attr_values, title, dash='solid', affected='affected'):
    line_charts = [go.Scatter(
        x=x_axis,
        y=y_axis,
        name=affected+' '+str(attr_values[index]),
        line=dict(
            # dash='dash',
            dash=dash,
        ),
        # marker=dict(
        # line=dict(
        # color='rgb(8,48,107)',
        # ),
        # ),
        mode='lines+markers',
    ) for index, y_axis in enumerate(y_axises)]

    return line_charts


def plot_chart(df, attr_values, start_date, end_date, affected_users, unaffected_users, chart_type='bar', period='day', title='Sum Of Rev Per User', day_column='date', id_column='id'):
    x_axis = []
    affected_y_axises = []
    unaffected_y_axises = []
    startdatetime = parser.parse(start_date)
    enddatetime = parser.parse(end_date)

    if df.shape[0] == 0:
        # x_axis = [('Day '+str(date)[:10])
        #           for date in list(df[(day_column)].unique())]
        x_axis = ['Day '+str(startdatetime + datetime.timedelta(days=x))[:10] for x in range((enddatetime-startdatetime).days + 1)]
        affected_y_axises = [[0 for i in range(len(x_axis))]]
        unaffected_y_axises = affected_y_axises
        attr_values = ['']

    if df.shape[0] != 0:
        df = df.sort_values(day_column)
        df[[id_column]] = df[[id_column]].astype('str')
        # if end date less than start date, then can not plot chart
        duration = (parser.parse(end_date) - parser.parse(start_date)).days

        # if duration < 0:
        #     return html.H1('Start date must be less than end date')

        # filter data has date between start date and end date
        filtered_df = df[df[day_column] >= start_date]
        filtered_df = filtered_df[filtered_df[day_column] <= end_date]

        # num_rows = filtered_df.shape[0]
        # if num_rows == 0:
        #     return html.H1('The data must be between date '+str(df[day_column].min())[:10]+' and date '+str(df[day_column].max())[:10], style={
        #         'width': '80%',
        #         'textAlign': 'center',
        #     })

        affected_filtered_df = filtered_df[filtered_df[id_column].isin(
            affected_users)]
        unaffected_filtered_df = filtered_df[filtered_df[id_column].isin(
            unaffected_users)]

        if period == 'day':
            x_axis, affected_y_axises, unaffected_y_axises = get_chart_data_as_day(
                affected_filtered_df, unaffected_filtered_df, attr_values, day_column=day_column)

        elif period == 'week':
            x_axis, affected_y_axises, unaffected_y_axises = get_chart_data_as_week(
                affected_filtered_df, unaffected_filtered_df, attr_values, start_date, duration)

        elif period == 'month':
            x_axis, affected_y_axises, unaffected_y_axises = get_chart_data_as_month(
                affected_filtered_df, unaffected_filtered_df, attr_values, start_date, end_date)
        else:
            pass

        # affected_chart = html.Div()
        # unaffected_chart = html.Div()

    if chart_type == 'bar':
        affected_chart = plot_bar_chart(
            x_axis, affected_y_axises, attr_values, title)
        unaffected_chart = plot_bar_chart(
            x_axis, unaffected_y_axises, attr_values, title, affected='unaffected')
    elif chart_type == 'line':
        affected_chart = plot_line_chart(
            x_axis, affected_y_axises, attr_values, title)
        unaffected_chart = plot_line_chart(
            x_axis, unaffected_y_axises, attr_values, title, dash='dash', affected='unaffected')

    graph = dcc.Graph(
        figure=go.Figure(
            data=affected_chart + unaffected_chart, layout=go.Layout(
                title=title,
                showlegend=True,
                legend=go.Legend(
                    x=1.0,
                    y=1.0
                ),
                # margin=go.Margin(l=40, r=40, t=40, b=30),
                barmode='group',
                bargap=0.7,
                bargroupgap=0.2,
                yaxis={'title': 'Total money'},
                xaxis={'title': 'Date time'},
            ),
        ),
        style={'height': 'auto', 'margin': '50px',
               'border': 'solid 1px #878787'},
        id='my-graph'
    )
    return html.Div(children=[graph])
