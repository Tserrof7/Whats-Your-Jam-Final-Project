import pandas as pd
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
ALLOWED_TYPES = (["track_name"])
app.title = "Music Recommender"


app.layout = html.Div([
    html.H1("Music Recommender and EDA", className="text-center my-4"),

    dcc.Tabs(id="tabs", value='tab-info', children=[
        dcc.Tab(label='Music Recommender', value='tab-recommender'),
        dcc.Tab(label='Spotify Dataset EDA', value='tab-eda')]),      
       
    html.Div(id='tab-content'),
    
],style={"margin-left": "30px", "margin-right": "30px"})



@callback(Output('tab-content', 'children'),Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-recommender':
        return html.Div([
            dbc.Row([
                dbc.Col([
                        dbc.Input(
                            id="song_name",
                            type="text",
                            placeholder="song name",
                            style={}),

    
    # html.Div(id="output"),
                ]),
            ])
        ])
    elif tab == 'tab-eda':
        return html.Div([
            dbc.Row([
                dbc.Col([
                    dbc.Carousel(
                        items=[
                            {"key":"1","src":"assets/Boston-cream-donut.png", "caption":"This is a boston cream donut", "img_style":{"max-height":"500px"}},
                             {"key":"2","src":"assets/assets/f8b76d1883e09d74.png",
                             "header":"minecraft skin","caption":"This is my minecraft skin","img_style":{"max_height":"500px"} }
                        ]
                    )
                 ])
            ])
        ])

# @callback(
#     Output("out-all-types", "children"),
#     [Input("input_{}".format(_), "value") for _ in ALLOWED_TYPES],
# )
# def cb_render(*vals):
#     return " | ".join((str(val) for val in vals if val))




if __name__== "__main__":
    app.run(debug=True)