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
    
  dcc.Input(
      id="song_input",
      type='text',
      placeholder='song name'
  )


    
],style={"marginLeft": "30px", "marginRight": "30px"})



@callback(Output('tab-content', 'children'), Input('tabs', 'value'))
def render_tab(tab):
    if tab == 'tab-recommender':
        return html.Div([
            dbc.Row([
                dbc.Col([
                        dcc.Input(
                            id="song_name",
                            type="text",
                            placeholder="song name",
                            style={"text-center"}),

    
    html.Div(id="output"),
                ])
            ])
        ])
    elif tab == 'tab-eda':
        return html.Div([
            dbc.Col([
                
            ])
        ])

@callback(
    Output("out-all-types", "children"),
    [Input("input_{}".format(_), "value") for _ in ALLOWED_TYPES],
)
def cb_render(*vals):
    return " | ".join((str(val) for val in vals if val))




if __name__== "__main__":
    app.run(debug=True)