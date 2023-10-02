import requests

from dash import Dash, dcc, html, Input, Output, no_update, callback
import plotly.graph_objects as go
import plotly.express as px

from PIL import Image
import numpy as np
import matplotlib.pyplot as pl
from matplotlib.colors import rgb2hex
import pandas as pd

# Helper functions
def px2deg(px):
    '''Convert pixel on screen to visual degrees'''
    px_per_cm = 1920/52. 
    deg = np.rad2deg(2 * np.arctan2(px/px_per_cm, 2*14))
    return deg

data = pd.read_pickle(r'https://github.com/ruditong/MouseAtlas/blob/main/interactive/interactive_data_noims.pkl?raw=True')
regions = ['V1', 'LM', 'LI', 'POR', 'AL', 'RL']
color_base = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
sf_axis =np.fft.fftfreq(64)[:32]
sf_axis = sf_axis[:-1] + np.diff(sf_axis)/2
sf_axis = 1/px2deg(1/sf_axis * 1080*1.6/135)
sf_axis=sf_axis[:30]

url_atlas = r'https://github.com/ruditong/MouseAtlas/blob/main/atlas/'
atlas_resolution = [10, 20, 30, 40, 50, 60, 70, 80]
ims = [Image.open(requests.get(url_atlas+f"image_atlas_{i}.jpeg?raw=True", stream=True).raw) for i in atlas_resolution]
color_picker = lambda x, y, a, b: (rgb2hex(((x-a)/(b-a))*(np.array(pl.cm.tab10(y))-0.5) + np.ones(4)-0.5))
label2num = {region: i for i, region in enumerate(regions)}

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([

    # Dropdown menu and checklist at the Top Left
    html.Div([

        html.Div([
            dcc.Dropdown(
                ['Embedding', 'Atlas'],
                'Embedding',
                id='dropdown1',
            ),
            dcc.Checklist(
                [{"label": html.Div([region], style={'color': color_base[i], 'display': 'inline-block', "font-size": 18, "font-weight": "bold"}), 
                  'value': region} for i, region in enumerate(regions)],
                #regions,
                regions,
                inline=True,
                id='check1',
                inputStyle={"margin-left": "20px", "margin-right": "5px"},
                labelStyle={"align-items": "center"},
            )
        ], style={'width': '49%', 'display': 'inline-block'}),

    ], style={'padding': '10px 5px'}),

    # Scatter map in the left column
    html.Div([
            dcc.Graph(id="umap-scatter", clear_on_unhover=True),
            dcc.Tooltip(id="umap-tooltip", direction='right'),#, background_color='#111111', border_color='#111111'),
            dcc.Slider(0, len(ims)-1,
                       step=None,
                       marks={i: f"{i+1}0x{i+1}0" for i in range(len(ims))},
                       value=4, id='slider-1'),
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),

    # Right column is plotting features
    html.Div([

        html.Div([
            dcc.Graph(id='neighbourhood'),
        ], style={'width': '49%', 'display': 'inline-block'}),
        
        html.Div([
            dcc.Graph(id='table'),
        ], style={'width': '49%', 'float': 'right', 'display': 'inline-block'}),

        html.Div([
            dcc.Graph(id='sf'),
        ], style={'width': '100%',})

    ], style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),

], style={'height':'600px', 'width': '1200px','margin':'0', 'padding':'0'})

@app.callback(
    Output('umap-scatter', 'figure'),
    Input('check1', 'value'),
    Input('dropdown1', 'value'),
    Input('slider-1', 'value'),
)
def update_graph(region, drop, slider):
    if drop == 'Embedding':
        mask = data['Label'].isin(region)
        
        fig = go.Figure(data=[go.Scatter(x=data['x'][mask], y=data['y'][mask], mode='markers', marker=dict(size=5,color=data['Colour'][mask]))])

        fig.update_layout(yaxis={'visible': False, 'showticklabels': False}, xaxis={'visible': False, 'showticklabels': False}, 
                        xaxis_range=[0-0.01, 1+0.01], yaxis_range=[0-0.01, 1+0.01], margin={'l': 0, 'b': 0, 't': 0, 'r': 0}, height=500, width=600,
                        template='plotly_dark')

        fig.update_traces(hoverinfo="none", hovertemplate=None,)

    elif drop == "Atlas":
        fig = go.Figure()
        fig.add_trace(go.Scatter())
        fig.add_layout_image(
        dict(
            source=f'https://raw.githubusercontent.com/ruditong/MouseAtlas/main/atlas/image_atlas_{int(slider)+1}0.jpeg',
            xref="x",
            yref="y",
            x=0,
            y=100,
            sizex=100,
            sizey=100,
            sizing="stretch",))
        #fig = px.imshow(ims[slider], color_continuous_scale='gray', aspect='auto')
        fig.update_layout(yaxis={'visible': False, 'showticklabels': False}, xaxis={'visible': False, 'showticklabels': False}, 
                          margin={'l': 0, 'b': 0, 't': 0, 'r': 0}, height=500, width=600, template='plotly_dark', 
                          xaxis_range=[0, 100], yaxis_range=[0, 100],)

        fig.update_traces(hoverinfo="none", hovertemplate=None)
        
    return fig

@app.callback(
    Output("umap-tooltip", "show"),
    Output("umap-tooltip", "bbox"),
    Output("umap-tooltip", "children"),
    Output("umap-tooltip", "direction"),
    Input("umap-scatter", "hoverData"),
    Input('check1', 'value'),
    Input('dropdown1', 'value'),
)
def display_hover(hoverData, region, drop):
    if hoverData is None:
        return False, no_update, no_update, no_update
    elif drop == 'Atlas':
        return False, no_update, no_update, no_update

    # demo only shows the first point, but other points may also be available
    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    x, y = hover_data["x"], hover_data["y"]
    direction = 'right'
    if x < 0.2: direction = 'right'
    elif x > 0.8: direction = 'left'
    if y < 0.2: direction = 'top'
    elif y > 0.8: direction = 'bottom'

    mask = data['Label'].isin(region)

    # im_matrix = data['Images'][mask].iloc[num]
    # im_url = np_image_to_base64(im_matrix)
    index = data['Label'][mask].index[num]
    im_url = f"https://raw.githubusercontent.com/ruditong/MouseAtlas/main/images/{str(index).zfill(4)}.jpeg"
    color = data['Colour'][mask].iloc[num]
    children = [
        html.Div([
            html.Img(
                src=im_url,
                style={"width": "90px", 'height':'90px', 'display': 'block', 'margin': 'auto auto', 
                       'border': f'5px solid {color}'},
            )
        ], style={'width': '100px', 'height': '100px'})
    ]
    return True, bbox, children, direction


def initialise_graph1():
    fig = px.pie(values=[1/len(regions)]*len(regions), color_discrete_sequence=color_base, names=regions)
    fig.update_layout(margin={'l': 10, 'b': 10, 'r': 10, 't': 10}, height=200, template="plotly_dark")
    fig.update_traces(sort=False, title='Neighbourhood')
    return fig

@app.callback(
    Output('neighbourhood', 'figure'),
    Input("umap-scatter", "hoverData"),
    Input('check1', 'value'),
    Input('dropdown1', 'value'),
)
def update_graph1(hoverData, region, drop):
    if hoverData is None:
        fig = initialise_graph1()
        return fig
    elif drop == 'Atlas':
        fig = initialise_graph1()
        return fig

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    mask = data['Label'].isin(region)

    fig = px.pie(values=data['Neighbourhood'][mask].iloc[num], color_discrete_sequence=color_base, names=regions, template="plotly_dark")
    fig.update_layout(legend={'font':{'size':12}}, margin={'l': 10, 'b': 10, 'r': 10, 't': 10}, height=200)
    fig.update_traces(sort=False, textposition='inside', title='Neighbourhood')
    return fig

def initialise_graph2():
    table_val = dict(values=[['Luminance', 'Folio', 'Quarto'],
                             ["{:.2f}".format(0), 
                              "{:.2f}".format(0), 
                              "{:.2f}".format(0)]],
                     fill_color=['#808080', '#111111'],
                     align=['center', 'center'],
                     line_color='#111111',
                     line=dict(width=2.5),
                     font=dict(color=['white', 'white'], size=18),
                     height=50)
    
    fig = go.Figure(data=[go.Table(header=dict(values=None, fill_color='#111111', line=dict(width=2.5),line_color='#111111'), cells=table_val)])
    fig.update_layout(margin={'l': 10, 'b': 10, 'r': 10, 't': 10}, height=200)
    fig.update_layout(plot_bgcolor='#111111', paper_bgcolor= '#111111')
    return fig

@app.callback(
    Output('table', 'figure'),
    Input("umap-scatter", "hoverData"),
    Input('check1', 'value'),
    Input('dropdown1', 'value'),
)
def update_graph2(hoverData, region, drop):
    if hoverData is None:
        fig = initialise_graph2()
        return fig
    elif drop == 'Atlas':
        fig = initialise_graph2()
        return fig

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    mask = data['Label'].isin(region)

    table_val = dict(values=[['Luminance', 'Folio', 'Quarto'],
                             ["{:.2f}".format(data['Luminance'][mask].iloc[num]), 
                              "{:.2f}".format(data['OSI2'][mask].iloc[num]), 
                              "{:.2f}".format(data['OSI4'][mask].iloc[num])]],
                     fill_color=[data['Colour'][mask].iloc[num], 
                                 [color_picker(float(data['Luminance'][mask].iloc[num]), label2num[data['Label'][mask].iloc[num]], 0.3, 0.7),
                                  color_picker(float(data['OSI2'][mask].iloc[num]), label2num[data['Label'][mask].iloc[num]], 0., 1),
                                  color_picker(float(data['OSI4'][mask].iloc[num]), label2num[data['Label'][mask].iloc[num]], 0., 1)]],
                     align=['center', 'center'],
                     line_color='#111111',
                     line=dict(width=2.5),
                     font=dict(color=['white', 'white'], size=18),
                     height=50)

    fig = go.Figure(data=[go.Table(header=dict(values=None, fill_color='rgba(0,0,0,0)', line=dict(width=2.5),line_color='#111111'), cells=table_val)])
    fig.update_layout(margin={'l': 10, 'b': 10, 'r': 10, 't': 10}, height=200)
    fig.update_layout(plot_bgcolor='#111111', paper_bgcolor= '#111111')
    return fig

def initialise_graph3():
    fig = px.line(x=sf_axis, y=np.zeros(30), markers=True)
    fig.update_layout(margin={'l': 10, 'b': 10, 'r': 10, 't': 10}, height=300, xaxis_range=[0, sf_axis[-1]], yaxis_range=[-2,2],
                      xaxis_title='Spatial Frequency (cyc/deg)', yaxis_title='Power', font=dict(family='Arial', size=16), template="plotly_dark")
    fig.update_traces(line_color='black')
    return fig

@app.callback(
    Output('sf', 'figure'),
    Input("umap-scatter", "hoverData"),
    Input('check1', 'value'),
    Input('dropdown1', 'value'),
)
def update_graph3(hoverData, region, drop):
    if hoverData is None:
        fig = initialise_graph3()
        return fig
    elif drop == 'Atlas':
        fig = initialise_graph3()
        return fig

    hover_data = hoverData["points"][0]
    bbox = hover_data["bbox"]
    num = hover_data["pointNumber"]
    mask = data['Label'].isin(region)

    y = data['Spatial Frequency'][mask].iloc[num]
    fig = px.line(x=sf_axis, y=y, markers=True)
    if y.max() < 2: yrange = [-2,2]
    else: yrange = [-y.max(), y.max()]
    fig.update_layout(margin={'l': 10, 'b': 10, 'r': 10, 't': 10}, height=300, xaxis_range=[0, sf_axis[-1]], yaxis_range=yrange,
                      xaxis_title='Spatial Frequency (cyc/deg)', yaxis_title='Power', font=dict(family='Arial', size=16), template="plotly_dark")
    fig.update_traces(line_color=data['Colour'][mask].iloc[num])
    return fig

if __name__ == '__main__':
    app.run_server()