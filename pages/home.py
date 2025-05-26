import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.express as px
from load_data import merged_df

# Random stuff
name_options = [{'label': name, 'value': name} for name in merged_df['common_name'].dropna().unique()]
author_options = [{'label': author, 'value': author} for author in merged_df['author'].dropna().unique()]
collection_options = [{'label': collection, 'value': collection} for collection in merged_df['collection'].dropna().unique()]
rating_options = [{'label': rating, 'value': rating} for rating in merged_df['rating'].dropna().unique()]

unique_classes = merged_df['class_name'].unique()
colors = px.colors.qualitative.Plotly[:len(unique_classes)]  # or use another color palette
color_map = dict(zip(unique_classes, colors))


# Layout
layout = html.Div(
    [
        html.Div(style={'height': '30px'}),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Row([
                            dbc.Col(
                                html.Div([
                                    html.Label('Name:', style={'marginBottom': '5px'}),
                                    dcc.Dropdown(
                                        id='name-selector',
                                        multi=True,
                                        placeholder="Select names",
                                        options=name_options,
                                        value=[],
                                        style={'marginBottom': '20px'}
                                    )
                                ]),
                                width=6,
                                style={'paddingRight': '10px'}
                            ),
                            dbc.Col(
                                html.Div([
                                    html.Label('Author:', style={'marginBottom': '5px'}),
                                    dcc.Dropdown(
                                        id='author-selector',
                                        multi=True,
                                        placeholder="Select authors",
                                        options=author_options,
                                        value=[],
                                        style={'marginBottom': '20px'}
                                    )
                                ]),
                                width=6,
                                style={'paddingLeft': '10px'}
                            ),
                        ], style={'marginBottom': '10px'}),
                        dbc.Row([
                            dbc.Col(
                                html.Div([
                                    html.Label('Collection:', style={'marginBottom': '5px'}),
                                    dcc.Dropdown(
                                        id='collection-selector',
                                        multi=True,
                                        placeholder="Select collections",
                                        options=collection_options,
                                        value=[],
                                        style={'marginBottom': '20px'}
                                    )
                                ]),
                                width=6,
                                style={'paddingRight': '10px'}
                            ),
                            dbc.Col(
                                html.Div([
                                    html.Label('Rating:', style={'marginBottom': '5px'}),
                                    dcc.Dropdown(
                                        id='rating-selector',
                                        multi=True,
                                        placeholder="Select ratings",
                                        options=rating_options,
                                        value=[],
                                        style={'marginBottom': '20px'}
                                    )
                                ]),
                                width=6,
                                style={'paddingLeft': '10px'}
                            ),
                        ]),
                        html.P(
                            id='dataset-info',
                            style={
                                'marginTop': '20px',
                                'fontSize': '14px',
                                'textAlign': 'center',
                                'fontWeight': 'bold'
                            }
                        ),
                    ],
                    width=5,
                    style={"fontFamily": "Arial"}
                ),
                dbc.Col(
                    dcc.Graph(id='bar-chart', figure={}),
                    width=7,
                )
            ], 
            align='center'
        ),
        dbc.Row([
            dbc.Col(
                dcc.Graph(id='geographic-scatter-plot'),
                width=9,
            ),
            dbc.Col(
                dcc.Graph(id='pie-chart', figure={}),
                width=3,
            ),
        ])
    ],
    style={
        "paddingLeft": "50px",
        "paddingRight": "50px",
    }
)


# Callbacks
@callback(
    Output('name-selector', 'options'),    
    Input('author-selector', 'value'),
    Input('collection-selector', 'value'),
    Input('rating-selector', 'value'),
)
def update_name_options(author, collection, rating):
    filtered_df = merged_df.copy()
    
    if author:
        filtered_df = filtered_df[filtered_df['author'].isin(author)]
    
    if collection:
        filtered_df = filtered_df[filtered_df['collection'].isin(collection)]
    
    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]
    
    unique_names = sorted(filtered_df['common_name'].dropna().unique())
    return [{'label': name, 'value': name} for name in unique_names]


@callback(
    Output('author-selector', 'options'),    
    Input('name-selector', 'value'),
    Input('collection-selector', 'value'),
    Input('rating-selector', 'value'),
)
def update_author_options(name, collection, rating):
    filtered_df = merged_df.copy()
      
    if name:  
        filtered_df = filtered_df[filtered_df['common_name'].isin(name)]
    
    if collection:
        filtered_df = filtered_df[filtered_df['collection'].isin(collection)]
    
    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]

    unique_authors = sorted(filtered_df['author'].dropna().unique())
    return [{'label': author, 'value': author} for author in unique_authors]


@callback(
    Output('collection-selector', 'options'),    
    Input('name-selector', 'value'),
    Input('author-selector', 'value'),
    Input('rating-selector', 'value'),
)
def update_collection_options(name, author, rating):
    filtered_df = merged_df.copy()

    if name:
        filtered_df = filtered_df[filtered_df['common_name'].isin(name)]
    
    if author:
        filtered_df = filtered_df[filtered_df['author'].isin(author)]
    
    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]

    unique_collections = sorted(filtered_df['collection'].dropna().unique())
    return [{'label': collection, 'value': collection} for collection in unique_collections]


@callback(
    Output('rating-selector', 'options'),    
    Input('name-selector', 'value'),
    Input('author-selector', 'value'),
    Input('collection-selector', 'value'),
)
def update_rating_options(name, author, collection):
    filtered_df = merged_df.copy()

    if name:
        filtered_df = filtered_df[filtered_df['common_name'].isin(name)]
    
    if author:
        filtered_df = filtered_df[filtered_df['author'].isin(author)]
    
    if collection:
        filtered_df = filtered_df[filtered_df['collection'].isin(collection)]

    unique_ratings = sorted(filtered_df['rating'].dropna().unique())
    return [{'label': rating, 'value': rating} for rating in unique_ratings]


@callback(
    Output('geographic-scatter-plot', 'figure'),
    Input('name-selector', 'value'),
    Input('author-selector', 'value'),
    Input('collection-selector', 'value'),
    Input('rating-selector', 'value')
)
def update_geographic_plot(name, author, collection, rating):    
    filtered_df = merged_df.copy()

    if name:
        filtered_df = filtered_df[filtered_df['common_name'].isin(name)]
    
    if author:
        filtered_df = filtered_df[filtered_df['author'].isin(author)]
    
    if collection:
        filtered_df = filtered_df[filtered_df['collection'].isin(collection)]
    
    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]
    
    filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])
    
    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No geographic data available for selected filters',
                'xaxis': {'title': 'Longitude'},
                'yaxis': {'title': 'Latitude'}
            }
        }
    
    hover_data = ['scientific_name']
    if 'author' in filtered_df.columns:
        hover_data.append('author')
    
    fig = px.scatter_geo(
        filtered_df,
        lon='longitude',
        lat='latitude',
        color='class_name',
        color_discrete_map=color_map,
        hover_data=hover_data,
        title='Geographic Distribution of Animal recordings',
        labels={
            'longitude': 'Longitude',
            'latitude': 'Latitude',
            'class_name': 'Class Name'
        }
    )
    fig.update_geos(
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(243, 243, 243)",
        coastlinecolor="rgb(204, 204, 204)",
        showocean=True,
        oceancolor="rgb(230, 245, 255)",
        showlakes=True,
        lakecolor="rgb(230, 245, 255)",
        showcountries=True,
        countrycolor="rgb(204, 204, 204)"
    )
    fig.update_layout(
        font=dict(family="Arial", size=14),
        title_font=dict(size=20, family="Arial"),
        height=500,
        showlegend=True,
        legend=dict(
            x=0.9,
            y=0.9,
            borderwidth=0,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(size=14)
        ),
        margin=dict(l=50, r=0, t=80, b=50)
    )
        
    return fig


@callback(
    Output('pie-chart', 'figure'),
    Input('name-selector', 'value'),
    Input('author-selector', 'value'),
    Input('collection-selector', 'value'),
    Input('rating-selector', 'value')
)
def update_pie_chart(name, author, collection, rating):
    filtered_df = merged_df.copy()

    if name:
        filtered_df = filtered_df[filtered_df['common_name'].isin(name)]
    
    if author:
        filtered_df = filtered_df[filtered_df['author'].isin(author)]
    
    if collection:
        filtered_df = filtered_df[filtered_df['collection'].isin(collection)]

    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]

    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No data available for selected filters',
            }
        }

    fig = px.pie(
        filtered_df,
        names='class_name',
        color='class_name',
        color_discrete_map=color_map,
        title='Distribution by Class'
    )
    fig.update_layout(
        font=dict(family="Arial", size=14),
        title_font=dict(size=20, family="Arial"),
        margin=dict(t=200),
        legend=dict(
            x=1.0,
            y=1.4,
            borderwidth=0,
            bgcolor='rgba(255,255,255,0.8)',
            font=dict(size=14.2)
        ),
    )

    return fig

@callback(
    Output('bar-chart', 'figure'),
    Input('name-selector', 'value'),
    Input('author-selector', 'value'),
    Input('collection-selector', 'value'),
    Input('rating-selector', 'value')
)
def update_bar_chart(name, author, collection, rating):
    filtered_df = merged_df.copy()

    if name:
        filtered_df = filtered_df[filtered_df['common_name'].isin(name)]
    
    if author:
        filtered_df = filtered_df[filtered_df['author'].isin(author)]
    
    if collection:
        filtered_df = filtered_df[filtered_df['collection'].isin(collection)]

    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]

    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No data available for selected filters',
            }
        }

    value_counts = filtered_df['primary_label'].value_counts().sort_values(ascending=False)
    
    fig = px.bar(
        x=value_counts.index,
        y=value_counts.values,
        title='Distribution by Species'
    )
    fig.update_traces(marker_color="#4d8ded")
    fig.update_xaxes(
        range=[0, 20],
        showgrid=False,
        tickangle=45,
        tickfont=dict(size=13)
    )
    fig.update_yaxes(
        showgrid=False,
        title_standoff=2
    )
    fig.update_layout(
        font=dict(family="Arial", size=14),
        title_font=dict(size=20, family="Arial"),
        xaxis_title='Species',
        yaxis_title='Count',
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=30),
        height=350,
        title_x=0.5
    )

    return fig


@callback(
    Output('dataset-info', 'children'),
    Input('name-selector', 'value'),
    Input('author-selector', 'value'),
    Input('collection-selector', 'value'),
    Input('rating-selector', 'value')
)
def update_dataset_info(name, author, collection, rating):
    filtered_df = merged_df.copy()

    if name:
        filtered_df = filtered_df[filtered_df['common_name'].isin(name)]
    
    if author:
        filtered_df = filtered_df[filtered_df['author'].isin(author)]
    
    if collection:
        filtered_df = filtered_df[filtered_df['collection'].isin(collection)]

    if rating:
        filtered_df = filtered_df[filtered_df['rating'].isin(rating)]

    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No data available for selected filters',
            }
        }

    total_records = len(filtered_df)
    unique_species = filtered_df['primary_label'].nunique()
    
    return f"Total Records: {total_records}, Unique Species: {unique_species}"

dash.register_page("Home", path='/', layout=layout, name="Home")