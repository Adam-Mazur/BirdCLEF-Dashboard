import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd

# Load and merge data
try:
    # Load the CSV files
    train_df = pd.read_csv('train.csv')
    taxonomy_df = pd.read_csv('taxonomy.csv')
    
    # Merge train_df with taxonomy_df based on 'scientific_name'
    # This adds the 'class_name' column to each row in train_df
    merged_df = train_df.merge(taxonomy_df[['scientific_name', 'class_name']], 
                              on='scientific_name', 
                              how='left')
    
    print(f"Successfully loaded data:")
    print(f"Train dataset shape: {train_df.shape}")
    print(f"Taxonomy dataset shape: {taxonomy_df.shape}")
    print(f"Merged dataset shape: {merged_df.shape}")
    print(f"Columns in merged dataset: {list(merged_df.columns)}")
    
    # Check for any unmatched scientific names
    unmatched = merged_df[merged_df['class_name'].isna()]
    if not unmatched.empty:
        print(f"Warning: {len(unmatched)} rows have no matching class_name")
    
except FileNotFoundError as e:
    print(f"Error loading CSV files: {e}")
    merged_df = pd.DataFrame()  # Create empty dataframe as fallback
except Exception as e:
    print(f"Error processing data: {e}")
    merged_df = pd.DataFrame()

# Initialize the Dash app with suppress_callback_exceptions=True
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Define the app layout
app.layout = html.Div([
    html.H1("Dashboard", 
            style={
                'textAlign': 'center',
                'marginBottom': '30px',
                'color': '#2c3e50',
                'fontFamily': 'Arial, sans-serif'
            }),
        
    dcc.Tabs(id="tabs", value='tab-1', children=[
        dcc.Tab(label='Tab 1', value='tab-1'),
        dcc.Tab(label='Tab 2', value='tab-2'),
        dcc.Tab(label='Tab 3', value='tab-3'),
        dcc.Tab(label='Tab 4', value='tab-4'),
    ], style={
        'height': '44px',
        'borderBottom': '1px solid #d6d6d6',
        'fontFamily': 'Arial, sans-serif'
    }),
        
    html.Div(id='tab-content', style={
        'padding': '20px',
        'minHeight': '400px'
    })
])

# Callback to render tab content
@app.callback(Output('tab-content', 'children'),
              Input('tabs', 'value'))
def render_content(tab):
    if tab == 'tab-1':
        if merged_df.empty:
            return html.Div([
                html.H3('Error: No Data Available', style={'color': '#e74c3c'}),
                html.P('Could not load CSV files. Please check that train.csv and taxonomy.csv exist in the current directory.',
                       style={'fontSize': '16px', 'color': '#7f8c8d'})
            ])
        
        return html.Div([
            # Geographic Distribution and Species Search side by side
            html.Div([
                # Left side - Geographic Distribution (65% width)
                html.Div([
                    html.H2('Geographic Distribution', style={'color': '#34495e', 'marginBottom': '20px', 'textAlign': 'center', 'fontSize': '28px'}),
                    
                    html.Div([
                        html.Div([
                            html.Label('Classes for map:', style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='map-class-selector',
                                multi=True,
                                placeholder="Select classes for map (leave empty to include all)",
                                style={'marginBottom': '20px'}
                            )
                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label('Author:', style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='author-selector',
                                multi=True,
                                placeholder="Select authors (leave empty to include all)",
                                style={'marginBottom': '20px'}
                            )
                        ], style={'width': '32%', 'display': 'inline-block', 'marginRight': '2%'}),
                        
                        html.Div([
                            html.Label('Search for specific scientific name:', style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                            dcc.Dropdown(
                                id='scientific-name-search',
                                multi=True,
                                placeholder="Type to search for scientific names (leave empty to show all)",
                                style={'marginBottom': '20px'}
                            )
                        ], style={'width': '32%', 'display': 'inline-block'})
                    ]),
                    
                    dcc.Graph(id='geographic-scatter-plot')
                ], style={
                    'width': '65%', 
                    'display': 'inline-block', 
                    'verticalAlign': 'top',
                    'paddingRight': '20px'
                }),
                
                # Right side - Species Search (35% width)
                html.Div([
                    html.H2('Species Search', style={
                        'color': '#34495e', 
                        'textAlign': 'center', 
                        'fontSize': '24px',
                        'marginBottom': '20px'
                    }),
                    
                    html.Div([
                        html.Label('Filter by class first (optional):', style={
                            'fontWeight': 'bold', 
                            'marginBottom': '10px', 
                            'display': 'block',
                            'fontSize': '14px'
                        }),
                        dcc.Dropdown(
                            id='species-class-filter',
                            placeholder="Select class to filter species (optional)",
                            style={'marginBottom': '15px'}
                        ),
                        
                        html.Label('Select a species:', style={
                            'fontWeight': 'bold', 
                            'marginBottom': '10px', 
                            'display': 'block',
                            'fontSize': '14px'
                        }),
                        dcc.Dropdown(
                            id='species-selector',
                            placeholder="Type to search...",
                            style={'marginBottom': '15px'}
                        ),
                        
                        html.Button(
                            'View External Link',
                            id='external-link-button',
                            n_clicks=0,
                            disabled=True,
                            style={
                                'backgroundColor': '#4CAF50',
                                'color': 'white',
                                'padding': '10px 20px',
                                'fontSize': '14px',
                                'border': 'none',
                                'borderRadius': '4px',
                                'cursor': 'pointer',
                                'width': '100%',
                                'marginBottom': '15px'
                            }
                        )
                    ]),
                    
                    html.Div(id='species-info-display', style={
                        'padding': '15px',
                        'backgroundColor': '#f8f9fa',
                        'borderRadius': '6px',
                        'border': '1px solid #dee2e6',
                        'minHeight': '150px',
                        'fontSize': '13px'
                    })
                ], style={
                    'width': '35%', 
                    'display': 'inline-block', 
                    'verticalAlign': 'top',
                    'backgroundColor': '#ffffff',
                    'padding': '15px',
                    'borderRadius': '8px',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.1)',
                    'marginLeft': '0px'
                })
            ], style={
                'marginBottom': '40px',
                'display': 'flex',
                'flexDirection': 'row',
                'alignItems': 'flex-start',
                'gap': '0px'
            }),
            
            html.Hr(style={'margin': '20px 0'}),
            
            html.H2('Scientific Names Distribution', style={'color': '#34495e', 'textAlign': 'center', 'fontSize': '28px'}),
            
            html.Div([
                html.Div([
                    html.Label('Number of names to show:', style={'fontWeight': 'bold', 'marginRight': '10px'}),
                    dcc.Slider(
                        id='num-names-slider',
                        min=5,
                        max=50,
                        step=5,
                        value=10,
                        marks={i: str(i) for i in range(5, 55, 10)},
                        tooltip={"placement": "bottom", "always_visible": True}
                    )
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div([
                        html.Label('Show:', style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.RadioItems(
                            id='frequency-type',
                            options=[
                                {'label': 'Most Common', 'value': 'most'},
                                {'label': 'Rarest', 'value': 'rarest'}
                            ],
                            value='most',
                            inline=True
                        )
                    ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '4%', 'verticalAlign': 'top'}),
                    
                    html.Div([
                        html.Label('Classes to include:', style={'fontWeight': 'bold', 'marginBottom': '10px', 'display': 'block'}),
                        dcc.Dropdown(
                            id='class-selector',
                            multi=True,
                            placeholder="Select classes (leave empty to include all)",
                            style={'marginBottom': '10px'}
                        )
                    ], style={'width': '48%', 'display': 'inline-block'})
                ], style={'marginBottom': '20px'})
            ]),
            
            dcc.Graph(id='scientific-names-plot')
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H3('Tab 2 Content', style={'color': '#34495e'}),
            html.P('This is the content for Tab 2. Add your components here.',
                   style={'fontSize': '16px', 'color': '#7f8c8d'})
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H3('Tab 3 Content', style={'color': '#34495e'}),
            html.P('This is the content for Tab 3. Add your components here.',
                   style={'fontSize': '16px', 'color': '#7f8c8d'})
        ])
    elif tab == 'tab-4':
        return html.Div([
            html.H3('Tab 4 Content', style={'color': '#34495e'}),
            html.P('This is the content for Tab 4. Add your components here.',
                   style={'fontSize': '16px', 'color': '#7f8c8d'})
        ])

# Callback to populate class selector dropdown
@app.callback(
    Output('class-selector', 'options'),
    Input('tabs', 'value')
)
def update_class_options(tab):
    if tab == 'tab-1' and not merged_df.empty:
        unique_classes = sorted(merged_df['class_name'].dropna().unique())
        return [{'label': cls, 'value': cls} for cls in unique_classes]
    return []

# Callback for the interactive plot
@app.callback(
    Output('scientific-names-plot', 'figure'),
    [Input('num-names-slider', 'value'),
     Input('frequency-type', 'value'),
     Input('class-selector', 'value')]
)
def update_plot(num_names, frequency_type, selected_classes):
    if merged_df.empty:
        return {}
    
    # Filter by selected classes if any are chosen
    filtered_df = merged_df.copy()
    if selected_classes:
        filtered_df = filtered_df[filtered_df['class_name'].isin(selected_classes)]
    
    # If no data remains after filtering, return empty plot
    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No data available for selected classes',
                'xaxis': {'title': 'Scientific Name'},
                'yaxis': {'title': 'Count'}
            }
        }
    
    # Count scientific names and get their class info
    name_counts = filtered_df['scientific_name'].value_counts()
    
    # Get the class name for each scientific name (take the first occurrence)
    name_to_class = filtered_df.drop_duplicates('scientific_name').set_index('scientific_name')['class_name']
    
    # Sort based on frequency type
    if frequency_type == 'most':
        selected_names = name_counts.head(num_names)
        title = f'Top {num_names} Most Common Scientific Names'
    else:
        selected_names = name_counts.tail(num_names)
        title = f'Top {num_names} Rarest Scientific Names'
    
    # Add class filter info to title
    if selected_classes:
        class_text = ', '.join(selected_classes) if len(selected_classes) <= 3 else f'{len(selected_classes)} classes'
        title += f' (Classes: {class_text})'
    
    # Create dataframe for plotting
    plot_data = pd.DataFrame({
        'scientific_name': selected_names.index,
        'count': selected_names.values,
        'class_name': [name_to_class[name] for name in selected_names.index]
    })
    
    # Sort the data based on count for proper ordering
    if frequency_type == 'most':
        plot_data = plot_data.sort_values('count', ascending=False)
    else:
        plot_data = plot_data.sort_values('count', ascending=True)
    
    # Create the bar plot with colors based on class_name
    fig = px.bar(
        plot_data,
        x='scientific_name',
        y='count',
        color='class_name',
        title=title,
        labels={
            'scientific_name': 'Scientific Name',
            'count': 'Count',
            'class_name': 'Class Name'
        }
    )
    
    # Ensure the x-axis follows the sorted order
    fig.update_xaxes(categoryorder='array', categoryarray=plot_data['scientific_name'].tolist())
    
    fig.update_layout(
        xaxis_tickangle=-45,
        height=600,
        margin=dict(b=150, l=50, r=50, t=80),
        showlegend=True
    )
    
    return fig

# Callback to populate map class selector dropdown
@app.callback(
    Output('map-class-selector', 'options'),
    Input('tabs', 'value')
)
def update_map_class_options(tab):
    if tab == 'tab-1' and not merged_df.empty:
        unique_classes = sorted(merged_df['class_name'].dropna().unique())
        return [{'label': cls, 'value': cls} for cls in unique_classes]
    return []

# Callback to populate author selector dropdown
@app.callback(
    Output('author-selector', 'options'),
    Input('tabs', 'value')
)
def update_author_options(tab):
    if tab == 'tab-1' and not merged_df.empty:
        if 'author' in merged_df.columns:
            unique_authors = sorted(merged_df['author'].dropna().unique())
            return [{'label': author, 'value': author} for author in unique_authors]
    return []

# Callback to populate scientific name search dropdown
@app.callback(
    Output('scientific-name-search', 'options'),
    [Input('tabs', 'value'),
     Input('map-class-selector', 'value'),
     Input('author-selector', 'value')]
)
def update_scientific_name_options(tab, selected_classes, selected_authors):
    if tab == 'tab-1' and not merged_df.empty:
        # Filter by selected classes if any are chosen
        filtered_df = merged_df.copy()
        if selected_classes:
            filtered_df = filtered_df[filtered_df['class_name'].isin(selected_classes)]
        
        # Filter by selected authors if any are chosen
        if selected_authors and 'author' in merged_df.columns:
            filtered_df = filtered_df[filtered_df['author'].isin(selected_authors)]
        
        # Get unique scientific names and sort them
        unique_names = sorted(filtered_df['scientific_name'].dropna().unique())
        return [{'label': name, 'value': name} for name in unique_names]
    return []

# Callback for the geographic scatter plot
@app.callback(
    Output('geographic-scatter-plot', 'figure'),
    [Input('map-class-selector', 'value'),
     Input('author-selector', 'value'),
     Input('scientific-name-search', 'value')]
)
def update_geographic_plot(selected_classes, selected_authors, selected_scientific_names):
    if merged_df.empty:
        return {}
    
    # Check if latitude and longitude columns exist
    if 'latitude' not in merged_df.columns or 'longitude' not in merged_df.columns:
        return {
            'data': [],
            'layout': {
                'title': 'Geographic data not available (latitude/longitude columns missing)',
                'xaxis': {'title': 'Longitude'},
                'yaxis': {'title': 'Latitude'}
            }
        }
    
    # Filter by selected classes if any are chosen
    filtered_df = merged_df.copy()
    if selected_classes:
        filtered_df = filtered_df[filtered_df['class_name'].isin(selected_classes)]
    
    # Filter by selected authors if any are chosen
    if selected_authors and 'author' in merged_df.columns:
        filtered_df = filtered_df[filtered_df['author'].isin(selected_authors)]
    
    # Filter by selected scientific names if any are chosen
    if selected_scientific_names:
        if isinstance(selected_scientific_names, str):
            selected_scientific_names = [selected_scientific_names]
        filtered_df = filtered_df[filtered_df['scientific_name'].isin(selected_scientific_names)]
    
    # Remove rows with missing latitude/longitude data
    filtered_df = filtered_df.dropna(subset=['latitude', 'longitude'])
    
    # If no data remains after filtering, return empty plot
    if filtered_df.empty:
        return {
            'data': [],
            'layout': {
                'title': 'No geographic data available for selected filters',
                'xaxis': {'title': 'Longitude'},
                'yaxis': {'title': 'Latitude'}
            }
        }
    
    # Create hover data list based on available columns
    hover_data = ['scientific_name']
    if 'author' in filtered_df.columns:
        hover_data.append('author')
    
    # Create the scatter plot
    fig = px.scatter_geo(
        filtered_df,
        lon='longitude',
        lat='latitude',
        color='class_name',
        hover_data=hover_data,
        title='Geographic Distribution of Scientific Names',
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
        height=500,
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Add filter info to title
    title_parts = []
    if selected_classes:
        class_text = ', '.join(selected_classes) if len(selected_classes) <= 3 else f'{len(selected_classes)} classes'
        title_parts.append(f'Classes: {class_text}')
    
    if selected_authors:
        author_text = ', '.join(selected_authors) if len(selected_authors) <= 2 else f'{len(selected_authors)} authors'
        title_parts.append(f'Authors: {author_text}')
    
    if selected_scientific_names:
        name_text = ', '.join(selected_scientific_names) if len(selected_scientific_names) <= 2 else f'{len(selected_scientific_names)} names'
        title_parts.append(f'Names: {name_text}')
    
    if title_parts:
        fig.update_layout(title=f'Geographic Distribution ({"; ".join(title_parts)})')
    
    return fig

# NEW CALLBACKS FOR MULTI-CLASS SPECIES SEARCH

# Callback to populate species class filter dropdown
@app.callback(
    Output('species-class-filter', 'options'),
    Input('tabs', 'value')
)
def update_species_class_filter_options(tab):
    if tab == 'tab-1' and not merged_df.empty:
        unique_classes = sorted(merged_df['class_name'].dropna().unique())
        return [{'label': cls, 'value': cls} for cls in unique_classes]
    return []

# Callback to populate species selector dropdown based on class filter
@app.callback(
    Output('species-selector', 'options'),
    [Input('tabs', 'value'),
     Input('species-class-filter', 'value')]
)
def update_species_options(tab, selected_class):
    if tab == 'tab-1' and not merged_df.empty:
        # Filter by selected class if one is chosen
        filtered_df = merged_df.copy()
        if selected_class:
            filtered_df = filtered_df[filtered_df['class_name'] == selected_class]
        
        if not filtered_df.empty:
            # Get unique scientific names
            unique_names = sorted(filtered_df['scientific_name'].dropna().unique())
            return [{'label': name, 'value': name} for name in unique_names]
    return []

# Callback to clear species selection when class filter changes
@app.callback(
    Output('species-selector', 'value'),
    Input('species-class-filter', 'value')
)
def clear_species_selection(selected_class):
    return None

# Callback to update species info display and button state
@app.callback(
    [Output('species-info-display', 'children'),
     Output('external-link-button', 'disabled'),
     Output('external-link-button', 'style'),
     Output('external-link-button', 'children')],
    Input('species-selector', 'value')
)
def update_species_info(selected_species):
    default_button_style = {
        'backgroundColor': '#6c757d',
        'color': 'white',
        'padding': '12px 24px',
        'fontSize': '16px',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'not-allowed',
        'width': '100%',
        'marginBottom': '20px'
    }
    
    active_button_style = {
        'backgroundColor': '#4CAF50',
        'color': 'white',
        'padding': '12px 24px',
        'fontSize': '16px',
        'border': 'none',
        'borderRadius': '4px',
        'cursor': 'pointer',
        'width': '100%',
        'marginBottom': '20px'
    }
    
    if not selected_species or merged_df.empty:
        return (
            html.Div([
                html.P("Select a species to view information and get external link.", 
                       style={'color': '#6c757d', 'fontStyle': 'italic'})
            ]),
            True,  # Button disabled
            default_button_style,
            'View External Link'
        )
    
    # Find the corresponding data for the selected scientific name
    species_data = merged_df[merged_df['scientific_name'] == selected_species]
    
    if species_data.empty:
        return (
            html.Div([
                html.P("No data found for selected species.", 
                       style={'color': '#dc3545'})
            ]),
            True,  # Button disabled
            default_button_style,
            'View External Link'
        )
    
    # Get the class and primary_label
    class_name = species_data['class_name'].iloc[0]
    
    if 'primary_label' in species_data.columns:
        primary_label = species_data['primary_label'].iloc[0]
        
        # Determine the appropriate URL based on class
        if class_name == 'Aves':
            external_url = f"https://ebird.org/species/{primary_label}"
            button_text = 'View on eBird'
            platform_name = 'eBird'
        else:
            external_url = f"https://www.inaturalist.org/taxa/{primary_label}"
            button_text = 'View on iNaturalist'
            platform_name = 'iNaturalist'
        
        # Count occurrences of this species
        species_count = len(species_data)
        
        # Get common name if available
        common_name = None
        if 'common_name' in species_data.columns:
            common_name = species_data['common_name'].iloc[0]
        
        info_content = html.Div([
            html.H4(f"{selected_species}", style={'color': '#2c3e50', 'marginBottom': '15px'}),
            html.P([
                html.Strong("Class: "), 
                html.Span(class_name, style={'color': '#007bff', 'fontWeight': 'bold'})
            ], style={'marginBottom': '10px'}),
            html.P([
                html.Strong("Common Name: "), 
                html.Span(common_name if common_name and pd.notna(common_name) else "Not available", 
                         style={'fontStyle': 'italic' if not common_name or pd.isna(common_name) else 'normal'})
            ], style={'marginBottom': '10px'}),
            html.P([
                html.Strong("Occurrences in dataset: "), 
                f"{species_count:,}"
            ], style={'marginBottom': '10px'}),
            html.P([
                html.Strong(f"{platform_name} URL: "), 
                html.A(external_url, href=external_url, target="_blank", 
                       style={'color': '#007bff', 'textDecoration': 'none'})
            ], style={'marginBottom': '15px'}),
            html.P(f"Click the '{button_text}' button to open this species page in a new tab.", 
                   style={'color': '#6c757d', 'fontStyle': 'italic', 'fontSize': '14px'})
        ])
        
        return (
            info_content,
            False,  # Button enabled
            active_button_style,
            button_text
        )
    else:
        return (
            html.Div([
                html.H4(f"{selected_species}", style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.P([
                    html.Strong("Class: "), 
                    html.Span(class_name, style={'color': '#007bff', 'fontWeight': 'bold'})
                ], style={'marginBottom': '10px'}),
                html.P("Primary label information not available in dataset.", 
                       style={'color': '#dc3545'}),
                html.P(f"Occurrences in dataset: {len(species_data):,}", 
                       style={'marginBottom': '10px'})
            ]),
            True,  # Button disabled
            default_button_style,
            'View External Link'
        )

# Callback to handle external link button clicks
@app.callback(
    Output('external-link-button', 'n_clicks'),
    [Input('external-link-button', 'n_clicks'),
     Input('species-selector', 'value')],
    prevent_initial_call=True
)
def handle_external_link_button_click(n_clicks, selected_species):
    if n_clicks and selected_species and not merged_df.empty:
        # Find the data for the selected species
        species_data = merged_df[merged_df['scientific_name'] == selected_species]
        
        if not species_data.empty and 'primary_label' in species_data.columns:
            class_name = species_data['class_name'].iloc[0]
            primary_label = species_data['primary_label'].iloc[0]
            
            # Generate the appropriate URL
            if class_name == 'Aves':
                external_url = f"https://ebird.org/species/{primary_label}"
            else:
                external_url = f"https://www.inaturalist.org/taxa/{primary_label}"
            
            # In a real Dash app, you would use dcc.Location or clientside callback 
            # to open the URL. For now, we'll just print it.
            print(f"Opening external URL: {external_url}")
            
            # You could also use a clientside callback to actually open the URL:
            # return dcc.Location(href=external_url, id="redirect")
    
    return 0  # Reset click count

# Run the app
if __name__ == '__main__':
    app.run(debug=True)