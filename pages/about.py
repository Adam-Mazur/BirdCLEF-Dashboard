from dash import html
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, callback
import dash


# Layout
layout = dbc.Container([
    html.Div(style={'height': '30px'}),
    
    html.H1("BirdCLEF Dashboards", className="text-center mb-4", style={'color': '#2c3e50'}),
    
    html.P([
        "This dashboard analyzes the data from the BirdCLEF 2025 Kaggle competition. The competition tries to find the best machine learning models for classifying animals based on sound recordings. The competition is held annually, and the best model (measured with a modified ROC AUC metric) wins. This years competition includes 206 different animal species, mainly birds.\n\n", 

        "The dashboard is organized into 4 tabs:\n",
        "1. Home\n",
        "2. Spectrograms\n",
        "3. Predictions\n",
        "4. About\n\n",

        "The Home page analyzes basic properties of the dataset like the species distribution, their geographic location and the distribution of animal classes.\n\n",

        "The Spectrograms page analyzes the Mel spectrograms generated from the audio recordings. Most machine learning models used in this competition first convert the audio signals to spectrograms to then use computer vision models on them. This page compares different settings of Mel spectrograms and also compares different voice activity detection settings on them, as the recordings contains many human voices along with animal sounds.\n\n",

        "The Predictions page analyzes a basic's CNN model predictions on those sounds. In particular, it shows the class activation mapping on a given audio segment. It also analyzes the predicted labels given some post-processing hyperparameters.\n\n",
    ], className="lead text-justify", style={'color': '#34495e', 'line-height': '1.6', 'padding-left': '3rem', 'padding-right': '3rem', 'white-space': 'pre-line'})
], fluid=True, className="py-4")


dash.register_page("About", path='/about', layout=layout, name="About")
