import dash
from dash import Dash, html, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
from dash import dcc, Input, Output, callback
import torchaudio
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import torch
from torchaudio.transforms import Resample
import base64
from io import BytesIO
import wave


# Loading the data
df = pd.read_csv("data/train.csv")
column_options = [{'label': col, 'value': col} for col in df.columns]


# Random stuff
SAMPLE_RATE = 32000

db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

def decode_array(b64_string):
    data = base64.b64decode(b64_string)
    buffer = BytesIO(data)
    return np.load(buffer)

vad_model = None

def load_vad_model():
    global vad_model
    if vad_model is None:
        vad_model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        vad_model.eval()
        
load_vad_model()

resample = Resample(orig_freq=32000, new_freq=16000)


# Layout
layout = dbc.Container([
    dcc.Store(id="spectrogram"),
    dcc.Store(id="vad-predictions"),
    # dbc.Row([
    #     html.Div(
    #         'Spectrograms',
    #         className="text-primary text-center fs-3",
    #         style={'padding': '20px'}
    #     )
    # ]),
    html.Div(style={'height': '30px'}),
    dbc.Row(
        [
            dbc.Col(
                dash_table.DataTable(
                    data=df.drop(columns=[
                        "secondary_labels", 
                        "type", 
                        "author", 
                        "license", 
                        "url", 
                        "primary_label", 
                        "latitude", 
                        "longitude", 
                        "scientific_name"]
                    ).to_dict("records"),
                    style_table={
                        'maxWidth': '1000px',
                        'margin': '0 auto',
                        'overflowX': 'clip',
                        'fontFamily': 'system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue","Noto Sans","Liberation Sans",Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"',
                    },
                    style_cell={
                        'minWidth': '80px',
                        'maxWidth': '200px',
                        'overflow': 'clip',
                        'textAlign': 'left',
                        'padding': '5px',
                        'textOverflow': 'ellipsis',
                        'fontFamily': 'system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue","Noto Sans","Liberation Sans",Arial,sans-serif,"Apple Color Emoji","Segoe UI Emoji","Segoe UI Symbol","Noto Color Emoji"'
                    },
                    style_header={
                        'fontWeight': 'bold'
                    },
                    style_data_conditional=[
                        {
                            'if': {'column_id': 'rating'},
                            'textAlign': 'right'
                        }
                    ],
                    page_size=15,
                    cell_selectable=False,
                    sort_action="native",
                    sort_mode="multi",
                    row_selectable="single",
                    selected_rows=[0],
                    id="data-table"
                ),
                width=6,
                style={'display': 'flex', 'justifyContent': 'center'}
            ),
            dbc.Col(
                [
                    dcc.Graph(figure={}, id="mel-settings-graph"),
                    dbc.Container([
                        dbc.Row([
                            dbc.Col([
                                html.Label("FFT size"),
                                dcc.Input(
                                    id='N_FFT', type='number', min=256, max=4096, step=1,
                                    debounce=True, value=1024,
                                    style={'width': '100%'}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Hop length"),
                                dcc.Input(
                                    id='HOP_LENGTH', type='number', min=64, max=1024, step=1,
                                    debounce=True, value=512,
                                    style={'width': '100%'}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Num. of mels"),
                                dcc.Input(
                                    id='N_MELS', type='number', min=64, max=1024, step=1,
                                    debounce=True, value=128,
                                    style={'width': '100%'}
                                )
                            ], width=4),
                        ], className="mb-3"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Min freq."),
                                dcc.Input(
                                    id='FMIN', type='number', min=10, max=5000, step=1,
                                    debounce=True, value=20,
                                    style={'width': '100%'}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Max freq."),
                                dcc.Input(
                                    id='FMAX', type='number', min=10000, max=20000, step=1,
                                    debounce=True, value=14000,
                                    style={'width': '100%'}
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Power"),
                                dcc.Input(
                                    id='POWER', type='number', min=1, max=5, step=1,
                                    debounce=True, value=2,
                                    style={'width': '100%'}
                                )
                            ], width=4)
                        ], className="mb-3")
                    ], style={'marginTop': '15px', 'marginBottom': '10px'})
                ],
                width=6
            )
        ],
        align="start",
        style={'display': 'flex', 'justifyContent': 'center'}
    ),
    html.Div(style={'height': '30px'}),
    dbc.Container(
        [    
            dbc.Row([
                html.Div(
                    'Audio',
                    className="text-primary text-start fs-4",
                    style={'paddingBottom': '15px', 'color': '#0d6efd', 'textAlign': 'left', 'marginLeft': '50px', 'paddingTop': '10px'}
                )
            ]),
            dbc.Row(
                [
                    html.Audio(
                        id='audio-player',
                        src='',
                        controls=True,
                        style={'width': '90%', 'marginBottom': '25px'}
                    )
                ],
                style={'display': 'flex', 'justifyContent': 'center'}
            )
        ],
        style={
            'border': '1px solid #ccc',
            'borderRadius': '6px',
            'padding': '10px',
            'marginBottom': '30px',
            'width': '95%',
        },
        fluid=True
    ),
    dbc.Row(
        [
            dbc.Col(
                dcc.Graph(figure={}, id="vad-graph")
            ),
            dbc.Col(
                [
                    dcc.Graph(id='vad-percentage', figure={}),
                    html.Label("VAD threshold"),
                    dcc.Input(
                        id='VAD_THRESHOLD', type='number', min=0, max=1, step=0.1,
                        debounce=True, value=0.5,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'height': '10px'}),
                    html.Label("Chunk size"),
                    dcc.Input(
                        id='CHUNK_SIZE', type='number', min=1, max=100, step=1,
                        debounce=True, value=1,
                        style={'width': '100%'}
                    ),
                ],
                width=2
            )
        ]
    ),
])


# Callbacks
@callback(
    Output(component_id='spectrogram', component_property='data'),
    Input(component_id='data-table', component_property='selected_rows'),
    Input(component_id='N_FFT', component_property='value'),
    Input(component_id='HOP_LENGTH', component_property='value'),
    Input(component_id='N_MELS', component_property='value'),
    Input(component_id='FMIN', component_property='value'),
    Input(component_id='FMAX', component_property='value'),
    Input(component_id='POWER', component_property='value')
)
def update_mel_settings(rows, N_FFT, HOP_LENGTH, N_MELS, FMIN, FMAX, POWER):
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        f_min=FMIN,
        f_max=FMAX,
        power=POWER
    )
    
    audio_path = "/mnt/c/users/adamm/documents/code/bird_clef_2025/bird_clef/data/train_audio/"
    audio_path += df.loc[rows[0], "filename"]
    
    audio, _ = torchaudio.load(audio_path) 
    
    spectrogram = mel_transform(audio)
    spectrogram = db_transform(spectrogram)
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)
    
    # Audio buffer
    buffer_audio = BytesIO()
    np.save(buffer_audio, audio.numpy())
    
    # Spectrogram buffer
    spec_np = spectrogram.squeeze().numpy()
    buffer_spec = BytesIO()
    np.save(buffer_spec, spec_np)
    
    # Audio WAV buffer
    audio_np = audio.squeeze().numpy()
    audio_normalized = np.clip(audio_np * 32767, -32768, 32767).astype(np.int16)
    
    buffer_wav = BytesIO()
    
    with wave.open(buffer_wav, 'wb') as wav_file:
        wav_file.setnchannels(1) 
        wav_file.setsampwidth(2) 
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_normalized.tobytes())

    return {
        "spectrogram": base64.b64encode(buffer_spec.getvalue()).decode('utf-8'), 
        "audio_wav": base64.b64encode(buffer_wav.getvalue()).decode('utf-8'), 
        "audio": base64.b64encode(buffer_audio.getvalue()).decode('utf-8'),
        "audio_length": audio.shape[1] / SAMPLE_RATE
    }
    

@callback(
    Output(component_id='vad-predictions', component_property='data'),
    Input(component_id='spectrogram', component_property='data'),
    Input(component_id='CHUNK_SIZE', component_property='value')
)   
def update_vad_settings(data, chunk_size):
    audio = data.get("audio", None)
    
    audio = torch.Tensor(decode_array(audio))
    
    audio_r = resample(audio)
    
    with torch.no_grad():
        predicts = vad_model.audio_forward(audio_r, sr=16000)
        
    predicts = predicts.unfold(1, chunk_size, chunk_size).mean(dim=2)
        
    buffer = BytesIO()
    np.save(buffer, predicts.numpy())
    
    return {"predictions": base64.b64encode(buffer.getvalue()).decode('utf-8')}
    
    
@callback(
    Output(component_id='vad-percentage', component_property='figure'),
    Input(component_id='vad-predictions', component_property='data'),
    Input(component_id='VAD_THRESHOLD', component_property='value'),
)
def update_vad_percentage(data, threshold):
    predicts = data.get("predictions", None)
    predicts = decode_array(predicts)
    
    if predicts is not None:
        total_frames = predicts.shape[1]
        vad_frames = (predicts > threshold).sum()
        percentage = (vad_frames / total_frames * 100).item() if total_frames > 0 else 0
    else:
        percentage = 0
        
    percentage = round(percentage, 2)

    figure = go.Figure(go.Pie(
        values=[percentage, 100 - percentage],
        hole=0.6,
        domain={'x': [0, 1], 'y': [0, 1]},
        marker_colors=[ '#0d6efd', "#E5E7E8"],
        textinfo='none',
        hoverinfo='none'
    ))

    figure.update_layout(
        title='Percentage of VAD',
        annotations=[dict(
            text=f'{percentage}%',
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )],
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        width=200,
        height=200
    )
    figure.update_layout(margin=dict(l=20, r=20, t=40, b=20))

    return figure


@callback(
    Output(component_id='mel-settings-graph', component_property='figure'),
    Input(component_id="spectrogram", component_property="data")
)
def main_spectrogram(data):
    spectrogram = data.get("spectrogram", None)
    audio_length = data.get("audio_length", None)
    
    spectrogram = decode_array(spectrogram)   
    
    num_frames = spectrogram.shape[1]
    x_vals = np.linspace(0, audio_length, num_frames)

    fig = px.imshow(
        spectrogram,
        origin='lower',
        aspect='auto',
        color_continuous_scale='plasma',
        labels={'x': 'Time (s)', 'y': 'Mel Bin', 'color': 'Power (dB)'},
        title='Mel Spectrogram (Settings)',
        x=x_vals
    )
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
        title='Time (s)',
        range=[0, 5]
    )
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=14),
        title_font=dict(size=20, family="Arial"),
        coloraxis_colorbar=dict(
            title="Power (dB)",
            thickness=15,
            len=0.75,
            yanchor='middle',
            y=0.5
        ),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    
    return fig


@callback(
    Output(component_id='vad-graph', component_property='figure'),
    Input(component_id='spectrogram', component_property='data'),
    Input(component_id='vad-predictions', component_property='data')
)
def vad_spectrogram(data, predictions_data):
    spectrogram = data.get("spectrogram", None)
    audio_length = data.get("audio_length", None)
    predictions = predictions_data.get("predictions", None)
    
    spectrogram = decode_array(spectrogram)   
    predictions = decode_array(predictions)
    
    num_frames = spectrogram.shape[1]
    x_vals = np.linspace(0, audio_length, num_frames)

    fig = px.imshow(
        spectrogram,
        origin='lower',
        aspect='auto',
        color_continuous_scale='plasma',
        labels={'x': 'Time (s)', 'y': 'Mel Bin', 'color': 'Power (dB)'},
        title='Mel Spectrogram with Voice Activity Detection',
        x=x_vals
    )

    fig.add_trace(go.Scatter(
        x=np.linspace(0, audio_length, predictions.shape[1]),
        y=predictions[0],
        mode='lines',
        name='Density',
        line=dict(color='#0d6efd', width=2),
        fill='tozeroy',
        fillcolor='rgba(13, 110, 253, 0.3)',
        yaxis='y2'
    ))

    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)',
        title='Time (s)',
        range=[0, audio_length]
    )

    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200,200,200,0.3)')

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=14),
        title_font=dict(size=20, family="Arial"),
        coloraxis_colorbar=dict(
            title="Power (dB)",
            thickness=15,
            len=0.75,
            yanchor='middle',
            y=0.5,
            x=1.07 
        ),
        margin=dict(l=20, r=50, t=40, b=20),
        yaxis2=dict(
            title='VAD',
            overlaying='y',
            side='right',
            range=[0, 1]
        )
    )
    
    return fig


@callback(
    Output(component_id='audio-player', component_property='src'),
    Input(component_id='spectrogram', component_property='data')
)
def update_audio_player(data):
    audio = data.get("audio_wav", None)
        
    if audio is not None:
        return f"data:audio/wav;base64,{audio}"
    return ""

dash.register_page("Spectrograms", path='/spectrograms', layout=layout, name="Spectrograms")