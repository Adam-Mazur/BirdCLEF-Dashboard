import dash
from dash import html, dash_table
import dash_bootstrap_components as dbc
from dash import dcc, Input, Output, callback
import torchaudio
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import torch
import base64
from io import BytesIO
import wave
from load_data import train_df as df
import torch.nn as nn
from torchvision import models
import torchvision
import torch.nn.functional as F
from pathlib import Path


# Random stuff
SAMPLE_RATE = 32000

db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")

def decode_array(b64_string):
    data = base64.b64decode(b64_string)
    buffer = BytesIO(data)
    return np.load(buffer)

mel_transform = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, 
    n_fft=2048, 
    hop_length=512,
    n_mels=512,
    f_min=20,
    f_max=14000,
    power=2
)

def to_spectogram(audio):
    mel_spec = mel_transform(audio)

    mel_spec_db = db_transform(mel_spec)
    mel_spectogram = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    # Reshaping it to (256, 256)
    if mel_spectogram.shape != (256, 256):
        mel_spectogram = resize(mel_spectogram)
        
    return mel_spectogram

def get_segments(audio_data):
    audio_segments = []
    total_segments = int(audio_data.shape[1] / (SAMPLE_RATE * 5))

    for segment_idx in range(total_segments):
        start_sample = segment_idx * SAMPLE_RATE * 5
        end_sample = start_sample + SAMPLE_RATE * 5
        segment_audio = audio_data[:, start_sample:end_sample]
        
        
        # Pad the audio if it is shorter than the required window size
        # TODO: Change this
        if segment_audio.shape[1] < SAMPLE_RATE * 5:
            segment_audio = F.pad(
                segment_audio,
                (0, SAMPLE_RATE * 5 - segment_audio.shape[1]),
                mode='constant'
            )

        audio_segments.append(segment_audio)

    audio_segments = torch.stack(audio_segments)
    
    spectograms = to_spectogram(audio_segments)
    spectograms = spectograms.float()

    return spectograms

def apply_power_to_low_ranked_cols(
    p: np.ndarray,
    top_k: int = 30,
    exponent = 2,
    inplace: bool = True
) -> np.ndarray:
    if not inplace:
        p = p.copy()

    # Identify columns whose max value ranks below `top_k`
    tail_cols = np.argsort(-p.max(axis=0))[top_k:]

    # Apply the power transformation to those columns
    p[:, tail_cols] = p[:, tail_cols] ** exponent
    return p

resize = torchvision.transforms.Resize((256, 256))

class BirdCLEFModel(nn.Module):
    def __init__(self, path):
        super().__init__()
        
        self.base_model = models.efficientnet_b0(weights=None)
        
        # Removing the first conv layer and replacing it 
        # with a 1 channel version
        old_conv = self.base_model.features[0][0]
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None
        )

        with torch.no_grad():
            new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)

        self.base_model.features[0][0] = new_conv
        
        # Removing the old classifier
        old_fc = self.base_model.classifier[1]
        self.base_model.classifier[1] = nn.Linear(old_fc.in_features, 206)
        
        # Loading the checkpoint
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.load_state_dict(checkpoint)
        self.base_model.eval()
    
    def forward(self, x):
        return self.base_model(x)  

model = BirdCLEFModel("resources/20-5-10-29.pt")


# Layout
layout = dbc.Container([
    dcc.Store(id="audio"),
    dcc.Store(id="spectrogram2"),
    dcc.Store(id="predictions"),
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
                    id="data-table2"
                ),
                width=6,
                style={'display': 'flex', 'justifyContent': 'center'}
            ),
            dbc.Col(
                [
                    dcc.Graph(figure={}, id="cam-graph"),
                    html.Div(style={'height': '40px'}),
                    html.Label("Time slider (seconds)", style={'paddingLeft': '30px'}),
                    dbc.Container([
                        dbc.Row([
                            html.Div(
                                dcc.Slider(0, 100, 5, value=0, id='time-slider', included=False),
                                style={'paddingRight': '30px'}
                            )
                        ], className="mb-3"),
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
                        id='audio-player2',
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
                dcc.Graph(figure={}, id="predictions-graph")
            ),
            dbc.Col(
                [
                    html.Label("Top K predictions"),
                    dcc.Input(
                        id='TOP_K', type='number', min=1, max=100, step=1,
                        debounce=True, value=30,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'height': '10px'}),
                    html.Label("Exponent"),
                    dcc.Input(
                        id='EXPONENT', type='number', min=1, max=10, step=0.1,
                        debounce=True, value=2,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'height': '10px'}),
                    html.Label("Smoothing param. 1"),
                    dcc.Input(
                        id='P1', type='number', min=0.01, max=1.0, step=0.01,
                        debounce=True, value=0.2,
                        style={'width': '100%'}
                    ),
                    html.Div(style={'height': '10px'}),
                    html.Label("Smoothing param. 2"),
                    dcc.Input(
                        id='P2', type='number', min=0.01, max=1.0, step=0.01,
                        debounce=True, value=0.8,
                        style={'width': '100%'}
                    )
                ],
                width=2
            )
        ],
        align="center"
    ),
])


# Callbacks
@callback(
    Output(component_id='audio', component_property='data'),
    Input(component_id='data-table2', component_property='selected_rows'),
)
def update_audio_settings(rows):
    audio_path = "/mnt/c/users/adamm/documents/code/bird_clef_2025/bird_clef/data/train_audio/"
    audio_path += df.loc[rows[0], "filename"]
    
    audio, _ = torchaudio.load(audio_path) 
    
    # Audio buffer
    buffer_audio = BytesIO()
    np.save(buffer_audio, audio.numpy())
    
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
        "audio_wav": base64.b64encode(buffer_wav.getvalue()).decode('utf-8'), 
        "audio": base64.b64encode(buffer_audio.getvalue()).decode('utf-8'),
        "audio_length": audio.shape[1] / SAMPLE_RATE
    }
    

@callback(
    Output(component_id='time-slider', component_property='max'),
    Input(component_id='audio', component_property='data')
)
def update_time_slider(data):
    audio_length = data.get("audio_length", None)
    
    if audio_length is not None:
        return (audio_length // 5) * 5
    return 0


@callback(
    Output(component_id="spectrogram2", component_property="data"),
    Input(component_id='audio', component_property='data'),
    Input(component_id='time-slider', component_property='value')
)
def update_spectrogram(data, time_value):
    audio = data.get("audio", None)
    
    if audio is None:
        return {}

    audio = decode_array(audio)
    audio = torch.tensor(audio, dtype=torch.float32)
    
    if time_value * SAMPLE_RATE >= audio.shape[1]:
        time_value = 0
        
    audio = audio[:, time_value * SAMPLE_RATE:(time_value + 5) * SAMPLE_RATE]

    spectrogram = mel_transform(audio)
    spectrogram = db_transform(spectrogram)
    spectrogram = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min() + 1e-8)

    # Spectrogram buffer
    spec_np = spectrogram.squeeze().numpy()
    buffer_spec = BytesIO()
    np.save(buffer_spec, spec_np)

    return {
        "spectrogram": base64.b64encode(buffer_spec.getvalue()).decode('utf-8')
    }


@callback(
    Output(component_id='cam-graph', component_property='figure'),
    Input(component_id="spectrogram2", component_property="data")
)
def cam_graph(data):
    spectrogram = data.get("spectrogram", None)
    
    if spectrogram is None:
        return go.Figure()
    
    spectrogram = decode_array(spectrogram)   
        
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
        
    target_layer = model.base_model.features[-1][0]  
    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    input_tensor = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
    
    if input_tensor.shape != (256, 256):
        input_tensor = resize(input_tensor)
        
    input_tensor = input_tensor.unsqueeze(0)  # shape: (1, 1, 256, 256)
    
    output = model(input_tensor)
    class_idx = output.argmax().item()
    
    model.zero_grad()
    output[0, class_idx].backward()
    
    grads = gradients[0]
    fmap = features[0]
    weights = grads.mean(dim=(2, 3), keepdim=True)

    cam = (weights * fmap).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)
    cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().detach().numpy()
    
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    handle_fwd.remove()
    handle_bwd.remove()

    fig = px.imshow(
        spectrogram,
        origin='lower',
        aspect='auto',
        color_continuous_scale='plasma',
        labels={'x': 'Time (s)', 'y': 'Mel Bin', 'color': 'Power (dB)'},
        title='Class Activation Map',
        x=np.linspace(0, 5, spectrogram.shape[1])
    )
    fig.add_trace(
        go.Heatmap(
            z=cam,
            x=np.linspace(0, 5, cam.shape[1]),
            y=np.linspace(0, spectrogram.shape[0], cam.shape[0]),
            colorscale='gray',
            opacity=0.4,
            showscale=False
        )
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
    Output(component_id='predictions', component_property='data'),
    Input(component_id='audio', component_property='data'),
)
def generate_predictions(audio):
    audio_data = audio.get("audio", None)
    if audio_data is None:
        return {}
    
    audio_data = decode_array(audio_data)
    audio_data = torch.tensor(audio_data, dtype=torch.float32)
    
    spectograms = get_segments(audio_data)
    
    with torch.no_grad():
        outputs = model(spectograms)
        predictions = torch.sigmoid(outputs)
        predictions = predictions.cpu().numpy().squeeze()

    buffer_preds = BytesIO()
    np.save(buffer_preds, predictions)
    
    return {
        "predictions": base64.b64encode(buffer_preds.getvalue()).decode('utf-8')
    }

@callback(
    Output(component_id='predictions-graph', component_property='figure'),
    Input(component_id='predictions', component_property='data'),
    Input(component_id='TOP_K', component_property='value'),
    Input(component_id='EXPONENT', component_property='value'),
    Input(component_id='P1', component_property='value'),
    Input(component_id='P2', component_property='value')
)
def predictions_graph(data, top_k, exponent, p1, p2):
    predictions = data.get("predictions", None)
    
    if predictions is None:
        return go.Figure()
    
    predictions = decode_array(predictions)
    
    predictions = apply_power_to_low_ranked_cols(predictions, top_k=top_k, exponent=exponent)
    
    new_predictions = predictions.copy()
    
    if predictions.shape[0] > 1:        
        for i in range(1, predictions.shape[0]-1):
            new_predictions[i] = (predictions[i-1] * p1) + (predictions[i] * (1 - 2 * p1)) + (predictions[i+1] * p1)
        
        new_predictions[0] = (predictions[0] * p2) + (predictions[1] * (1-p2))
        new_predictions[-1] = (predictions[-1] * p2) + (predictions[-2] * (1-p2))

    predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-8)
    heatmap_data = predictions.T
    
    time_labels = list(range(0, len(predictions) * 5, 5))
        
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=time_labels,
        y=list(range(predictions.shape[1])),
        colorscale='Blues',
        zmin=0,
        zmax=1,
        colorbar=dict(
            title="Probability"
        ),
        hovertemplate='Time: %{x}s<br>Species: %{y}<br>Probability: %{z:.3f}<extra></extra>'
    ))
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Arial", size=14),
        title_font=dict(size=20, family="Arial"),
        title=f"Predictions Heatmap (for different post-processing parameters)",
        xaxis_title="Time (s)",
        yaxis_title="Species",
    )
    
    fig.update_xaxes(
        tickvals=time_labels,
        ticktext=time_labels
    )
    
    return fig


@callback(
    Output(component_id='audio-player2', component_property='src'),
    Input(component_id='audio', component_property='data')
)
def update_audio_player(data):
    audio = data.get("audio_wav", None)
        
    if audio is not None:
        return f"data:audio/wav;base64,{audio}"
    return ""

dash.register_page("Predictions", path='/predictions', layout=layout, name="Predictions")