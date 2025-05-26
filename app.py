import dash
from dash import Dash
import dash_bootstrap_components as dbc


name = "BirdCLEF Dashboard"
app = Dash(__name__, external_stylesheets=[dbc.themes.CERULEAN], title=name, use_pages=True)

app.layout = dbc.Container(
    [
        dbc.Nav(
            [
                dbc.NavItem(dbc.NavLink("Home", active="exact", href="/", style={"fontSize": "17px"})),
                dbc.NavItem(dbc.NavLink("Spectrograms", active="exact", href="/spectrograms", style={"fontSize": "17px"})),
                dbc.NavItem(dbc.NavLink("Predictions", active="exact", href="/predictions", style={"fontSize": "17px"})),
                dbc.NavItem(dbc.NavLink("About", active="exact", href="/about", style={"fontSize": "17px"})),
            ],
            pills=True,
            fill=True,
            className="border",
            style={"margin": "0px", "padding": "5px"}
        ),
        dash.page_container,
    ],
    fluid=True,
    className="p-0 m-0",
    style={"padding": "0", "margin": "0"}
)

if __name__ == '__main__':
    app.run(debug=True, port=8052)