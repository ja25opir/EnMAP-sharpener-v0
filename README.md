# EnMAP-sharpener-v0

Sharpening the spatial resolution of scenes from EnMAP satellite mission using a CNN and scenes from Sentinel-2.
The used model can be found in the models directory (output/models/supErMAPnet.keras).

## setup project

### install dependencies:

``pip install -r requirements.txt``

### set the PYTHONPATH environment variable to include the project root:
``export PYTHONPATH=$(pwd)``

## create .env

Copy .env.example to .env and fill in COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET received from your Copernicus user account.
This is needed to fetch Sentinel-2 data from the Sentinel Hub Process API.
