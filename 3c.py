#upload all the neccessary library
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import pycountry
import pygal
from pygal.style import Style
from pygal_maps_world.maps import World
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html, Dash
import base64
from skimage import io

# #uploaded the dataset
# database=pd.read_csv('TemperatureDataCountryWise.csv')
# database.pop("AverageTemperatureUncertainty")
# database

# #data cleaning
# dataset2=database[['dt']].copy() #we copied the dates column from database
# dataset2.rename(columns = {'dt':'Date'}, inplace = True) #replace the heading with date
# dataset2.drop_duplicates(inplace=True) #drop all the duplicates
# dataset2.set_index('Date', inplace=True) #Set the DataFrame index using existing columns.

# for i in database.index: #iterate over database
#   date,country,temp=database['dt'][i],database['Country'][i],database['AverageTemperature'][i] #set the value of date , country , temperature
#   dataset2.at[date,country]=temp  #set the dataset

# for i in dataset2.columns[dataset2.isnull().any(axis=0)]:     #Applying Only on variables with NaN values
#     dataset2[i].fillna(dataset2[i].mean(),inplace=True) #we will fill all the null values

# dataset2.shape #Return a tuple representing the dimensionality of the DataFrame.

# dataset2 = dataset2.dropna(axis='columns') #remove the empty columns
# # dataset2.shape  #Return a tuple representing the dimensionality of the DataFrame.

# dataset2=dataset2.transpose() #transpose the dataset
# # dataset2

# # Define a normalizer
# normalizer = Normalizer()
# # Reduce the data
# reduced_data = PCA(n_components = 2)
# # Create Kmeans model
# kmeans = KMeans(n_clusters = 10,max_iter = 1000)
# # Make a pipeline chaining normalizer and kmeans
# pipeline = make_pipeline(normalizer,kmeans)
# # Fit pipeline to daily stock movements
# pipeline.fit(dataset2)
# labels = pipeline.predict(dataset2)

# countries=dataset2.index.tolist()
# dataframe1 = pd.DataFrame({'labels':labels,'countries':countries}).sort_values(by=['labels'],axis = 0) #create a dataframe with labels and countries
# # dataframe1

# # Get all country mappings, using two digit codes
# map_of_country = {country.name: country.alpha_2 for country in pycountry.countries}
# # create a Style object
# color_plate = Style( background='#C2B8A3',
#   plot_background='#EEEEEE',opacity='.6',
#   opacity_hover='.9',
#   transition='400ms ease-in', colors = ('#B68973' , '#9D5353' ,'#3282B8' ,'#424874', '#0D7377','#533E85','#FFDE7D','#3C415C','#C5E3F6','#D4ECDD'))

# # create a world map,
# # Style class is used for using
# # the custom colours in the map,
# wm =  World(style= color_plate)

# # set the title of the map
# wm.title = 'Clustering done on the basis on average temperature movement over the years'

# #create  new dict
# clusters={}
# #iterating over dataframe
# for i in dataframe1.index:
#   #labels of index i
#   label=dataframe1['labels'][i]
#   #country at index i
#   country=dataframe1['countries'][i]
#   #Checking if country is in our map or not
#   if country in map_of_country:
#     #set the country code
#     countrycode=map_of_country[country]
#     # print(countrycode)
#     #check is this label is present in cluseters or not
#     if label not in clusters:
#       clusters[label]=[countrycode.lower()]
#     #if present we append in label's index
#     else:
#       clusters[label].append(countrycode.lower())

# for c in clusters:
#   wm.add('Cluster '+str(c),clusters[c])

# wm.render_to_file('img.svg')
# wm.render_to_png('imgn.png')

# app = dash.Dash()
# external_stylesheets = ["dash_design.css"]


# app = Dash(__name__)

# image_filename = 'img.png'  # replace with your own image
# encoded_image = base64.b64encode(open(image_filename, 'rb').read())

# df = pd.DataFrame({
#     "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
#     "Amount": [4, 1, 2, 2, 4, 5],
#     "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
# })

# fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")

# image = io.imread('img.png')
# fig2 = px.imshow(image)

# app.layout = html.Div(children=[
#     html.H1(children='Hello Dash'),
#     html.Div(children='''
#         Dash: A web application framework for your data.
#     '''),
#     dcc.Graph(id='example-graph', figure=fig),
#     html.Br(),
#     # plot the image
#     html.Div(
#         html.Img(
#             src='data:image/png;base64,{}'.format(encoded_image.decode()),
#             style={
#                 "width": "100%",
#                 "display": "block",
#                 "backgroundColor": "#FAEDF0",
#                 "backgroundColor": "#FAEDF0",
#                 # align it to the centre
#                 "margin": "20px auto",
#             },
#         )),
#     html.Div(
#         [dcc.Graph(figure=fig2)],
#         style={
#             "width": "100%",
#             # "display": "block",
#             "backgroundColor": "#FAEDF0",
#         },
#     ),
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)

app = Dash(__name__)

image_filename = 'img.png'  # replace with your own image
encoded_image = base64.b64encode(open(image_filename, 'rb').read())



image = io.imread('img.png')
fig = px.imshow(image)

app.layout = html.Div(children=[
    html.H1(children='Clusterings of Countries'),
    html.Br(),
    # plot the image
    html.Div(
        html.Img(
            src='data:image/png;base64,{}'.format(encoded_image.decode()),
            style={
                "width": "100%",
                "display": "block",
                "backgroundColor": "#FAEDF0",
                # align it to the centre
                "margin": "20px auto",
            },
        )),
    html.Div(
        [dcc.Graph(figure=fig)],
        style={
            "width": "100%",
            # "display": "block",
            "backgroundColor": "#FAEDF0",
        },
    ),
])

if __name__ == '__main__':
    app.run_server(debug=True,use_reloader=False,port=8051)
