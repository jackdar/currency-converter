import sys
import requests
import json
import pandas as pd
import networkx as nx
import math

# API Variables
api_key = '3ab0907fb2d8be67e7289d8ce2aa7119'
base_url = 'http://api.exchangeratesapi.io/v1/'

# Get currencies from command-line arguments
# currency_symbols = sys.argv[1]


# Currency Data Function
def get_currency_data(base_currency, symbols):

    currencies = ','.join(symbols)

    # Define the endpoint, API key and query parameters
    endpoint = 'latest'
    params = {
        'access_key': {api_key},
        'base': {base_currency},
        'symbols': {currencies}
    }

    # Construct the complete URL
    url = f"{base_url}{endpoint}"

    # Make the GET request
    response = requests.get(url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the JSON response
        json_data = response.json()

        # Access exchange rate data
        return json_data["rates"]
    else:
        # Print error message if response is not as expected
        print(f"Error: {response.status_code} - {response.text}")


# Create Adjacency Matrix in the form of a Pandas Dataframe
def create_adjacency_matrix(symbols):
    data = []
    for symbol in symbols:
        data.append(get_currency_data(symbol, symbols))

    # Create dataframe holding the exchange rate information
    df = pd.DataFrame(data)
    df.index = df.columns

    return df


# Print adjacency matrix from Pandas Dataframe
def print_adjacency_matrix(df):
    print(df)


# Create weighted graph G
def create_weighted_graph(df):
    # Create a weighted graph
    G = nx.DiGraph()

    # Add nodes to the graph
    for currency in df.index:
        G.add_node(currency)

    # Add weighted edges to the graph from columns to rows
    for source_currency in df.columns:
        for target_currency in df.index:
            if source_currency != target_currency:
                weight = df.at[source_currency, target_currency]
                G.add_edge(source_currency, target_currency, weight=weight)

    return G


# Print weighted graph G
def print_weighted_graph(G):
    # Print the graph edges and weights
    for source, target, data in G.edges(data=True):
        weight = data['weight']
        logarithm = math.log(data['weight'])
        print(f"{source} -> {target}: {weight}, log({source},{target}) = {logarithm}")


# Calculate logarithm
def calculate_logarithm():
    logarithm = 0

    for source, target, data in G.edges(data=True):
        if logarithm == 0:
            logarithm = math.log(data['weight'])
        else:
            logarithm *= math.log(data['weight'])

    print(logarithm)


df = create_adjacency_matrix(['NZD', 'AUD', 'EUR'])
G = create_weighted_graph(df)

print_adjacency_matrix(df)
print()
print_weighted_graph(G)
print()
calculate_logarithm(G)
