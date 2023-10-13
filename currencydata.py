import sys
import requests
import json
import pandas as pd
import math

# API Variables
api_key = '3ab0907fb2d8be67e7289d8ce2aa7119'
base_url = 'http://api.exchangeratesapi.io/v1/'

# Get currencies from command-line arguments
# currency_symbols = sys.argv[1]


class Graph:
    def __init__(self, vertices):
        self.V = vertices
        self.edges = []

    def add_edge(self, v, u, w):
        self.edges.append((v, u, w))

    def bellman_ford(self, src):
        # Initialize distances
        d = [float("Inf")] * (self.V + 1)  # starting from 1
        d[src] = 0

        # Repeat |V| - 1 times
        for _ in range(self.V-1):
            d_prime = list(d)  # Copy current distances to d'
            # self.print_distances(d) # Print the intermediate distances for d.
            for v, u, w in self.edges:
                d_prime[u] = min(d_prime[u], d[v] + w)

            d = d_prime  # Replace d by d'

        # Print distances
        self.print_distances(d)

    def print_distances(self, dist):
        print("Vertex Distance from Source")
        for i in range(1, len(dist)):  # Start from 1
            print("{0}\t\t{1}".format(i, dist[i]))


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
    G = Graph(len(df.index))

    # Add weighted edges to the graph from columns to rows
    for source_currency in range(len(df.columns)):
        for target_currency in range(len(df.index)):
            if source_currency != target_currency:
                weight = df.at[df.columns[source_currency],
                               df.index[target_currency]]
                G.add_edge(source_currency, target_currency, -math.log(weight))

    return G


# Print weighted graph G
def print_weighted_graph(G):
    # Print the graph edges and weights
    for source, target, data in G.edges(data=True):
        weight = data['weight']
        print(f"{source} -> {target}: {weight}")


# TEST CASES
df = create_adjacency_matrix(['NZD', 'AUD', 'EUR'])
G = create_weighted_graph(df)
print_adjacency_matrix(df)
print()
G.bellman_ford(1)
