import math
import pandas as pd
import requests

#
#   Jack Darlington
#   Student ID: 19082592
#   COMP611 - Software Assignment
#

# API Variables
api_key = '3ab0907fb2d8be67e7289d8ce2aa7119'
base_url = 'http://api.exchangeratesapi.io/v1/'

# Graph Class to store nodes, edges, and includes Bellman Ford algorithm
class Graph:
    def __init__(self):
        self.graph = {}
        self.negative_cycle_found = False  # Flag to track negative weight cycle
        self.cycle_nodes = []  # List to store nodes in the negative cycle

    def add_edge(self, from_currency, to_currency, exchange_rate):
        if from_currency not in self.graph:
            self.graph[from_currency] = []
        self.graph[from_currency].append(
            (to_currency, -math.log(exchange_rate)))
        print(f"{from_currency} -> {to_currency} - rate: {exchange_rate}, weight: {-math.log(exchange_rate)}")

    def bellman_ford(self, start_currency):
        distances = {currency: float('inf') for currency in self.graph}
        predecessors = {currency: None for currency in self.graph}
        distances[start_currency] = 0

        for _ in range(len(self.graph) - 1):
            for currency in self.graph:
                for neighbor, weight in self.graph[currency]:
                    if distances[currency] + weight < distances[neighbor]:
                        distances[neighbor] = distances[currency] + weight
                        predecessors[neighbor] = currency

        # Check for negative weight cycles (arbitrage opportunities)
        for currency in self.graph:
            for neighbor, weight in self.graph[currency]:
                if distances[currency] + weight < distances[neighbor]:
                    self.negative_cycle_found = True
                    cycle_start = currency
                    # Traverse the cycle and store nodes
                    while cycle_start not in self.cycle_nodes:
                        self.cycle_nodes.append(cycle_start)
                        cycle_start = predecessors[cycle_start]
                    self.cycle_nodes = self.cycle_nodes[self.cycle_nodes.index(cycle_start):]
                    self.cycle_nodes.append(cycle_start)

                    raise ValueError()

        return distances, predecessors

    def find_best_exchange_path(self, start_currency, end_currency):
        distances, predecessors = self.bellman_ford(start_currency)
        if distances is None:
            return None, 0.0  # Return None if a negative cycle is found

        path = []
        current_currency = end_currency

        while current_currency is not None:
            path.insert(0, current_currency)
            current_currency = predecessors[current_currency]

        return path, math.exp(-distances[end_currency])


# Get currency data function returns currency rates 'json_data' using a string 'base_currency' and array of currency symbols 'symbols'
def get_currency_data(base_currency, symbols):
    currencies = ','.join(symbols)

    # Define the endpoint, API key and query parameters
    endpoint = 'latest'
    params = {
        'access_key': {api_key},
        'base': {base_currency},
        'symbols': {currencies}
    }

    # Construct the complete URL and get response
    url = f"{base_url}{endpoint}"
    response = requests.get(url, params=params)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        json_data = response.json()
        # Access exchange rate data
        return json_data["rates"]
    else:
        # Print error message
        print(f"Error: {response.status_code} - {response.text}")

# Create pandas dataframe 'df' using an array of currency symbols 'symbols'
def create_dataframe(symbols):
    data = []
    for symbol in symbols:
        data.append(get_currency_data(symbol, symbols))

    # Create dataframe holding the exchange rate information
    df = pd.DataFrame(data)
    df.index = df.columns

    return df

# Create pandas dataframe 'df' using a graph 'G'
def create_dataframe_from_graph(G: Graph):
    print(G.graph.items())
    # Create dataframe holding the exchange rate information
    df = pd.DataFrame(G.graph)
    df.index = df.columns

    return df

# Create graph 'G' from pandas dataframe 'df'
def create_graph_from_dataframe(df):
    # Create a weighted graph
    G = Graph()

    # Add weighted edges to the graph from columns to rows
    for source_currency in range(len(df.columns)):
        for target_currency in range(len(df.index)):
            if source_currency != target_currency:
                weight = df.at[df.columns[source_currency],
                               df.index[target_currency]]
                G.add_edge(df.columns[source_currency], df.index[target_currency], weight)

    return G             



# PROGRAM START
print("\n==================================================")
print("\n  Currency Program - Shortest Path and Arbitrage  ")
print("  Created by Jack Darlington  ")
print("  Student ID: 19082592")
print("  COMP611 - Software Assignment")
print("\n==================================================\n")



### TEST DATA: COMMENT OUT TO USE API DATA
print("======================")
print("  USING TEST DATA...  ")
print("======================\n")

# TEST DATA A: No negative cycle
G = Graph()
G.add_edge("USD", "EUR", 0.85)
G.add_edge("USD", "GBP", 0.73)
G.add_edge("EUR", "GBP", 0.86)
G.add_edge("GBP", "EUR", 1.16)
G.add_edge("EUR", "JPY", 125.22)
G.add_edge("JPY", "USD", 0.009)
G.add_edge("GBP", "JPY", 140.22)

## TEST DATA B: This test data has a negative cycle USD -> GBP -> EUR -> USD
#G = Graph()
#G.add_edge("USD", "EUR", 0.85)
#G.add_edge("USD", "GBP", 0.73)
#G.add_edge("EUR", "GBP", 0.86)
#G.add_edge("GBP", "EUR", 1.16)
#G.add_edge("EUR", "JPY", 125.22)
#G.add_edge("JPY", "USD", 0.009)
#G.add_edge("GBP", "JPY", 140.22)
#G.add_edge("GBP", "USD", 1.5)



# API DATA: COMMENT OUT TO USE TEST DATA
#print("=====================")
#print("  USING API DATA...  ")
#print("=====================\n")
#
#data = create_dataframe(['NZD', 'AUD', 'USD', 'CAD', 'JPY', 'EUR', 'GBP'])
#G = create_graph_from_dataframe(data)



# SET START AND END CURRENCIES
start_currency = "USD"
end_currency = "JPY"



# Create a list of currency codes from the graph keys
currencies = list(G.graph.keys())
# Initialize an empty DataFrame with the currency codes as both rows and columns
df = pd.DataFrame(index=currencies, columns=currencies)

# Fill the DataFrame with exchange rates from the graph
for from_currency, edges in G.graph.items():
    for to_currency, exchange_rate in edges:
        df.at[from_currency, to_currency] = exchange_rate

# Fill diagonal with 1s (exchange rate from a currency to itself)
for currency in currencies:
    df.at[currency, currency] = 1.0



# PRINTING THE RESULTS
print("\nCurrency Negative Logarithm Table (Data = Negative logarithm from currency x to y)\n")
print(df)

print(f"\nCalculate best path from: \"{start_currency}\" to \"{end_currency}\"\n")

try:
    path, best_exchange_rate = G.find_best_exchange_path(
        start_currency, end_currency)
    print("Best path:", " -> ".join(path))
    print("Best exchange rate:", best_exchange_rate)
except:
    print("Negative weight cycle detected, arbitrage opportunity:")
    print(" -> ".join(G.cycle_nodes))
finally:
    print("\nGoodbye")