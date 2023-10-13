import math
import pandas as pd
import requests


class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, from_currency, to_currency, exchange_rate):
        if from_currency not in self.graph:
            self.graph[from_currency] = []
        self.graph[from_currency].append(
            (to_currency, -math.log(exchange_rate)))

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
                    raise ValueError(
                        "Negative weight cycle detected, arbitrage opportunity!")

        return distances, predecessors

    def find_best_exchange_path(self, start_currency, end_currency):
        distances, predecessors = self.bellman_ford(start_currency)
        path = []
        current_currency = end_currency

        while current_currency is not None:
            path.insert(0, current_currency)
            current_currency = predecessors[current_currency]

        return path, math.exp(-distances[end_currency])


# API Variables
api_key = '3ab0907fb2d8be67e7289d8ce2aa7119'
base_url = 'http://api.exchangeratesapi.io/v1/'


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
def create_adjacency_matrix(G: Graph):
    # Create dataframe holding the exchange rate information
    df = pd.DataFrame(G.graph)
    df.index = df.columns

    return df


# Add Edges from API Call


# Example usage:


graph = Graph()


# Test Data
graph.add_edge("USD", "EUR", 0.85)
graph.add_edge("USD", "GBP", 0.73)
graph.add_edge("EUR", "GBP", 0.86)
graph.add_edge("GBP", "EUR", 1.16)
graph.add_edge("EUR", "JPY", 125.22)
graph.add_edge("JPY", "USD", 0.009)
graph.add_edge("GBP", "JPY", 140.22)

start_currency = "JPY"
end_currency = "GBP"

# Create a list of currency codes from the graph keys
currencies = list(graph.graph.keys())

# Initialize an empty DataFrame with the currency codes as both rows and columns
df = pd.DataFrame(index=currencies, columns=currencies)

# Fill the DataFrame with exchange rates from the graph
for from_currency, edges in graph.graph.items():
    for to_currency, exchange_rate in edges:
        df.at[from_currency, to_currency] = exchange_rate

# Fill diagonal with 1s (exchange rate from a currency to itself)
for currency in currencies:
    df.at[currency, currency] = 1.0

# Print the resulting DataFrame
print(df)

print()
print(create_adjacency_matrix(graph))
print()

path, best_exchange_rate = graph.find_best_exchange_path(
    start_currency, end_currency)
print("Best path:", " -> ".join(path))
print("Best exchange rate:", best_exchange_rate)
print()
