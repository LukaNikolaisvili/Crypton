import streamlit as st
from web3 import Web3
import logging
import requests
import decimal
import pandas as pd
import ccxt
import altair as alt

# Initialize Web3
web3 = Web3(Web3.HTTPProvider('https://mainnet.infura.io/v3/0ce4b7eb2c8649ff8e0f62708735089f'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_latest_block_number():
    return web3.eth.block_number

def get_network_statistics():
    block_number = web3.eth.block_number
    last_block = web3.eth.get_block(block_number)
    timestamp_last_block = last_block.timestamp
    block_time = timestamp_last_block - web3.eth.get_block(block_number - 50).timestamp
    transactions_last_hour = web3.eth.get_block_transaction_count(block_number) - web3.eth.get_block_transaction_count(block_number - 3600 // block_time)
    return block_time, transactions_last_hour

def get_token_information(token_address):
    response = requests.get(f'https://api.ethplorer.io/getTokenInfo/{token_address}?apiKey=freekey')
    if response.status_code == 200:
        data = response.json()
        return data

def get_transaction(tx_hash):
    return web3.eth.get_transaction(tx_hash)

def get_eth_to_currency_rate(currency):
    response = requests.get(f'https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies={currency}')
    if response.status_code == 200:
        data = response.json()
        return data['ethereum'].get(currency.lower())

def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None
    return wrapper

@safe_run
def get_transaction_safe(tx_hash):
    transaction = get_transaction(tx_hash)
    if transaction:
        value_eth = web3.from_wei(transaction['value'], 'ether')
        rate_usd = get_eth_to_currency_rate('USD')
        rate_eur = get_eth_to_currency_rate('EUR')
        if rate_usd:
            value_usd = decimal.Decimal(value_eth) * decimal.Decimal(rate_usd)
        if rate_eur:
            value_eur = decimal.Decimal(value_eth) * decimal.Decimal(rate_eur)
            enhanced_transaction = {
                'hash': transaction['hash'],
                'from': transaction['from'],
                'to': transaction['to'],
                'value_eth': f"{value_eth} ETH",
                'value_usd': f"${value_usd:.2f} USD" if rate_usd else "N/A",
                'value_eur': f"${value_eur:.2f} EUR " if rate_eur else "N/A"
            }
            return enhanced_transaction
    return None



# Function to fetch historical Ethereum prices
def get_ethereum_historical_prices(exchange='binance', symbol='ETH/USDT', timeframe='1d'):
    # Initialize the exchange
    exchange = getattr(ccxt, exchange)()
    # Load historical OHLCV data
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
    # Convert the data into a Pandas DataFrame
    df = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    return df





st.sidebar.title('Navigation')
app_mode = st.sidebar.selectbox('Choose the app section',
                                ['Home', 'Blockchain Info', 'Transaction Details', 'Settings'])

if app_mode == 'Home':
    st.title('Welcome to Crypton')
    if web3.is_connected():
        st.success("Connected to Ethereum blockchain!")
        
        # Call the function to fetch historical prices and display them
        ethereum_prices = get_ethereum_historical_prices()
        if ethereum_prices is not None:
            st.write("Historical Ethereum Prices:")
            ethereum_prices = get_ethereum_historical_prices()

# Convert timestamp to datetime for Altair plot
            ethereum_prices['Timestamp'] = pd.to_datetime(ethereum_prices['Timestamp'], unit='ms')

            # Create Altair chart
            chart = alt.Chart(ethereum_prices).mark_line().encode(
                x='Timestamp',
                y='Close',
                tooltip=['Timestamp', 'Close']
            ).properties(
                width=800,
                height=400
            ).interactive()

            st.title('Historical Ethereum Prices')
            st.altair_chart(chart)

            # Example usage:
            ethereum_prices = get_ethereum_historical_prices()
            print(ethereum_prices)
        else:
            st.error("Failed to fetch historical Ethereum prices.")
            
    else:
        st.error("Failed to connect to Ethereum blockchain.")
    st.markdown("Select options from the sidebar to navigate through the app.")

elif app_mode == 'Blockchain Info':
    st.title('Blockchain Information')
    block_time, transactions_last_hour = get_network_statistics()
    st.write(f"Block Time: {block_time} seconds")
    st.write(f"Transactions in the Last Hour: {transactions_last_hour}")
    
    if st.button('Get Latest Block Number'):
        block_number = get_latest_block_number()
        st.write(f"Latest Block Number: {block_number}")

elif app_mode == 'Transaction Details':
    st.title('Transaction Details')
    tx_hash = st.text_input('Enter transaction hash to get details:')
    if tx_hash:
        transaction = get_transaction_safe(tx_hash)
        if transaction:
            st.json(transaction)
        else:
            st.error("Failed to retrieve transaction details or invalid hash.")


elif app_mode == 'Settings':
    st.title('Settings')
    
    # Background Color Selection
    bg_color = st.color_picker('Select Background Color', '#FFFFFF')
    st.write(f'You selected background color: {bg_color}')
    
    # Language Selection
    language = st.selectbox('Select Language', ['English', 'Spanish', 'French'])
    st.write(f'You selected language: {language}')
    
    # Apply Settings Button
    if st.button('Apply Settings'):
        st.session_state.language = language
        st.markdown("----")
        st.markdown("## Settings Applied Successfully!")