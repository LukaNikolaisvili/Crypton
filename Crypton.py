import streamlit as st
from web3 import Web3
import logging
import requests
import decimal
import pandas as pd
import ccxt
import altair as alt
import configparser
from googletrans import Translator, LANGUAGES


# Initialize session state variables at the beginning of your script
if 'backgroundColor' not in st.session_state:
    st.session_state['backgroundColor'] = '#00152B'  # Default background color
if 'text_color' not in st.session_state:
    st.session_state['text_color'] = '#FFFFFF'  # Default text color
if 'language' not in st.session_state:
    st.session_state['language'] = 'en'  # Default to English

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


def get_blockchain_statistics():
    latest = web3.eth.get_block('latest')
    block_time = latest.timestamp - web3.eth.get_block(latest.number - 1).timestamp

    # Sum the number of transactions in the last 10 blocks
    total_transactions = sum(len(web3.eth.get_block(i).transactions) for i in range(latest.number - 9, latest.number + 1))

    avg_gas_price = web3.eth.gas_price
    difficulty = latest.difficulty
    hash_rate = difficulty / block_time if block_time > 0 else 0

    return {
        "block_time": block_time,
        "transactions_last_10_blocks": total_transactions,
        "average_gas_price": avg_gas_price,
        "difficulty": difficulty,
        "hash_rate": hash_rate
    }

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

translator = Translator()

def translate_text(text, dest_language):
    if dest_language in LANGUAGES:
        return translator.translate(text, dest=dest_language).text
    return text

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
                height=400,
                background=st.session_state['backgroundColor']
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
    stats = get_blockchain_statistics()
    st.write(f"Block Time: {stats['block_time']} seconds")
    st.write(f"Transactions in Last 10 Blocks: {stats['transactions_last_10_blocks']}")
    st.write(f"Average Gas Price: {stats['average_gas_price']} Wei")
    st.write(f"Current Mining Difficulty: {stats['difficulty']}")
    st.write(f"Network Hash Rate: {stats['hash_rate']} hashes/second")
    
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
    
    
   # Load config file
    config = configparser.ConfigParser()
    config.optionxform = str  # Preserve the case of configuration keys
    config.read('.streamlit/config.toml')

    # Settings
    st.title('Settings')

    # Retrieve background color from config and ensure it's in the correct format
    bg_color_config = config['theme']['backgroundColor']
    bg_color_default = bg_color_config if bg_color_config.startswith('#') else '#FFFFFF'
    
    text_color_config = config['theme']['textColor']
    text_color_default = text_color_config if text_color_config.startswith('#') else '#FFFFFF'

    # Background Color Selection
    bg_color = st.color_picker('Select Background Color', bg_color_default)
    st.write(f'You selected background color: {bg_color}')

    text_color = st.color_picker('Select text Color', text_color_default)
    st.write(f'You selected background color: {text_color}')

    
    # Language Selection
    language = st.selectbox('Select Language', ['English', 'Spanish', 'French'], index=st.session_state.get('lang_index', 0))
    st.write(f'You selected language: {language}')

    # Apply Settings Button
    if st.button('Apply Settings'):
        # Update config with new background color
        config['theme']['backgroundColor'] = f'"{bg_color}"'  # Enclose bg_color in double quotes
        config['theme']['textColor'] = f'"{text_color}"'  # Enclose bg_color in double quotes
        config['theme']['language'] = f'"{language}"'
        
        with open('.streamlit/config.toml', 'w') as configfile:
            config.write(configfile)
            
        # Display a success message
        st.success("Settings Applied Successfully!")
        
        st.experimental_rerun()
    
    if 'background_color' not in st.session_state:
        st.session_state['background_color'] = '#00152B'  # Default value
    if 'text_color' not in st.session_state:
        st.session_state['text_color'] = '#FFFFFF'  # Default value
    if 'language' not in st.session_state:
        st.session_state['language'] = 'en'  # Default language

        
    if(st.button('reset to defaults')):
        config['theme']['backgroundColor'] = f'"{"#00152B"}"'  # Enclose bg_color in double quotes
        config['theme']['textColor'] = f'"{"#FFF"}"'  # Enclose bg_color in double quotes
        
        config['theme']['language'] = f'"{language}"'
        with open('.streamlit/config.toml', 'w') as configfile:
            config.write(configfile)
        
        # Display a success message
            st.success("Reseted to default settings Successfully!")
            import threading as thread
            
            st.rerun()