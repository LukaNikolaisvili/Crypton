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
import psycopg2
import os
import time
import krakenex
from pykrakenapi import KrakenAPI
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import configparser
import csv
import datetime as timenow



config = configparser.ConfigParser()
config.read('config.ini')

kraken_api_key = config['DEFAULT']['KRAKEN_API_KEY']
kraken_api_secret = config['DEFAULT']['KRAKEN_API_SECRET']
infura_project_id = config['DEFAULT']['INFURA_PROJECT_ID']
db_user = config['DEFAULT']['DB_USER']
db_pass = config['DEFAULT']['DB_PASS']
db_host = config['DEFAULT']['DB_HOST']
eht_price = config["DEFAULT"]['ETH_PRICE']

# Initialize session state variables at the beginning of your script
if "backgroundColor" not in st.session_state:
    st.session_state["backgroundColor"] = "#00152B"  # Default background color
if "text_color" not in st.session_state:
    st.session_state["text_color"] = "#FFFFFF"  # Default text color
if "language" not in st.session_state:
    st.session_state["language"] = "en"  # Default to English
if "is_authenticated" not in st.session_state:
    st.session_state["is_authenticated"] = False
if "user_info" not in st.session_state:
    st.session_state["user_info"] = {"username": "", "email": "", "picture": None}
if "last_active_time" not in st.session_state:
    st.session_state["last_active_time"] = time.time()

if time.time() - st.session_state["last_active_time"] > 15 * 60:  # 15 minutes
    # Perform logout or redirect to logout page
    st.experimental_rerun()
else:
    # Update last active time for session timeout
    st.session_state["last_active_time"] = time.time()


def check_activity_and_logout():
    current_time = time.time()
    inactive_duration = current_time - st.session_state["last_active_time"]
    if inactive_duration > 900:  # 15 minutes in seconds
        st.warning("You have been inactive for 15 minutes. Are you still here?")
        response = st.button("Yes, I'm still here")
        if not response:
            st.session_state.pop("is_authenticated", None)
            st.session_state.pop("user_info", None)
            st.experimental_rerun()


# Update last active time on any user interaction
def update_last_active_time():
    st.session_state["last_active_time"] = time.time()


# Initialize Web3
web3 = Web3(
    Web3.HTTPProvider(f"https://mainnet.infura.io/v3/{infura_project_id}")
)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Utility functions for Ethereum blockchain interaction
def get_wallet_balance(address):
    balance = web3.eth.get_balance(address)
    return web3.from_wei(balance, "ether")


def get_wallet_transactions(address, count=100):
    transactions = []
    block = web3.eth.get_block("latest")
    while len(transactions) < count and block["number"] > 0:
        for tx_hash in block["transactions"]:
            tx = web3.eth.get_transaction(tx_hash)
            if tx["from"] == address or tx["to"] == address:
                transactions.append(tx)
            if len(transactions) == count:
                break
        block = web3.eth.get_block(block["parentHash"])
    return transactions


def get_current_eth_price():
    url = eht_price
    response = requests.get(url)
    data = response.json()
    return data['ethereum']['usd']


def send_transaction(wallet_address, to_address, amount_eth, private_key):
    try:
        # Convert the provided amount of Ether to Wei
        value_in_wei = web3.to_wei(amount_eth, 'ether')
        
        # Check balance first to ensure sufficient funds are available
        balance = web3.eth.get_balance(wallet_address)
        gas_price = web3.eth.gas_price
        estimated_gas = 21000  # Typical gas limit for a simple transaction; adjust based on your needs
        
        # Check if balance covers the gas and the value
        total_cost = value_in_wei + (gas_price * estimated_gas)
        
        current_eth_price_usd = get_current_eth_price()

# Calculate USD value
        usd_value = (value_in_wei / 10**18) * current_eth_price_usd
        
       
        if balance < total_cost:
            return {'success': False, 'error': f'Insufficient funds for gas and value.{usd_value} USD'}

        # Create a transaction dictionary
        transaction = {
            'to': to_address,
            'value': value_in_wei,
            'gas': estimated_gas,
            'gasPrice': gas_price,
            'nonce': web3.eth.get_transaction_count(wallet_address),
        }

        # Sign the transaction using the private key
        signed_tx = web3.eth.account.sign_transaction(transaction, private_key)

        # Send the signed transaction and get the transaction hash
        tx_hash = web3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        return {'success': True, 'tx_hash': tx_hash.hex()}  # Convert bytes to hex string
    except Exception as e:
        error_message = str(e)
        if "insufficient funds" in error_message:
            error_message = "Insufficient funds for gas * price + value."
        return {'success': False, 'error': error_message}

if 'action' in st.session_state and st.session_state.action == "Send Ether":
    st.subheader("Send Ether")
    wallet_address = st.text_input("Your Wallet Address", help="Enter your Ethereum wallet address")
    to_address = st.text_input("Recipient's Address", help="Enter the recipient's Ethereum address")
    amount_eth = st.number_input("Amount (ETH)", min_value=0.01, step=0.01, format="%.6f")
    private_key = st.text_input("Your Private Key", type="password", help="Enter your private key for transaction")

    if st.button("Send Ether"):
        if not wallet_address or not to_address or amount_eth <= 0 or not private_key:
            st.error("Please provide a valid wallet address, recipient address, amount, and your private key.")
        else:
            transaction_result = send_transaction(wallet_address, to_address, amount_eth, private_key)
            if transaction_result['success']:
                st.success(f"Transaction sent successfully! Transaction Hash: {transaction_result['tx_hash']}")
            else:
                st.error(f"Transaction failed: {transaction_result['error']} Please check your inputs and retry.")
                
                
# Initialize API
api = krakenex.API(
    key=kraken_api_key,
    secret=kraken_api_secret,
)
kraken = KrakenAPI(api)


def sell_ethereum(amount, price="market"):
    """Sell Ethereum for USD at market price or a specified limit price."""
    try:
        if price == "market":
            response = kraken.add_standard_order(
                pair="XETHZUSD", type="sell", ordertype="market", volume=amount
            )
        else:
            response = kraken.add_standard_order(
                pair="XETHZUSD",
                type="sell",
                ordertype="limit",
                price=price,
                volume=amount,
            )
        print("Sell order response:", response)
    except Exception as e:
        print("Failed to place sell order:", e)


def withdraw_fiat(amount, currency="USD", key="your_bank_account_key"):
    """Withdraw fiat to a linked bank account."""
    try:
        response = kraken.withdraw_funds(asset=currency, key=key, amount=amount)
        print("Withdrawal response:", response)
    except Exception as e:
        print("Failed to initiate withdrawal:", e)


# Function to connect to PostgreSQL database
def connect_to_db():
    conn = None
    try:
        conn = psycopg2.connect(
            dbname="initial_db",
            user= db_user,
            password= db_pass,
            host= db_host,
            port="5432",
        )
        st.success("Connected to PostgreSQL database!")
    except Exception as e:
        st.error(f"Error connecting to PostgreSQL database: {e}")
    return conn


def authenticate_user(username, password):
    conn = connect_to_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, username, email, picture FROM users WHERE username = %s AND password = %s",
                (username, password),
            )
            user = cursor.fetchone()
            if user:
                st.session_state["is_authenticated"] = True
                st.session_state["user_info"] = {
                    "username": user[1],
                    "email": user[2],
                    "picture": user[3] if user[3] else None,
                }
                st.session_state["last_active_time"] = (
                    time.time()
                )  # Store the current time
                st.success("Login successful. Redirecting to the dashboard...")
            else:
                st.error("Invalid username or password.")
        except Exception as e:
            st.error(f"Error authenticating user: {e}")
        finally:
            cursor.close()
            conn.close()


def safe_run(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

    return wrapper


@safe_run
def update_user_profile(username, new_username, new_email, profile_pic):
    conn = connect_to_db()
    try:
        cursor = conn.cursor()
        if profile_pic:
            # Convert picture to a binary format suitable for BYTEA
            profile_pic_binary = profile_pic.getvalue()
            cursor.execute(
                "UPDATE users SET username=%s, email=%s, picture=%s WHERE username=%s",
                (
                    new_username,
                    new_email,
                    psycopg2.Binary(profile_pic_binary),
                    username,
                ),
            )
        else:
            cursor.execute(
                "UPDATE users SET username=%s, email=%s WHERE username=%s",
                (new_username, new_email, username),
            )
        conn.commit()
        return True
    except Exception as e:
        st.error(f"Failed to update profile: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


def user_profile_page():
    st.title("User Profile")
    current_username = st.session_state["user_info"]["username"]
    profile_pic = st.file_uploader("Upload a profile picture", type=["jpg", "png"])
    new_username = st.text_input(
        "Change Username", st.session_state["user_info"]["username"]
    )
    new_email = st.text_input("Change Email", st.session_state["user_info"]["email"])

    if st.button("Update Profile"):
        success = update_user_profile(
            current_username, new_username, new_email, profile_pic
        )
        if success:
            st.session_state["user_info"]["username"] = new_username
            st.session_state["user_info"]["email"] = new_email
            if profile_pic:
                st.session_state["user_info"]["profile_pic"] = profile_pic.getvalue()
            st.success("Profile updated successfully!")
        else:
            st.error("Failed to update profile.")


# Function to register a new user in the database
def register_user(email, username, password):
    conn = connect_to_db()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (email, username, password) VALUES (%s, %s, %s)",
                (email, username, password),
            )
            conn.commit()
            st.success("Registration successful. You can now login.")
            st.session_state["is_authenticated"] = True
        except Exception as e:
            st.error(f"Error registering user: {e}")
        finally:
            cursor.close()
            conn.close()


# Registration form
def register():
    st.title("Registration")
    email = st.text_input("Email")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Register"):
        if password != confirm_password:
            st.error("Passwords do not match.")
        else:
            register_user(email, username, password)


# Login form
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        authenticate_user(username, password)


# Check authentication state before displaying content
if not st.session_state.get("is_authenticated", False):
    st.title("Authentication Required")
    auth_mode = st.radio("Choose authentication option", ["Login", "Register"])
    if auth_mode == "Login":
        login()
    elif auth_mode == "Register":
        register()
else:
    # Display Advanced User Profile and Logout option
    st.sidebar.title("User Profile")
    st.sidebar.write(f"Logged in as: {st.session_state['user_info']['username']}")

    # Display the user's picture if available
    if (
        "picture" in st.session_state["user_info"]
        and st.session_state["user_info"]["picture"] is not None
    ):
        # Convert binary data to a displayable format
        picture_data = st.session_state["user_info"]["picture"]
        st.sidebar.image(
            picture_data, caption="Profile Picture", width=150, output_format="PNG"
        )
    else:
        st.sidebar.write("No profile picture available.")

    logout = st.sidebar.button("Logout")
    if logout:
        # Clear all session state related to the user
        del st.session_state["is_authenticated"]
        del st.session_state["user_info"]
        st.experimental_rerun()

    def get_latest_block_number():
        return web3.eth.block_number

    def get_network_statistics():
        block_number = web3.eth.block_number
        last_block = web3.eth.get_block(block_number)
        timestamp_last_block = last_block.timestamp
        block_time = (
            timestamp_last_block - web3.eth.get_block(block_number - 50).timestamp
        )
        transactions_last_hour = web3.eth.get_block_transaction_count(
            block_number
        ) - web3.eth.get_block_transaction_count(block_number - 3600 // block_time)
        return block_time, transactions_last_hour

    def get_blockchain_statistics():
        latest = web3.eth.get_block("latest")
        block_time = latest.timestamp - web3.eth.get_block(latest.number - 1).timestamp

        # Sum the number of transactions in the last 10 blocks
        total_transactions = sum(
            len(web3.eth.get_block(i).transactions)
            for i in range(latest.number - 9, latest.number + 1)
        )

        avg_gas_price = web3.eth.gas_price
        difficulty = latest.difficulty
        hash_rate = difficulty / block_time if block_time > 0 else 0

        return {
            "block_time": block_time,
            "transactions_last_10_blocks": total_transactions,
            "average_gas_price": avg_gas_price,
            "difficulty": difficulty,
            "hash_rate": hash_rate,
        }

    def get_token_information(token_address):
        response = requests.get(
            f"https://api.ethplorer.io/getTokenInfo/{token_address}?apiKey=freekey"
        )
        if response.status_code == 200:
            data = response.json()
            return data

    def get_transaction(tx_hash):
        return web3.eth.get_transaction(tx_hash)

    def get_eth_to_currency_rate(currency):
        response = requests.get(
            f"https://api.coingecko.com/api/v3/simple/price?ids=ethereum&vs_currencies={currency}"
        )
        if response.status_code == 200:
            data = response.json()
            return data["ethereum"].get(currency.lower())

    def train_predict_eth_prices():
        # Fetch historical data
        df = get_ethereum_historical_prices()
        df["Previous_Close"] = df["Close"].shift(
            1
        )  # Create a feature column with previous day's close prices
        df = df.dropna()  # Drop the first row as it now contains NaN

        # Define features and target
        X = df[["Previous_Close"]]  # features
        y = df["Close"]  # target

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Create and train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred_test = model.predict(X_test)

        # Calculate MSE and R-squared for the test set
        mse = mean_squared_error(y_test, y_pred_test)
        r2 = r2_score(y_test, y_pred_test)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_test)

        # Predict using the latest data point
        latest_price = df.iloc[-1]["Previous_Close"]
        predicted_next_price = model.predict([[latest_price]])
        

        print(
            f"Predicted next close price of Ethereum: {predicted_next_price[0]:.2f} USD"
            
        )
        print(f"Mean Squared Error (MSE) on test set: {mse:.2f}")
        print(f"R-squared Value: {r2:.2f}")
        print(f"Mean Absolute Error (MAE): {mae:.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        latest_timestamp = df.iloc[-1]["Timestamp"]
        
        with open('predictedEthPrices.csv', 'w', newline='') as csvfile:
            fieldnames = ['TIME','PREDICTED ETH PRICES']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'TIME':timenow.datetime.now(),'PREDICTED ETH PRICES': predicted_next_price[0]})
        
        return predicted_next_price[0]

    @safe_run
    def get_transaction_safe(tx_hash):
        transaction = get_transaction(tx_hash)
        if transaction:
            value_eth = web3.from_wei(transaction["value"], "ether")
            rate_usd = get_eth_to_currency_rate("USD")
            rate_eur = get_eth_to_currency_rate("EUR")
            if rate_usd:
                value_usd = decimal.Decimal(value_eth) * decimal.Decimal(rate_usd)
            if rate_eur:
                value_eur = decimal.Decimal(value_eth) * decimal.Decimal(rate_eur)
                enhanced_transaction = {
                    "hash": transaction["hash"],
                    "from": transaction["from"],
                    "to": transaction["to"],
                    "value_eth": f"{value_eth} ETH",
                    "value_usd": f"${value_usd:.2f} USD" if rate_usd else "N/A",
                    "value_eur": f"${value_eur:.2f} EUR " if rate_eur else "N/A",
                }
                return enhanced_transaction
        return None

    translator = Translator()

    def translate_text(text, dest_language):
        if dest_language in LANGUAGES:
            return translator.translate(text, dest=dest_language).text
        return text

    # Function to fetch historical Ethereum prices
    def get_ethereum_historical_prices(
        exchange="binance", symbol="ETH/USDT", timeframe="1d"
    ):
        # Initialize the exchange
        exchange = getattr(ccxt, exchange)()
        # Load historical OHLCV data
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe)
        # Convert the data into a Pandas DataFrame
        df = pd.DataFrame(
            ohlcv, columns=["Timestamp", "Open", "High", "Low", "Close", "Volume"]
        )
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], unit="ms")
        return df

    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose a section",
        [
            "Profile",
            "Home",
            "Blockchain Info",
            "My_Wallet",
            "Transaction Details",
            "Settings",
        ],
    )

    if app_mode == "Home":
        st.title("Welcome to Crypton")
        if web3.is_connected():
            st.success("Connected to Ethereum blockchain!")

            # Call the function to fetch historical prices and display them
            ethereum_prices = get_ethereum_historical_prices()
            if ethereum_prices is not None:
                
                ethereum_prices = get_ethereum_historical_prices()

                # Convert timestamp to datetime for Altair plot
                ethereum_prices["Timestamp"] = pd.to_datetime(
                    ethereum_prices["Timestamp"], unit="ms"
                )

                # Create Altair chart
                chart = (
                    alt.Chart(ethereum_prices)
                    .mark_line()
                    .encode(x="Timestamp", y="Close", tooltip=["Timestamp", "Close"])
                    .properties(
                        width=800,
                        height=400,
                        background=st.session_state["backgroundColor"],
                    )
                    .interactive()
                )

                st.info("Historical Ethereum Prices")
                st.altair_chart(chart)

                # Example usage:
                ethereum_prices = get_ethereum_historical_prices()
                print(ethereum_prices)
            else:
                st.error("Failed to fetch historical Ethereum prices.")

        else:
            st.error("Failed to connect to Ethereum blockchain.")
        st.markdown("Select options from the sidebar to navigate through the app.")
        predicted_price = train_predict_eth_prices()
        st.info(f"Predicted next close price of Ethereum: {predicted_price:.2f} USD")

    elif app_mode == "Blockchain Info":
        st.title("Blockchain Information")
        stats = get_blockchain_statistics()
        st.write(f"Block Time: {stats['block_time']} seconds")
        st.write(
            f"Transactions in Last 10 Blocks: {stats['transactions_last_10_blocks']}"
        )
        st.write(f"Average Gas Price: {stats['average_gas_price']} Wei")
        st.write(f"Current Mining Difficulty: {stats['difficulty']}")
        st.write(f"Network Hash Rate: {stats['hash_rate']} hashes/second")

        if st.button("Get Latest Block Number"):
            block_number = get_latest_block_number()
            st.write(f"Latest Block Number: {block_number}")

    elif app_mode == "Transaction Details":
        st.title("Transaction Details")
        tx_hash = st.text_input("Enter transaction hash to get details:")
        if tx_hash:
            transaction = get_transaction_safe(tx_hash)
            if transaction:
                st.json(transaction)
            else:
                st.error("Failed to retrieve transaction details or invalid hash.")

    elif app_mode == "Settings":

        # Load config file
        config = configparser.ConfigParser()
        config.optionxform = str  # Preserve the case of configuration keys
        config.read(".streamlit/config.toml")

        # Settings
        st.title("Settings")

        # Retrieve background color from config and ensure it's in the correct format
        bg_color_config = config["theme"]["backgroundColor"]
        bg_color_default = (
            bg_color_config if bg_color_config.startswith("#") else "#FFFFFF"
        )

        text_color_config = config["theme"]["textColor"]
        text_color_default = (
            text_color_config if text_color_config.startswith("#") else "#FFFFFF"
        )

        # Background Color Selection
        bg_color = st.color_picker("Select Background Color", bg_color_default)
        st.write(f"You selected background color: {bg_color}")

        text_color = st.color_picker("Select text Color", text_color_default)
        st.write(f"You selected background color: {text_color}")

        # Language Selection
        language = st.selectbox(
            "Select Language",
            ["English", "Spanish", "French"],
            index=st.session_state.get("lang_index", 0),
        )
        st.write(f"You selected language: {language}")

        # Apply Settings Button
        if st.button("Apply Settings"):
            # Update config with new background color
            config["theme"][
                "backgroundColor"
            ] = f'"{bg_color}"'  # Enclose bg_color in double quotes
            config["theme"][
                "textColor"
            ] = f'"{text_color}"'  # Enclose bg_color in double quotes
            config["theme"]["language"] = f'"{language}"'

            with open(".streamlit/config.toml", "w") as configfile:
                config.write(configfile)

            # Display a success message
            st.success("Settings Applied Successfully!")

            st.experimental_rerun()

        if "background_color" not in st.session_state:
            st.session_state["background_color"] = "#00152B"  # Default value
        if "text_color" not in st.session_state:
            st.session_state["text_color"] = "#FFFFFF"  # Default value
        if "language" not in st.session_state:
            st.session_state["language"] = "en"  # Default language

        if st.button("reset to defaults"):
            config["theme"][
                "backgroundColor"
            ] = f'"{"#00152B"}"'  # Enclose bg_color in double quotes
            config["theme"][
                "textColor"
            ] = f'"{"#FFF"}"'  # Enclose bg_color in double quotes

            config["theme"]["language"] = f'"{language}"'
            with open(".streamlit/config.toml", "w") as configfile:
                config.write(configfile)

                # Display a success message
                st.success("Reseted to default settings Successfully!")
                import threading as thread

                st.rerun()

    elif app_mode == "Profile":
        user_profile_page()

    def app_mode_my_wallet():
        st.title("My Wallet")
        st.write(f"Welcome, {st.session_state['user_info']['username']}!")

        wallet_address = st.text_input("Enter your wallet address")
        if wallet_address:
            balance = get_wallet_balance(wallet_address)
            st.write(f"Your current balance: {balance} ETH")

            action = st.radio("Select Action:", ["Send Ether", "Bank Withdrawal"])

            if action == "Send Ether":
                st.subheader("Send Ether")
                to_address = st.text_input(
                    "Recipient's Address", help="Enter the recipient's Ethereum address"
                )
                amount_eth = st.number_input(
                    "Amount (ETH)", min_value=0.0, format="%.6f"
                )
                private_key = st.text_input(
                    "Your Private Key",
                    type="password",
                    help="Enter your private key for transaction",
                )

                if st.button("Send Ether"):
                    if not to_address or amount_eth <= 0:
                        st.warning(
                            "Please provide a valid recipient address and amount."
                        )
                    elif not private_key:
                        st.warning("Please enter your private key.")
                    else:
                        try:
                            tx_hash = send_transaction(wallet_address, to_address, amount_eth, private_key)
                            st.success(f"Transaction sent successfully! Transaction Hash: {tx_hash}")
                        except Exception as e:
                            st.error(f"Transaction failed: {str(e)}")

            elif action == "Bank Withdrawal":
                st.subheader("Bank Withdrawal")
                eth_amount = st.number_input(
                    "Enter the amount of Ethereum to sell:",
                    min_value=0.0,
                    format="%.4f",
                    key="eth_amount",
                )
                withdraw_amount = st.number_input(
                    "Enter USD amount to withdraw to bank:",
                    min_value=0.0,
                    format="%.2f",
                    key="withdraw_amount",
                )
                bank_account_key = st.text_input(
                    "Enter your bank account key for withdrawal:",
                    key="bank_account_key",
                )

                if st.button("Sell Ethereum", key="sell_ethereum"):
                    sell_ethereum(eth_amount)
                if st.button("Withdraw USD", key="withdraw_usd"):
                    withdraw_fiat(withdraw_amount, "USD", bank_account_key)

            st.subheader("Transaction History")
            transactions = get_wallet_transactions(wallet_address)
            if transactions:
                for tx in transactions:
                    st.write(
                        f"From: {tx['from']} | To: {tx['to']} | Value: {web3.from_wei(tx['value'], 'ether')} ETH"
                    )
            else:
                st.info("No transactions found for this wallet address.")

        else:
            st.info(
                "Please enter your wallet address to view balance and perform transactions."
            )

    def perform_bank_withdrawal():
        eth_amount = st.number_input(
            "Enter the amount of Ethereum to sell:",
            min_value=0.0,
            format="%.4f",
            key="eth_amount",
        )
        withdraw_amount = st.number_input(
            "Enter USD amount to withdraw to bank:",
            min_value=0.0,
            format="%.2f",
            key="withdraw_amount",
        )
        bank_account_key = st.text_input(
            "Enter your bank account key for withdrawal:", key="bank_account_key"
        )

        if st.button("Sell Ethereum", key="sell_ethereum"):
            sell_ethereum(eth_amount)
        if st.button("Withdraw USD", key="withdraw_usd"):
            withdraw_fiat(withdraw_amount, "USD", bank_account_key)

    # Main app control flow
    if app_mode == "My_Wallet":
        app_mode_my_wallet()
