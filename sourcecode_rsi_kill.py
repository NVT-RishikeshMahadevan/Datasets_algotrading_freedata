import pandas as pd
from datetime import datetime
import pytz
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


url = 'https://raw.githubusercontent.com/NVT-RishikeshMahadevan/Datasets_algotrading_freedata/refs/heads/rishi/SPY.csv'
df=pd.read_csv(url)
# Convert the timestamp column to datetime and set timezone to UTC
df['datetime'] = pd.to_datetime(df['t'], unit='ms', utc=True)

# Convert datetime to EST
est = pytz.timezone('US/Eastern')
df['datetime_est'] = df['datetime'].dt.tz_convert(est)

#######RSI##########
# Compute RSI using the closing prices
window_length = 14  # Standard RSI period

# Calculate price differences
delta = df['c'].diff()

# Calculate gains (positive deltas) and losses (negative deltas)
gains = delta.where(delta > 0, 0)
losses = -delta.where(delta < 0, 0)

# Compute average gains and losses for the first RSI window
avg_gain = gains.rolling(window=window_length, min_periods=window_length).mean()
avg_loss = losses.rolling(window=window_length, min_periods=window_length).mean()

# Calculate the Relative Strength (RS) and RSI
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

# Add RSI to the DataFrame
df['RSI'] = rsi

# Leave the first 14 rows of RSI empty as requested
df.loc[:window_length - 1, 'RSI'] = None


# Filter rows for time between 9:30 AM and 4:00 PM EST
df['time_est'] = df['datetime_est'].dt.time  # Extract time in EST
start_time = datetime.strptime("09:30:00", "%H:%M:%S").time()
end_time = datetime.strptime("16:00:00", "%H:%M:%S").time()

df['date'] = df['datetime_est'].dt.date
df['date'] = pd.to_datetime(df['date'])

df1=df[(df['time_est'] >= start_time) & (df['time_est'] <= end_time)]

df1 = df1.reset_index(drop=True)

######### Portfolio Management #############3

# Constants
initial_capital = 1000000  # Starting capital
trade_fraction = 0.25  # Fraction of cash to use per trade
stoploss_percent = 0.0005  # Stoploss as a percentage of initial capital
kill_switch = False  # Default value for kill switch

# Initialize tracking columns
df1['cash'] = initial_capital  # Cash balance
df1['shares'] = 0  # Number of shares held
df1['portfolio_value'] = initial_capital  # Total portfolio value
df1['signal'] = ''  # Signal column for Buy ('B') or Sell ('S')
df1['stoploss'] = 0  # Stoploss flag (1 if stoploss hit, 0 otherwise)

# Initialize variables
current_cash = initial_capital
current_shares = 0
stoploss_hit = False
prev_date = None

# Stoploss amount
stoploss_limit = initial_capital * stoploss_percent

# Initialize prev_portfolio_value
prev_portfolio_value = initial_capital  # This should start as the initial capital

# Variable to track the date when kill switch is triggered
kill_switch_triggered_date = None

# Loop through the DataFrame with progress tracking
total_rows = len(df1)
skip_trading_today = False  # Flag to indicate whether to skip trading for the day

for i, row in tqdm(df1.iterrows(), total=total_rows, desc="Processing rows"):
    price = row['c']  # Closing price for the bar
    rsi = row['RSI']  # RSI value
    current_date = row['date']  # Trading date

    # Check if we need to reset the kill switch for the new day
    if kill_switch_triggered_date and current_date != kill_switch_triggered_date:
        skip_trading_today = False  # Allow trading to resume the next day
        stoploss_hit = False  # Reset stoploss flag for the new day
        prev_portfolio_value = current_cash + current_shares * price  # Reset portfolio value for new day
        kill_switch_triggered_date = None  # Clear the kill switch trigger date

    # Calculate current portfolio value (Cash + Shares)
    portfolio_value = current_cash + current_shares * price

    # Check stoploss based on portfolio value difference
    if not stoploss_hit and portfolio_value - prev_portfolio_value < -stoploss_limit:
        stoploss_hit = True
        df1.loc[i, 'stoploss'] = 1  # Mark stoploss hit
        current_cash += current_shares * price  # Liquidate all shares
        current_shares = 0

        if kill_switch:  # If kill switch is True, stop trading for the day
            skip_trading_today = True
            kill_switch_triggered_date = current_date  # Set kill switch for today
            df1.loc[i, 'cash'] = current_cash
            df1.loc[i, 'shares'] = current_shares
            df1.loc[i, 'portfolio_value'] = current_cash + current_shares * price  # Update portfolio value correctly
            prev_portfolio_value = current_cash  # Manually set prev_portfolio_value to current cash
            prev_date = current_date
            continue

        # If kill_switch is False, we exit the position but continue to the next signal
        else:
            df1.loc[i, 'cash'] = current_cash
            df1.loc[i, 'shares'] = current_shares
            df1.loc[i, 'portfolio_value'] = current_cash + current_shares * price
            prev_portfolio_value = current_cash  # Set prev_portfolio_value to current cash
            continue  # Continue to the next signal

    # Buy logic: RSI < 30
    if not stoploss_hit and rsi < 30:
        trade_cash = initial_capital * trade_fraction  # Maximum cash to use for this trade
        shares_to_buy = trade_cash // price  # Calculate how many shares to buy
        cost = shares_to_buy * price  # Cost of buying these shares

        # Execute the trade if cash is sufficient
        if current_cash >= cost and shares_to_buy > 0:
            current_cash -= cost
            current_shares += shares_to_buy
            df1.loc[i, 'signal'] = 'B'  # Mark as a buy signal

    # Sell logic: RSI > 70
    if not stoploss_hit and rsi > 70 and current_shares > 0:  # Sell only if shares are held
        current_cash += current_shares * price  # Sell all shares
        current_shares = 0
        df1.loc[i, 'signal'] = 'S'  # Mark as a sell signal

    # At the end of the day, sell all remaining shares
    if i == total_rows - 1 or df1.loc[i + 1, 'date'] != current_date:
        current_cash += current_shares * price
        current_shares = 0
        df1.loc[i, 'signal'] = 'S'  # Mark as a sell signal for the last minute

    # Update tracking columns with the correct portfolio value
    df1.loc[i, 'cash'] = current_cash
    df1.loc[i, 'shares'] = current_shares
    df1.loc[i, 'portfolio_value'] = current_cash + current_shares * price  # Correct portfolio value after liquidation

    # Update prev_portfolio_value after each iteration, not after the stoploss hit
    prev_portfolio_value = current_cash + current_shares * price

    if kill_switch == False:
        stoploss_hit = False

    # Update previous date
    prev_date = current_date


df2 = df1.groupby('date').tail(1)[['date', 'portfolio_value','c']].reset_index(drop=True)

######### Figures #########
# Create the figure and axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot the portfolio value on the first y-axis
ax1.plot(df2['date'], df2['portfolio_value'], color='blue', label='Portfolio Value', linestyle='-', linewidth=2)
ax1.set_xlabel('Date')
ax1.set_ylabel('Portfolio Value', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis to plot stock price close
ax2 = ax1.twinx()
ax2.plot(df2['date'], df2['c'], color='red', label='Stock Close Price', linestyle='-', linewidth=2)
ax2.set_ylabel('Stock Close Price', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and show the plot
plt.title('Portfolio Value and Stock Close Price')
fig.tight_layout()  # To make sure everything fits without overlap
plt.show()

import matplotlib.pyplot as plt

# Specify the date to filter the data
specific_date = "2023-09-27"
filtered_df = df1[df1['date'] == specific_date]

# Extract time and relevant data
time = filtered_df['datetime_est']
portfolio_value = filtered_df['portfolio_value']
stock_price = filtered_df['c']
rsi = filtered_df['RSI']
buy_signals = filtered_df[filtered_df['signal'] == 'B']
sell_signals = filtered_df[filtered_df['signal'] == 'S']
cash = filtered_df['cash']
shares = filtered_df['shares']

# Calculate portfolio allocation: cash and shares (share value = shares * stock price)
cash_allocation = cash
share_value = shares * stock_price

# Create the figure and axes
fig, axs = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={'height_ratios': [3, 1, 1]})
plt.subplots_adjust(hspace=0.3)

# Plot Portfolio Value and Stock Price
ax1 = axs[0]
ax1.plot(time, portfolio_value, label="Portfolio Value", color="blue", linewidth=2)
ax1.set_ylabel("Portfolio Value ($)", color="blue")
ax1.tick_params(axis='y', labelcolor="blue")
ax1.legend(loc="upper left")

# Create a twin y-axis for stock price
ax2 = ax1.twinx()
ax2.plot(time, stock_price, label="Stock Price", color="orange", linewidth=2)
ax2.scatter(buy_signals['datetime_est'], buy_signals['c'], label="Buy Signal", color="green", marker="^", s=100)
ax2.scatter(sell_signals['datetime_est'], sell_signals['c'], label="Sell Signal", color="red", marker="v", s=100)

# Plot blue square markers for stoploss
stoploss_hits = filtered_df[filtered_df['stoploss'] == 1]
ax2.scatter(stoploss_hits['datetime_est'], stoploss_hits['c'], label="Stoploss Hit", color="blue", marker="s", s=100)

ax2.set_ylabel("Stock Price ($)", color="orange")
ax2.tick_params(axis='y', labelcolor="orange")
ax2.legend(loc="upper right")

# Title for the first plot
ax1.set_title(f"Portfolio Value and Stock Price on {specific_date}")

# Plot RSI
ax3 = axs[1]
ax3.plot(time, rsi, label="RSI", color="purple", linewidth=2)
ax3.axhline(30, color="green", linestyle="--", linewidth=1, label="RSI Oversold (30)")
ax3.axhline(70, color="red", linestyle="--", linewidth=1, label="RSI Overbought (70)")
ax3.set_ylabel("RSI")
ax3.set_xlabel("Time")
ax3.legend(loc="upper right")
ax3.set_title(f"RSI on {specific_date}")

# Plot Portfolio Allocation
ax4 = axs[2]
# Use filtered_df.index for time on x-axis in the bar plot to ensure proper alignment
ax4.bar(filtered_df.index, cash_allocation, label="Cash", color="lightblue")
ax4.bar(filtered_df.index, share_value, bottom=cash_allocation, label="Shares", color="orange")
ax4.set_ylabel("Portfolio Allocation ($)")
ax4.set_xlabel("Time")
ax4.set_title(f"Portfolio Allocation on {specific_date}")
ax4.legend(loc="upper left")
ax4.set_xticklabels([])

# Show the plot
plt.show()




# Calculate daily return
df2['daily_return'] = df2['portfolio_value'].pct_change()

# Filter NaN values introduced by `pct_change`
daily_returns = df2['daily_return'].dropna()

# Calculate geometric return
initial_value = df2['portfolio_value'].iloc[0]  # First portfolio value
final_value = df2['portfolio_value'].iloc[-1]  # Last portfolio value
days = len(df2['date'].unique())  # Total number of trading days

geometric_return = ((final_value / initial_value) ** (252 / days) - 1)*100

# Calculate mean and standard deviation of daily returns
mean_daily_return = daily_returns.mean()*252*100
std_dev_daily_return = daily_returns.std()* np.sqrt(252)*100

# Calculate Sharpe ratio (assuming risk-free rate = 0)
sharpe_ratio = mean_daily_return / std_dev_daily_return

# Print metrics
print(f"Geometric Return (%): {geometric_return:.6f}")
print(f"Mean Daily Return (%): {mean_daily_return:.6f}")
print(f"Standard Deviation of Daily Return (%): {std_dev_daily_return:.6f}")
print(f"Sharpe Ratio: {sharpe_ratio:.6f}")


