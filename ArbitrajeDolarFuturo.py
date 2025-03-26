import pyRofex
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations


tickers = ["DLR/MAY25", "DLR/AGO25", "DLR/JUN25", "DLR/JUL25", "DLR/ABR25", "DLR/OCT25", "DLR/DIC25", "DLR/SEP25"]
end_date = datetime(2025, 3, 24)
start_date = end_date - timedelta(days=90)  # Trading period begins 90 days prior
warm_up_days = 60  # Extra days for historical data warm-up
data_start_date = start_date - timedelta(days=warm_up_days)
data_start_str = data_start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")


def initialize_pyrofex():
    return pyRofex.initialize(
        user="xxx",
        password="xxx",
        account="xxx",
        environment=pyRofex.Environment.REMARKET
    )

#Fetch historical trade data for a given ticker
def fetch_historical_data(ticker, start_str, end_str):
    initialize_pyrofex()
    try:
        historical_data = pyRofex.get_trade_history(ticker, start_date=start_str, end_date=end_str)
        if historical_data.get("status") == "OK" and "trades" in historical_data:
            return historical_data["trades"]
        else:
            print(f"Error fetching data for {ticker}: {historical_data.get('description', 'Unknown error')}")
            return []
    except Exception as e:
        print(f"Exception occurred for {ticker}: {str(e)}")
        return []
    finally:
        try:
            pyRofex.close()
        except AttributeError:
            pass

#Collect trade data for all tickers
all_trades = []
for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    trades = fetch_historical_data(ticker, data_start_str, end_str)
    for trade in trades:
        date = trade.get("datetime")
        price = trade.get("price")
        if date and price:
            all_trades.append({"instrument": ticker, "datetime": date, "price": price})

if not all_trades:
    print(f"No trade data available for {tickers} in the range {data_start_str} to {end_str}.")
    exit()


df = pd.DataFrame(all_trades)
df["time"] = pd.to_datetime(df["datetime"])
df = df[["instrument", "time", "price"]].sort_values(by=["time", "instrument"])
print(f"\nHistorical Trade Data from {data_start_str} to {end_str}:")
print(df)

# Process data into 10-minute intervals and add bid/ask prices
df_last = df.groupby(['time', 'instrument']).last().reset_index()
df_pivot = df_last.set_index('time').pivot(columns='instrument', values='price')
last_prices_10min = df_pivot.resample('10T').last().ffill()

spread = 0.01  # Fixed spread assumption for bid/ask
for inst in tickers:
    last_prices_10min[f'{inst}_bid'] = last_prices_10min[inst] - spread / 2
    last_prices_10min[f'{inst}_ask'] = last_prices_10min[inst] + spread / 2

#Generate unique ticker pairs for pairs trading
pairs = list(combinations(tickers, 2))
print(f"\nNumber of unique pairs: {len(pairs)}")

#Calculate spreads and z-scores for each pair
data = {}
window = 4320  # Roughly 30 days of 10-minute intervals
for pair in pairs:
    inst1, inst2 = pair
    spread_name = f"{inst1}_{inst2}_spread"
    mean_name = f"{spread_name}_mean"
    std_name = f"{spread_name}_std"
    zscore_name = f"{inst1}_{inst2}_zscore"
    
    spread = last_prices_10min[inst1] - last_prices_10min[inst2]
    mean = spread.rolling(window=window, min_periods=1).mean().shift(1)
    std = spread.rolling(window=window, min_periods=2).std().shift(1)
    zscore = np.where((std != 0) & (~std.isna()), (spread - mean) / std, np.nan)
    
    data[spread_name] = spread
    data[mean_name] = mean
    data[std_name] = std
    data[zscore_name] = pd.Series(zscore, index=spread.index)

spreads_df = pd.DataFrame(data, index=last_prices_10min.index).reset_index()

#Trading strategy implementation
initial_capital = 50_000_000.0  #Starting capital in pesos
cumulative_realized_profit = 0.0
positions = {pair: None for pair in pairs}
closed_trades = {pair: [] for pair in pairs}
spreads_df['portfolio_value'] = 0.0
stop_loss_threshold = 0.05  #5% stop-loss per trade
max_scaling_factor = 3.0  

# Main loop: manage positions and execute trades
for idx, row in spreads_df.iterrows():
    time = row['time']
    total_unrealized_profit = 0.0
    
    for pair in pairs:
        if positions[pair] is not None:
            zscore_name = f"{pair[0]}_{pair[1]}_zscore"
            current_zscore = row[zscore_name]
            pos = positions[pair]
            inst1, inst2 = pair
            if pd.isna(current_zscore):
                continue
            if pos['type'] == 'long':
                current_bid_A = last_prices_10min.loc[time, f'{inst1}_bid']
                current_ask_B = last_prices_10min.loc[time, f'{inst2}_ask']
                unrealized_profit = pos['N'] * (current_bid_A - pos['entry_price_A'] + pos['entry_price_B'] - current_ask_B)
                close_condition = current_zscore >= -0.5
            elif pos['type'] == 'short':
                current_ask_A = last_prices_10min.loc[time, f'{inst1}_ask']
                current_bid_B = last_prices_10min.loc[time, f'{inst2}_bid']
                unrealized_profit = pos['N'] * (pos['entry_price_A'] - current_ask_A + current_bid_B - pos['entry_price_B'])
                close_condition = current_zscore <= 0.5
            
            if not pd.isna(unrealized_profit):
                total_unrealized_profit += unrealized_profit
                pos['unrealized_profits'].append(unrealized_profit)
                if unrealized_profit < -stop_loss_threshold * pos['entry_portfolio_value']:
                    if pos['type'] == 'long':
                        exit_price_A = last_prices_10min.loc[time, f'{inst1}_bid']
                        exit_price_B = last_prices_10min.loc[time, f'{inst2}_ask']
                        profit = pos['N'] * (exit_price_A - pos['entry_price_A'] + pos['entry_price_B'] - exit_price_B)
                    else:
                        exit_price_A = last_prices_10min.loc[time, f'{inst1}_ask']
                        exit_price_B = last_prices_10min.loc[time, f'{inst2}_bid']
                        profit = pos['N'] * (pos['entry_price_A'] - exit_price_A + exit_price_B - pos['entry_price_B'])
                    if not pd.isna(profit):
                        cumulative_realized_profit += profit
                    trade = {
                        'pair': pair, 'type': pos['type'], 'entry_time': pos['entry_time'], 'exit_time': time,
                        'entry_price_A': pos['entry_price_A'], 'entry_price_B': pos['entry_price_B'],
                        'exit_price_A': exit_price_A, 'exit_price_B': exit_price_B, 'N': pos['N'], 'profit': profit,
                        'max_adverse_excursion': min(pos['unrealized_profits']) if pos['unrealized_profits'] else 0,
                        'percentage_drawdown': ((-min(pos['unrealized_profits']) / pos['entry_portfolio_value']) * 100 
                                               if pos['unrealized_profits'] and min(pos['unrealized_profits']) < 0 else 0),
                        'close_reason': 'stop_loss'
                    }
                    closed_trades[pair].append(trade)
                    positions[pair] = None
                    continue
            else:
                pos['unrealized_profits'].append(0)
            
            if close_condition:
                if pos['type'] == 'long':
                    exit_price_A = last_prices_10min.loc[time, f'{inst1}_bid']
                    exit_price_B = last_prices_10min.loc[time, f'{inst2}_ask']
                    profit = pos['N'] * (exit_price_A - pos['entry_price_A'] + pos['entry_price_B'] - exit_price_B)
                else:
                    exit_price_A = last_prices_10min.loc[time, f'{inst1}_ask']
                    exit_price_B = last_prices_10min.loc[time, f'{inst2}_bid']
                    profit = pos['N'] * (pos['entry_price_A'] - exit_price_A + exit_price_B - pos['entry_price_B'])
                if not pd.isna(profit):
                    cumulative_realized_profit += profit
                trade = {
                    'pair': pair, 'type': pos['type'], 'entry_time': pos['entry_time'], 'exit_time': time,
                    'entry_price_A': pos['entry_price_A'], 'entry_price_B': pos['entry_price_B'],
                    'exit_price_A': exit_price_A, 'exit_price_B': exit_price_B, 'N': pos['N'], 'profit': profit,
                    'max_adverse_excursion': min(pos['unrealized_profits']) if pos['unrealized_profits'] else 0,
                    'percentage_drawdown': ((-min(pos['unrealized_profits']) / pos['entry_portfolio_value']) * 100 
                                           if pos['unrealized_profits'] and min(pos['unrealized_profits']) < 0 else 0),
                    'close_reason': 'normal'
                }
                closed_trades[pair].append(trade)
                positions[pair] = None

    portfolio_value = initial_capital + cumulative_realized_profit + total_unrealized_profit
    if pd.isna(portfolio_value):
        portfolio_value = initial_capital + cumulative_realized_profit
    spreads_df.at[idx, 'portfolio_value'] = portfolio_value

    #Open new positions (after warmup)
    if time >= start_date:
        available_pairs = [pair for pair in pairs if positions[pair] is None]
        if available_pairs:
            deviations = [(pair, row[f"{pair[0]}_{pair[1]}_zscore"]) for pair in available_pairs if not pd.isna(row[f"{pair[0]}_{pair[1]}_zscore"])]
            if deviations:
                deviations.sort(key=lambda x: abs(x[1]), reverse=True)
                top_pair, top_zscore = deviations[0]
                inst1, inst2 = top_pair
                price1 = last_prices_10min.loc[time, inst1]
                price2 = last_prices_10min.loc[time, inst2]
                
                scaling_factor = min(abs(top_zscore) / 2, max_scaling_factor) if abs(top_zscore) > 2 else 1.0
                
                if top_zscore > 2:  # short the spread
                    if pd.isna(price1) or pd.isna(price2) or price1 + price2 == 0 or pd.isna(portfolio_value):
                        print(f"Skipping trade for {inst1}-{inst2} at {time} due to invalid data.")
                        continue
                    base_N = int((0.75 * portfolio_value) / (price1 + price2))
                    N = int(base_N * scaling_factor)
                    positions[top_pair] = {
                        'type': 'short', 'N': N, 'entry_price_A': last_prices_10min.loc[time, f'{inst1}_bid'],
                        'entry_price_B': last_prices_10min.loc[time, f'{inst2}_ask'], 'entry_time': time,
                        'unrealized_profits': [], 'entry_portfolio_value': portfolio_value
                    }
                elif top_zscore < -2:  # long the spread
                    if pd.isna(price1) or pd.isna(price2) or price1 + price2 == 0 or pd.isna(portfolio_value):
                        print(f"Skipping trade for {inst1}-{inst2} at {time} due to invalid data.")
                        continue
                    base_N = int((0.75 * portfolio_value) / (price1 + price2))
                    N = int(base_N * scaling_factor)
                    positions[top_pair] = {
                        'type': 'long', 'N': N, 'entry_price_A': last_prices_10min.loc[time, f'{inst1}_ask'],
                        'entry_price_B': last_prices_10min.loc[time, f'{inst2}_bid'], 'entry_time': time,
                        'unrealized_profits': [], 'entry_portfolio_value': portfolio_value
                    }

# Plots
colors = {
    'DLR/MAY25': 'blue', 'DLR/AGO25': 'red', 'DLR/JUN25': 'green', 'DLR/JUL25': 'orange',
    'DLR/ABR25': 'purple', 'DLR/OCT25': 'brown', 'DLR/DIC25': 'gray', 'DLR/SEP25': 'cyan'
}

fig1, ax1 = plt.subplots(figsize=(12, 6))
for ticker in tickers:
    if ticker in last_prices_10min.columns:
        ax1.plot(last_prices_10min.index, last_prices_10min[ticker], label=ticker, color=colors[ticker])
ax1.set_xlabel('Time')
ax1.set_ylabel('Price (Pesos)')
ax1.set_title('Prices of All Instruments Over Time')
ax1.legend()
plt.grid(True)

#portfolio value over time
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(spreads_df['time'], spreads_df['portfolio_value'], label='Portfolio Value', color='purple')
ax2.set_xlabel('Time')
ax2.set_ylabel('Portfolio Value (Pesos)')
ax2.set_title('Portfolio Value Over Time')
ax2.legend()
plt.grid(True)

# Performance metrics
all_closed_trades = [trade for pair_trades in closed_trades.values() for trade in pair_trades]
if all_closed_trades:
    total_profit = sum(trade['profit'] for trade in all_closed_trades if not pd.isna(trade['profit']))
    num_trades = len(all_closed_trades)
    winning_trades = sum(1 for trade in all_closed_trades if not pd.isna(trade['profit']) and trade['profit'] > 0)
    win_rate = winning_trades / num_trades if num_trades > 0 else 0
    average_profit = total_profit / num_trades if num_trades > 0 else 0
    max_per_trade_drawdown = (max(-trade['max_adverse_excursion'] for trade in all_closed_trades if trade['max_adverse_excursion'] < 0)
                              if any(trade['max_adverse_excursion'] < 0 for trade in all_closed_trades) else 0)
    running_max = spreads_df['portfolio_value'].cummax()
    drawdown = running_max - spreads_df['portfolio_value']
    max_drawdown_equity = drawdown.max() if not drawdown.empty else 0
    
    print("\nStrategy Performance Metrics")
    print(f"- Total Profit: {total_profit:.2f} pesos")
    print(f"- Number of Trades: {num_trades}")
    print(f"- Win Rate: {win_rate:.2%}")
    print(f"- Average Profit per Trade: {average_profit:.2f} pesos")
    print(f"- Maximum Per-Trade Adverse Excursion: {max_per_trade_drawdown:.2f} pesos")
    print(f"- Maximum Drawdown from Equity Curve: {max_drawdown_equity:.2f} pesos")
else:
    print("\nNo trades were executed.")

# export to excel
closed_trades_df = pd.DataFrame(all_closed_trades)
if not closed_trades_df.empty:
    closed_trades_df['entry_time'] = closed_trades_df['entry_time'].astype(str)
    closed_trades_df['exit_time'] = closed_trades_df['exit_time'].astype(str)
    closed_trades_df['pair'] = closed_trades_df['pair'].apply(lambda x: f"{x[0]}-{x[1]}")

# previous fix
zscore_columns = [f"{pair[0]}_{pair[1]}_zscore" for pair in pairs]
zscores_df = spreads_df[['time'] + zscore_columns].copy()
zscores_df['time'] = zscores_df['time'].astype(str)
portfolio_value_df = spreads_df[['time', 'portfolio_value']].copy()
portfolio_value_df['time'] = portfolio_value_df['time'].astype(str)

excel_file = 'trading_strategy_results.xlsx'
with pd.ExcelWriter(excel_file) as writer:
    if not closed_trades_df.empty:
        closed_trades_df.to_excel(writer, sheet_name='Closed Trades', index=False)
    zscores_df.to_excel(writer, sheet_name='Z-Scores', index=False)
    portfolio_value_df.to_excel(writer, sheet_name='Portfolio Value', index=False)

plt.show()
