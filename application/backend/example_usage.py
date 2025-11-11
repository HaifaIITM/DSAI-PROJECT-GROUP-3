"""
Example usage of the Hybrid Model API
"""
import requests
import pandas as pd
from datetime import datetime


def get_latest_predictions():
    """
    Get predictions and display trading recommendations
    """
    # Call API
    response = requests.get("http://localhost:8000/predict")
    
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return
    
    data = response.json()
    
    # Convert to DataFrame for easy analysis
    df = pd.DataFrame(data['predictions'])
    
    print("="*80)
    print(f"SPY Predictions - Generated: {data['generated_at']}")
    print("="*80)
    print()
    
    # Show last 5 days
    print("Last 5 Days:")
    print(df[['date', 'actual_close', 'h20_prediction', 'h20_signal']].tail())
    print()
    
    # Latest signals
    latest = df.iloc[-1]
    print("Latest Trading Signals:")
    print("-" * 40)
    print(f"Date: {latest['date']}")
    print(f"Close: ${latest['actual_close']:.2f}")
    print()
    print(f"1-Day (h1):  {latest['h1_prediction']:+.6f} → {latest['h1_signal']}")
    print(f"5-Day (h5):  {latest['h5_prediction']:+.6f} → {latest['h5_signal']}")
    print(f"20-Day (h20): {latest['h20_prediction']:+.6f} → {latest['h20_signal']} ⭐")
    print()
    
    # Consensus
    signals = [latest['h1_signal'], latest['h5_signal'], latest['h20_signal']]
    buy_count = signals.count('BUY')
    
    if buy_count >= 2:
        print("✅ CONSENSUS: STRONG BUY")
    elif buy_count == 1:
        print("⚠️  CONSENSUS: MIXED")
    else:
        print("❌ CONSENSUS: STRONG SELL")
    print()
    
    # Recent news
    print("Recent News:")
    print("-" * 40)
    for i, news in enumerate(data['recent_news'][:5], 1):
        print(f"{i}. [{news['date']}] {news['title']}")
        print(f"   Source: {news['publisher']}")
        print()
    
    return df, data


def calculate_accuracy(df):
    """
    Calculate how accurate predictions were (if we have next day's data)
    """
    # This would require comparing predictions with actual next-day returns
    # For now, just show prediction statistics
    
    print("Prediction Statistics (Last 30 Days):")
    print("-" * 40)
    
    for horizon in ['h1', 'h5', 'h20']:
        pred_col = f'{horizon}_prediction'
        signal_col = f'{horizon}_signal'
        
        avg_pred = df[pred_col].mean()
        buy_pct = (df[signal_col] == 'BUY').sum() / len(df) * 100
        
        print(f"{horizon:3s}: Avg prediction = {avg_pred:+.6f}, "
              f"Buy signals = {buy_pct:.1f}%")


def backtest_signals(df):
    """
    Simple backtest: Check if buying on signal would have been profitable
    """
    print("\nSimple Signal Backtest (h20):")
    print("-" * 40)
    
    # Calculate actual returns (using close prices)
    df['actual_return'] = df['actual_close'].pct_change()
    
    # Check if h20 signal direction matches actual return
    df['h20_correct'] = ((df['h20_signal'] == 'BUY') & (df['actual_return'] > 0)) | \
                        ((df['h20_signal'] == 'SELL') & (df['actual_return'] < 0))
    
    accuracy = df['h20_correct'].sum() / len(df) * 100
    
    print(f"Directional Accuracy: {accuracy:.1f}%")
    
    # Calculate hypothetical returns
    df['strategy_return'] = df['actual_return'] * df['h20_signal'].map({'BUY': 1, 'SELL': -1})
    
    cumulative_return = (1 + df['strategy_return']).prod() - 1
    print(f"Cumulative Return (30 days): {cumulative_return*100:+.2f}%")


if __name__ == "__main__":
    try:
        # Get predictions
        df, data = get_latest_predictions()
        
        # Show statistics
        calculate_accuracy(df)
        
        # Simple backtest
        backtest_signals(df)
        
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to API")
        print("Make sure the API is running: python main.py")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

