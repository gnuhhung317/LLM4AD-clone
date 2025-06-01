import requests
from pycoingecko import CoinGeckoAPI
import pandas as pd
from datetime import datetime

# Khởi tạo client CoinGecko
cg = CoinGeckoAPI()

def fetch_coin_data(coin_id):
    try:
        # Lấy dữ liệu chi tiết của đồng coin
        coin_data = cg.get_coin_by_id(coin_id, localization='false', 
                                     tickers=False, market_data=True, 
                                     community_data=True, developer_data=True)
        
        # Lấy dữ liệu lịch sử giao dịch (dùng để tính NVT)
        market_data = cg.get_coin_market_chart_by_id(coin_id, vs_currency='usd', days='1')
        
        # Trích xuất các thông tin cần thiết
        market_cap = coin_data.get('market_data', {}).get('market_cap', {}).get('usd', 0)
        current_price = coin_data.get('market_data', {}).get('current_price', {}).get('usd', 0)
        circulating_supply = coin_data.get('market_data', {}).get('circulating_supply', 0)
        total_volume_24h = coin_data.get('market_data', {}).get('total_volume', {}).get('usd', 0)
        
        # Tính NVT Ratio (Market Cap / Volume 24h)
        nvt_ratio = market_cap / total_volume_24h if total_volume_24h > 0 else 'N/A'
        
        # Lấy thông tin bổ sung
        ath = coin_data.get('market_data', {}).get('ath', {}).get('usd', 0)  # Giá cao nhất mọi thời đại
        ath_date = coin_data.get('market_data', {}).get('ath_date', {}).get('usd', 'N/A')
        community_score = coin_data.get('community_data', {}).get('twitter_followers', 'N/A')
        developer_score = coin_data.get('developer_data', {}).get('code_additions_deletions_4_weeks', 'N/A')
        
        # Tạo dictionary chứa dữ liệu
        data = {
            'Coin': coin_data.get('name', coin_id),
            'Symbol': coin_data.get('symbol', '').upper(),
            'Current Price (USD)': current_price,
            'Market Cap (USD)': market_cap,
            'Circulating Supply': circulating_supply,
            '24h Trading Volume (USD)': total_volume_24h,
            'NVT Ratio': nvt_ratio,
            'All-Time High (USD)': ath,
            'ATH Date': ath_date,
            'Twitter Followers': community_score,
            'Code Activity (4 weeks)': developer_score
        }
        
        return data
    
    except Exception as e:
        print(f"Error fetching data for {coin_id}: {e}")
        return None

def main():
    # Nhận input từ người dùng
    coin_id = input("Nhập ID hoặc tên đồng coin (ví dụ: bitcoin, ethereum, cardano): ").lower().strip()
    
    # Lấy dữ liệu
    data = fetch_coin_data(coin_id)
    
    if data:
        # Tạo DataFrame để hiển thị dữ liệu
        df = pd.DataFrame([data])
        print("\nDữ liệu của đồng coin:")
        print(df.to_string(index=False))
        
        # Lưu dữ liệu vào file CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{coin_id}_data_{timestamp}.csv"
        df.to_csv(filename, index=False)
        print(f"\nDữ liệu đã được lưu vào file: {filename}")
    else:
        print("Không thể lấy dữ liệu. Vui lòng kiểm tra ID coin hoặc kết nối mạng.")

if __name__ == "__main__":
    main()