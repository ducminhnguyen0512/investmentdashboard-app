import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from scipy.optimize import minimize
import matplotlib

import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Tiêu đề ứng dụng
st.title("Dashboard Chứng Khoán")

@st.cache_data
def load_data():
    basic_url = "https://docs.google.com/spreadsheets/d/1AyXH3e7GOlN4uyOKYSf3kK0I4BM7Igsn/gviz/tq?tqx=out:csv"
    basic_drive = pd.read_csv(basic_url, index_col=0, usecols=range(8))
    
    sp400_url = "https://docs.google.com/spreadsheets/d/1ALSl_Eh-m8kAruIxDYrHaZoD1z02Dh-g/gviz/tq?tqx=out:csv"
    df_sp400 = pd.read_csv(sp400_url, index_col=0)
    df_sp400.index = pd.to_datetime(df_sp400.index).strftime('%Y-%m-%d')
    
    sp500_url = "https://docs.google.com/spreadsheets/d/1_03xDO9J3kwwvNjiTpiu5njM721o7ZHa/gviz/tq?tqx=out:csv"
    df_sp500 = pd.read_csv(sp500_url, index_col=0)
    df_sp500.index = pd.to_datetime(df_sp500.index).strftime('%Y-%m-%d')
    
    sp600_url = "https://docs.google.com/spreadsheets/d/1myNirpiJJXjyzjDKUM_AW8x4GZH6UJXC/gviz/tq?tqx=out:csv"
    df_sp600 = pd.read_csv(sp600_url, index_col=0)
    df_sp600.index = pd.to_datetime(df_sp600.index).strftime('%Y-%m-%d')
    
    combined_df = pd.concat([df_sp500, df_sp400, df_sp600], axis=1)


    # Lấy dữ liệu lịch sử cho S&P 500
    sp500_index = yf.download("^GSPC", start="2019-01-01", end="2025-03-29")  # Sửa lại ngày kết thúc
    sp500_close = sp500_index['Close']

    sp400_index = yf.download("^SP400", start="2019-01-01", end="2025-03-29")  # Sửa lại ngày kết thúc
    sp400_close = sp400_index['Close']

    sp600_index = yf.download("^SP600", start="2019-01-01", end="2025-03-29")  # Sửa lại ngày kết thúc
    sp600_close = sp600_index['Close']

    sp1500_index = yf.download("^SP1500", start="2019-01-01", end="2025-03-29")  # Sửa lại ngày kết thúc
    sp1500_close = sp1500_index['Close']

    NYA_index = yf.download("^NYA", start="2019-01-01", end="2025-03-29")  # Sửa lại ngày kết thúc
    NYA_close = NYA_index['Close']

    IXIC_index = yf.download("^IXIC", start="2019-01-01", end="2025-03-29")  # Sửa lại ngày kết thúc
    IXIC_close = IXIC_index['Close']

    combined_market_df = pd.concat([sp500_close, sp400_close, sp600_close,sp1500_close,NYA_close, IXIC_close  ], axis=1)
    # Gộp các DataFrame lại thành một DataFrame duy nhất và đặt tên cho các cột
    combined_market_df = pd.concat(
        [sp500_close, sp400_close, sp600_close, sp1500_close, NYA_close, IXIC_close],
    axis=1
    )

    # Đặt tên cho các cột
    combined_market_df.columns = [
        "S&P 500", 
        "S&P 400", 
        "S&P 600", 
        "S&P 1500 Composite", 
        "NYSE Composite", 
        "NASDAQ Composite"
    ] 
    #TiNH TOAAN YEARLY RETURN VA STD
    log_returns = np.log(combined_df / combined_df.shift(1))
    # Tinh do lech chuanchuan (standard deviation) cua moi co phieu trong log_return
    std_devs = log_returns.std()
    # Tinh yearly return va yearly standard deviation 
    days_in_year = 252
    yearly_return = log_returns.mean() * days_in_year
    yearly_std = log_returns.std() * np.sqrt(days_in_year)
    return basic_drive, combined_df, sp500_close, sp400_close, sp600_close, sp1500_close, NYA_close, IXIC_close, combined_market_df, yearly_return, yearly_std 
basic_drive, combined_df,  sp500_close, sp400_close, sp600_close, sp1500_close, NYA_close, IXIC_close, combined_market_df, yearly_return, yearly_std    = load_data()

# Tạo sidebar để chọn trang
page = st.sidebar.selectbox("Chọn trang:", ["Trang 1: Giá Đóng Cửa", "Trang 2: Tối Ưu Hóa Danh Mục"])

# Trang 1: Hiển thị giá đóng cửa của các chỉ số
if page == "Trang 1: Giá Đóng Cửa":
    st.header("Giá Đóng Cửa của Các Chỉ Số Chứng Khoán")

    # Lấy dữ liệu lịch sử cho các chỉ số
    tickers = {
        "S&P 500": "^GSPC",
        "S&P 400": "^SP400",
        "S&P 600": "^SP600",
        "S&P 1500 Composite": "^SP1500",
        "NYSE Composite": "^NYA",
        "NASDAQ Composite": "^IXIC"
    }

    # Tạo biểu đồ cho từng chỉ số
    for name, ticker in tickers.items():
        # Lấy dữ liệu lịch sử
        index_data = yf.download(ticker, start="2019-01-01", end="2025-03-29")
        
        # Chỉ lấy cột "Close"
        close_data = index_data['Close']
        
        # Vẽ biểu đồ
        plt.figure(figsize=(12, 6))
        plt.plot(close_data.index, close_data, label=name)
        plt.title(f'Giá Đóng Cửa: {name}', fontsize=16)
        plt.xlabel('Ngày', fontsize=14)
        plt.ylabel('Giá Đóng Cửa', fontsize=14)
        plt.legend()
        plt.grid()
        
        # Hiển thị biểu đồ trong Streamlit
        st.pyplot(plt)

# Trang 2: Tối Ưu Hóa Danh Mục (chưa thực hiện)
if page == "Trang 2: Tối Ưu Hóa Danh Mục":
    st.header("Tối Ưu Hóa Danh Mục")
    st.write("Chức năng này sẽ được phát triển trong phần tiếp theo.")
     
    # Ham loc theo khau vi rui roro
    def filter_stocks_by_risk_profile(risk_profile, df):
        if risk_profile == "Risk-seeking":
            # Dieu kien bat buocbuoc
            roe_condition = (df['ROE'] > 0.2)
            # Cac dieu kien khac (chi can thoa it nhatat 2)
            sector_condition = df['Sector'].isin(['Information Technology', 'Consumer Cyclical', 'Energy',
                                              'Communication Services', 'Real Estate'])
            beta_condition = (df['Beta'] > 1.5)
            pe_condition = (df['P/E'] > 25)
            # Dem so dieu kien thoa man (ngoai ROE)
            condition_count = (sector_condition.astype(int) +
                          beta_condition.astype(int) +
                          pe_condition.astype(int))
            # Loc: ROE bat buoc + it nhat 2 dieu kien khac
            conditions = roe_condition & (condition_count >= 2)

        elif risk_profile == "Risk-neutral":
         # Cac dieu kien khac (chi can thaa it nhat 3)
            roe_condition = (df['ROE'] > 0.15)
            beta_condition = (df['Beta'].between(0.8, 1.2))
            dividend_condition = (df['Dividend Yield'] > 2)
            pe_condition = (df['P/E'] > 15)
            # Dem so dieu kien thoa man
            condition_count = (roe_condition.astype(int) +
                          beta_condition.astype(int) +
                          dividend_condition.astype(int) +
                          pe_condition.astype(int))
        # Loc: it nhat 3 dieu kien
            conditions = (condition_count >= 3)

        elif risk_profile == "Risk-averse":
        # Dieu kien bat buocbuoc
            market_cap_condition = (df['Market Capitalization'] > 10000000000)
            dividend_condition = (df['Dividend Yield'] > 3)
            sector_condition = df['Sector'].isin(['Healthcare', 'Utilities', 'Consumer Defensive'])
        # Cac dieu kien khackhac
        
            pe_condition = (df['P/E'].between(10, 20))
            roe_condition = (df['ROE'] > 0.15)
            beta_condition = (df['Beta'] < 0.8)
        
            condition_count = (
                          pe_condition.astype(int) +
                          roe_condition.astype(int) +
                          beta_condition.astype(int))
        # 
            conditions = market_cap_condition & dividend_condition & sector_condition &  (condition_count >= 2)

        else:
            raise ValueError("Khẩu vị rủi ro không hợp lệ! Chọn: Risk-seeking, Risk-neutral, Risk-averse")

    # Loc du lieu va tao DataFrame stock_port
        stock_port = df[conditions]
        return stock_port
    st.sidebar.header("Chọn Khẩu Vị Rủi Ro")
    risk_choice = st.sidebar.selectbox("Chọn khẩu vị rủi ro:", ["Risk-seeking", "Risk-neutral", "Risk-averse"])


    stock_port = filter_stocks_by_risk_profile(risk_choice, basic_drive)
    # Tạo cột "Yearly Return" chỉ với các chỉ số trùng khớp
    stock_port['Yearly Return'] = yearly_return.loc[stock_port.index]

    # Tạo cột "Yearly Std" chỉ với các chỉ số trùng khớp
    stock_port['Yearly Std'] = yearly_std.loc[stock_port.index]
    st.header(f"Cổ phiếu phù hợp với {risk_choice}")
    st.dataframe(stock_port)
    
    import yfinance as yf
    from datetime import datetime

    # Lấy ngày hiện tại
    end_date = datetime.today().strftime('%Y-%m-%d')

    # Tải dữ liệu lãi suất trái phiếu Kho bạc Hoa Kỳ kỳ hạn 10 năm (^TNX) từ 2025-01-01 đến ngày hôm nay
    risk_free_rate_data = yf.download('^TNX', start='2025-01-01')

    # Lấy giá trị lãi suất của ngày cuối cùng
    risk_free_rate = risk_free_rate_data['Close'].values[-1]/100

    # Chuyển đổi hehe thành kiểu float
    risk_free_rate  = float(risk_free_rate )

    import pandas as pd
    import numpy as np

    # Tạo log_returns_query
    # Lấy danh sách mã cổ phiếu từ index của stock_port
    tickers = stock_port.index.tolist()

    # Lấy các cột tương ứng từ combine_df
    selected_data = combined_df[tickers]

    # Tính log returns, giữ nguyên index của combine_df
    log_returns_query = np.log(selected_data / selected_data.shift(1)).dropna()



    import pandas as pd


    # Tính ma trận tương quan từ DataFrame log_returns
    correlation_matrix = log_returns_query.corr()

    # Tính độ lệch chuẩn (standard deviation) của mỗi cổ phiếu trong log_return
    std_devs = log_returns_query.std()

    # Chuyển ma trận tương quan (correlation_matrix) thành ma trận numpy
    correlation_matrix_np = correlation_matrix.to_numpy()

    # Tạo ma trận độ lệch chuẩn với độ lệch chuẩn ở đường chéo
    std_devs_matrix = np.outer(std_devs, std_devs)

    # Tính ma trận hiệp phương sai
    covariance_matrix = correlation_matrix_np * std_devs_matrix

    # Chuyển ma trận hiệp phương sai thành DataFrame với các tên cột và chỉ số tương ứng
    covariance_df = pd.DataFrame(covariance_matrix, columns=log_returns_query.columns, index=log_returns_query.columns)

    #Hàm tính độ lệch chuẩn danh
    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    # Hàm tính lợi nhuận kỳ vọng của danh mục đầu tư
    def expected_return(weights, log_returns_query):
        return np.sum(log_returns_query.mean() * weights) * 252  # Quy đổi về lợi nhuận hàng năm

    # Hàm tính chỉ số Sharpe của danh mục đầu tư
    def sharpe_ratio(weights, log_returns_query, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns_query) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    from scipy.optimize import minimize

    stock_list = log_returns_query.columns.tolist()
    def neg_sharpe_ratio(weights, log_returns_query, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, log_returns_query, cov_matrix, risk_free_rate)

    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(0 , 0.5) for _ in range(len(stock_list))]
    initial_weights = np.array([1/len(stock_list)]*len(stock_list))

    optimized_results = minimize(neg_sharpe_ratio, initial_weights, args=(log_returns_query, covariance_df, risk_free_rate), method='SLSQP', constraints=constraints, bounds=bounds)

    optimal_weights = optimized_results.x

    optimal_portfolio_return = expected_return(optimal_weights, log_returns_query)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, covariance_df)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns_query, covariance_df, risk_free_rate)

    # Tạo DataFrame từ stock_list và optimal_weights
    portfolio_df = pd.DataFrame({
    'Ticker': stock_list,
    'Weight': optimal_weights
    })

    # Lọc ra các mã có trọng số khác 0
    portfolio_df = portfolio_df[portfolio_df['Weight'] > 0.0001]

    # Định dạng hiển thị số thập phân với 4 chữ số sau dấu phẩy
    pd.options.display.float_format = '{:.5f}'.format
# Hoặc sử dụng st.metric để hiển thị theo cách khác
    st.metric(label="Expected Annual Return", value=f"{optimal_portfolio_return:.4f}")
    st.metric(label="Expected Volatility", value=f"{optimal_portfolio_volatility:.4f}")
    st.metric(label="Sharpe Ratio", value=f"{optimal_sharpe_ratio:.4f}")
    # In ra DataFrame đã lọc
    portfolio_df
    # Tiêu đề ứng dụng
    st.title("Thông tin các cổ phiếu được lựa chọn cho danh mục")
    
    

# Lấy danh sách các mã cổ phiếu từ cột "Ticker" của portfolio_df
    tickers_to_select = portfolio_df['Ticker'].tolist()

# Tạo DataFrame stock_port_opt bằng cách lọc stock_port
    stock_port_opt = stock_port[stock_port.index.isin(tickers_to_select)].copy()

# Hiển thị DataFrame stock_port_opt trong Streamlit
    st.subheader("DataFrame stock_port_opt")
    st.dataframe(stock_port_opt)  # Hiển thị DataFrame trong ứng dụng Streamlit
    
    import streamlit as st
    import matplotlib.pyplot as plt

    # Lấy dữ liệu từ DataFrame
    labels = portfolio_df['Ticker']
    sizes = portfolio_df['Weight']

    # Tạo biểu đồ hình tròn
    fig, ax = plt.subplots(figsize=(10, 8))  # Tạo Figure và Axes
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=plt.cm.tab10.colors)

    # Tùy chỉnh các thuộc tính của văn bản
    for text in texts:
        text.set_fontsize(12)
    for autotext in autotexts:
        autotext.set_color('white')
    autotext.set_fontsize(12)

    # Đảm bảo biểu đồ hình tròn và thêm tiêu đề
    ax.axis('equal')  # Đảm bảo hình tròn
    ax.set_title('Optimal Portfolio Weights', fontsize=16, fontweight='bold')

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(fig)




    portfolio_cumulative = (log_returns_query+1).cumprod().dot(optimal_weights)
    

    # Tính toán lợi suất logarit
    benchmark_returns = np.log(combined_market_df / combined_market_df.shift(1))
    benchmark_return = benchmark_returns.mean() *252
    benchmark_std = benchmark_returns.std() * np.sqrt(252)

    st.write(benchmark_return)
    st.write(benchmark_std)

    benchmark_cumulative = (benchmark_returns + 1).cumprod()
    


    # Đảm bảo rằng chỉ số của portfolio_cumulative và benchmark_cumulative là kiểu datetime
    portfolio_cumulative.index = pd.to_datetime(portfolio_cumulative.index)
    benchmark_cumulative.index = pd.to_datetime(benchmark_cumulative.index)

    # Tạo các DataFrame riêng biệt từ các cột trong benchmark_cumulative
    sp500_bench = benchmark_cumulative[['S&P 500']].copy()
    sp400_bench = benchmark_cumulative[['S&P 400']].copy()
    sp600_bench = benchmark_cumulative[['S&P 600']].copy()
    sp1500_bench = benchmark_cumulative[['S&P 1500 Composite']].copy()
    nyse_bench = benchmark_cumulative[['NYSE Composite']].copy()
    nasdaq_bench = benchmark_cumulative[['NASDAQ Composite']].copy()

    # Tạo từ điển để dễ dàng truy cập các DataFrame
    benchmarks = {
    "S&P 500": sp500_bench,
    "S&P 400": sp400_bench,
    "S&P 600": sp600_bench,
    "S&P 1500 Composite": sp1500_bench,
    "NYSE Composite": nyse_bench,
    "NASDAQ Composite": nasdaq_bench
    }

    # Tiêu đề ứng dụng
    st.title("So sánh Cumulative Returns: Portfolio vs Benchmark")

    # Tùy chọn cho người dùng chọn các chỉ số
    selected_benchmarks = st.multiselect("Chọn các chỉ số để so sánh:", list(benchmarks.keys()))

    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_cumulative.index, portfolio_cumulative, label='Portfolio Cumulative', color='blue')

    # Vẽ đường cho các chỉ số đã chọn
    for benchmark in selected_benchmarks:
        plt.plot(benchmarks[benchmark].index, benchmarks[benchmark], label=benchmark)

    # Thêm tiêu đề và nhãn
    plt.title('Cumulative Returns: Portfolio vs Selected Benchmarks', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Cumulative Return', fontsize=14)
    plt.legend()  # Hiển thị chú thích
    plt.grid()  # Thêm lưới cho biểu đồ

    # Hiển thị biểu đồ trong Streamlit
    st.pyplot(plt)