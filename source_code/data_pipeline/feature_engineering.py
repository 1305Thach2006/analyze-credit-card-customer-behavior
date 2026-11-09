# Import thư viện numpy để tính toán các phép toán ma trận và số học
import numpy as np
# Import thư viện pandas để thao tác trên các cột của DataFrame
import pandas as pd

# Hàm thực hiện kỹ thuật tạo đặc trưng mới phục vụ phân tích hành vi
def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    # In thông báo bắt đầu quy trình tạo đặc trưng
    print("[FEATURE ENGINEERING] Bắt đầu tạo đặc trưng tỷ trọng chi tiêu và chỉ số rủi ro...")
    # Tạo bản sao DataFrame để không làm ảnh hưởng đến dữ liệu ban đầu
    data = df.copy()
    # Tính tỷ trọng chi tiêu mua sắm online trên tổng chi tiêu hàng tháng
    data["Pct_Online"] = data["Online_Shopping_Spending"] / data["Monthly_Spending"]
    # Tính tỷ trọng chi tiêu tạp hóa thực phẩm trên tổng chi tiêu hàng tháng
    data["Pct_Grocery"] = data["Grocery_Spending"] / data["Monthly_Spending"]
    # Tính tỷ trọng chi tiêu nhiên liệu xăng dầu trên tổng chi tiêu hàng tháng
    data["Pct_Fuel"] = data["Fuel_Spending"] / data["Monthly_Spending"]
    # Tính tỷ trọng chi tiêu ăn uống nhà hàng trên tổng chi tiêu hàng tháng
    data["Pct_Dining"] = data["Dining_Spending"] / data["Monthly_Spending"]
    # Tính tỷ trọng chi tiêu du lịch khách sạn trên tổng chi tiêu hàng tháng
    data["Pct_Travel"] = data["Travel_Spending"] / data["Monthly_Spending"]
    # Tính tỷ trọng chi tiêu giải trí trên tổng chi tiêu hàng tháng
    data["Pct_Entertainment"] = data["Entertainment_Spending"] / data["Monthly_Spending"]
    # Tính tỷ trọng chi tiêu hóa đơn sinh hoạt trên tổng chi tiêu hàng tháng
    data["Pct_Utility"] = data["Utility_Bill_Spending"] / data["Monthly_Spending"]
    # Tính tổng số tiền chi tiêu cho 4 danh mục hưởng thụ tùy ý
    data["Discretionary_Spending"] = data["Online_Shopping_Spending"] + data["Dining_Spending"] + data["Travel_Spending"] + data["Entertainment_Spending"]
    # Tính tỷ trọng chi tiêu tùy ý trên tổng chi tiêu hàng tháng
    data["Discretionary_Ratio"] = data["Discretionary_Spending"] / data["Monthly_Spending"]
    # Tính tổng số tiền chi tiêu cho 3 danh mục thiết yếu cuộc sống
    data["Essential_Spending"] = data["Grocery_Spending"] + data["Utility_Bill_Spending"] + data["Fuel_Spending"]
    # Tính tỷ trọng chi tiêu thiết yếu trên tổng chi tiêu hàng tháng
    data["Essential_Ratio"] = data["Essential_Spending"] / data["Monthly_Spending"]
    # Tính tỷ lệ đổi điểm thưởng trên tổng số điểm đã tích lũy
    data["Redemption_Rate"] = np.where(data["Reward_Points_Earned"] > 0, data["Reward_Points_Redeemed"] / data["Reward_Points_Earned"], 0.0)
    # Giới hạn tỷ lệ đổi thưởng trong khoảng tối đa bằng 1.0
    data["Redemption_Rate"] = data["Redemption_Rate"].clip(upper=1.0)
    # Danh sách các cột chi tiêu dùng để xác định danh mục chủ đạo
    spending_cols = ["Online_Shopping_Spending", "Grocery_Spending", "Fuel_Spending", "Dining_Spending", "Travel_Spending", "Entertainment_Spending", "Utility_Bill_Spending"]
    # Xác định danh mục chi tiêu có giá trị lớn nhất cho từng khách hàng
    data["Dominant_Expense_Category"] = data[spending_cols].idxmax(axis=1)
    # Gắn cờ cảnh báo rủi ro tín dụng dựa trên các ngưỡng an toàn tài chính
    data["Risk_Flag"] = np.where((data["Credit_Utilization"] > 0.75) | (data["Credit_Score"] < 550) | (data["Payment_Ratio"] < 0.40), 1, 0)
    # In thông báo đã hoàn thành tạo đặc trưng
    print(f"[FEATURE ENGINEERING] Hoàn tất! Số cột dữ liệu sau xử lý: {data.shape[1]} cột.")
    # Trả về DataFrame chứa các cột đặc trưng mới tạo
    return data
