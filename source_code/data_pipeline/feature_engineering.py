# Import thu vien numpy de tinh toan cac phep toan ma tran va so hoc
import numpy as np
# Import thu vien pandas de thao tac tren cac cot cua DataFrame
import pandas as pd

# Ham thuc hien ky thuat tao dac trung moi phuc vu phan tich hanh vi
def engineer_credit_features(df: pd.DataFrame) -> pd.DataFrame:
    # In thong bao bat dau quy trinh tao dac trung
    print("[FEATURE ENGINEERING] Bat dau tao dac trung ty trong chi tieu va chi so rui ro...")
    # Tao ban sao DataFrame de khong anh huong den du lieu ban dau
    data = df.copy()
    # Tinh ty trong chi tieu mua sam online tren tong chi tieu hang thang
    data["Pct_Online"] = data["Online_Shopping_Spending"] / data["Monthly_Spending"]
    # Tinh ty trong chi tieu tap hoa tren tong chi tieu hang thang
    data["Pct_Grocery"] = data["Grocery_Spending"] / data["Monthly_Spending"]
    # Tinh ty trong chi tieu nhien lieu tren tong chi tieu hang thang
    data["Pct_Fuel"] = data["Fuel_Spending"] / data["Monthly_Spending"]
    # Tinh ty trong chi tieu an uong nha hang tren tong chi tieu hang thang
    data["Pct_Dining"] = data["Dining_Spending"] / data["Monthly_Spending"]
    # Tinh ty trong chi tieu du lich tren tong chi tieu hang thang
    data["Pct_Travel"] = data["Travel_Spending"] / data["Monthly_Spending"]
    # Tinh ty trong chi tieu giai tri tren tong chi tieu hang thang
    data["Pct_Entertainment"] = data["Entertainment_Spending"] / data["Monthly_Spending"]
    # Tinh ty trong chi tieu hoa don sinh hoat tren tong chi tieu hang thang
    data["Pct_Utility"] = data["Utility_Bill_Spending"] / data["Monthly_Spending"]
    # Tinh tong so tien chi tieu cho cac danh muc tuy y
    data["Discretionary_Spending"] = (
        data["Online_Shopping_Spending"] +
        data["Dining_Spending"] +
        data["Travel_Spending"] +
        data["Entertainment_Spending"]
    )
    # Tinh ty trong chi tieu tuy y tren tong chi tieu hang thang
    data["Discretionary_Ratio"] = data["Discretionary_Spending"] / data["Monthly_Spending"]
    # Tinh tong so tien chi tieu cho cac danh muc thiet yeu
    data["Essential_Spending"] = (
        data["Grocery_Spending"] +
        data["Utility_Bill_Spending"] +
        data["Fuel_Spending"]
    )
    # Tinh ty trong chi tieu thiet yeu tren tong chi tieu hang thang
    data["Essential_Ratio"] = data["Essential_Spending"] / data["Monthly_Spending"]
    # Tinh ty le doi diem thuong tren tong so diem da tich luy
    data["Redemption_Rate"] = np.where(
        data["Reward_Points_Earned"] > 0,
        data["Reward_Points_Redeemed"] / data["Reward_Points_Earned"],
        0.0
    )
    # Gioi han ty le doi thuong trong khoang toi da bang 1.0
    data["Redemption_Rate"] = data["Redemption_Rate"].clip(upper=1.0)
    # Danh sach cac cot chi tieu dung de xac dinh danh muc chu dao
    spending_cols = [
        "Online_Shopping_Spending", "Grocery_Spending", "Fuel_Spending",
        "Dining_Spending", "Travel_Spending", "Entertainment_Spending", "Utility_Bill_Spending"
    ]
    # Xac dinh danh muc chi tieu co gia tri lon nhat cho tung dong
    data["Dominant_Expense_Category"] = data[spending_cols].idxmax(axis=1)
    # Gan co canh bao rui ro tin dung dua tren cac nguong an toan tai chinh
    data["Risk_Flag"] = np.where(
        (data["Credit_Utilization"] > 0.75) | (data["Credit_Score"] < 550) | (data["Payment_Ratio"] < 0.40),
        1,
        0
    )
    # In thong bao da hoan thanh tao dac trung
    print(f"[FEATURE ENGINEERING] Hoan tat! So cot du lieu sau xu ly: {data.shape[1]} cot.")
    # Tra ve DataFrame chua cac cot dac trung moi tao
    return data
