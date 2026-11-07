# Import thu vien os de thao tac voi duong dan he thong
import os

# Duong dan tuyet doi den tep du lieu goc
DATA_PATH = r"D:\PTDL\dataset\synthetic_credit_card_customer_behavior_dataset.csv"

# Thu muc goc chua cac ket qua dau ra
OUTPUT_DIR = r"D:\PTDL\source_code\outputs"

# Thu muc luu tru cac bieu do truc quan hoa duoi dang hinh anh
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures") #tạo đường dẫn con

# Thu muc luu tru cac bao cao thong ke duoi dang tap tin csv
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports") #tạo đường dẫn con

# Gia tri hat giong ngau nhien de co dinh ket qua thuc thi
RANDOM_STATE = 42

# So luong cum lua chon cho thuat toan phan cum K-Means
N_CLUSTERS = 4

# Danh sach cac cot chi tieu chi tiet theo 7 danh muc trong du lieu goc
SPENDING_CATEGORIES = [
    "Online_Shopping_Spending",
    "Grocery_Spending",
    "Fuel_Spending",
    "Dining_Spending",
    "Travel_Spending",
    "Entertainment_Spending",
    "Utility_Bill_Spending"
]

# Danh sach cac dac trung ty trong chi tieu duoc tao moi
RATIO_CATEGORIES = [
    "Pct_Online",
    "Pct_Grocery",
    "Pct_Fuel",
    "Pct_Dining",
    "Pct_Travel",
    "Pct_Entertainment",
    "Pct_Utility"
]

# He so tinh diem thuong co ban theo tung hang the tin dung
BASE_REWARD_MULTIPLIERS = {
    "Signature": 2.0,
    "Platinum": 1.5,
    "Gold": 1.0,
    "Silver": 0.5,
    "Basic": 0.5
}

# He so thuong them theo danh muc chi tieu co gia tri lon nhat
EXPENSE_BONUS_FACTORS = {
    "Travel_Spending": 3.0,
    "Dining_Spending": 3.0,
    "Online_Shopping_Spending": 2.0,
    "Entertainment_Spending": 1.5,
    "Grocery_Spending": 0.5,
    "Utility_Bill_Spending": 0.5,
    "Fuel_Spending": 0.5
}

# Cac nguong chi so tai chinh de xac dinh rui ro tin dung
RISK_THRESHOLDS = {
    "max_credit_utilization": 0.75,
    "min_credit_score": 550,
    "min_payment_ratio": 0.40
}
