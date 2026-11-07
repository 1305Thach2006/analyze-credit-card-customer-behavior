# Import thu vien os de thao tac voi duong dan he thong
import os

# Duong dan tuyet doi den tep du lieu goc
DATA_PATH = r"D:\PTDL\dataset\synthetic_credit_card_customer_behavior_dataset.csv"

# Thu muc goc chua cac ket qua dau ra
OUTPUT_DIR = r"D:\PTDL\source_code\outputs"

# Thu muc luu tru cac bieu do truc quan hoa duoi dang hinh anh
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Thu muc luu tru cac bao cao thong ke duoi dang tap tin csv
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Gia tri hat giong ngau nhien de co dinh ket qua thuc thi
RANDOM_STATE = 42

# So luong cum lua chon cho thuat toan phan cum K-Means
N_CLUSTERS = 4

# Danh sach cac cot chi tieu chi tiet theo 7 danh muc trong du lieu goc
SPENDING_CATEGORIES = [
    # Cot chi tieu mua sam truc tuyen
    "Online_Shopping_Spending",
    # Cot chi tieu tap hoa va thuc pham
    "Grocery_Spending",
    # Cot chi tieu nhien lieu xang dau
    "Fuel_Spending",
    # Cot chi tieu an uong nha hang
    "Dining_Spending",
    # Cot chi tieu du lich khach san
    "Travel_Spending",
    # Cot chi tieu giai tri xem phim
    "Entertainment_Spending",
    # Cot chi tieu hoa don dien nuoc sinh hoat
    "Utility_Bill_Spending"
# Ket thuc khoi danh sach hoac tu dien
]

# Danh sach cac dac trung ty trong chi tieu duoc tao moi
RATIO_CATEGORIES = [
    # Ty trong chi tieu mua sam truc tuyen
    "Pct_Online",
    # Ty trong chi tieu tap hoa
    "Pct_Grocery",
    # Ty trong chi tieu nhien lieu
    "Pct_Fuel",
    # Ty trong chi tieu an uong nha hang
    "Pct_Dining",
    # Ty trong chi tieu du lich
    "Pct_Travel",
    # Ty trong chi tieu giai tri
    "Pct_Entertainment",
    # Ty trong chi tieu hoa don sinh hoat
    "Pct_Utility"
# Ket thuc khoi danh sach hoac tu dien
]

# He so tinh diem thuong co ban theo tung hang the tin dung
BASE_REWARD_MULTIPLIERS = {
    # Hang the Signature cao cap nhat nhan he so gap doi
    "Signature": 2.0,
    # Hang the Platinum nhan he so cao
    "Platinum": 1.5,
    # Hang the Gold nhan he so tieu chuan
    "Gold": 1.0,
    # Hang the Silver nhan he so co ban
    "Silver": 0.5,
    # Hang the Basic pho thong nhan he so co ban
    "Basic": 0.5
# Ket thuc khoi danh sach hoac tu dien
}

# He so thuong them theo danh muc chi tieu co gia tri lon nhat
EXPENSE_BONUS_FACTORS = {
    # Uu dai dac biet cho chi tieu du lich
    "Travel_Spending": 3.0,
    # Uu dai dac biet cho chi tieu nha hang an uong
    "Dining_Spending": 3.0,
    # Uu dai cho chi tieu mua sam truc tuyen
    "Online_Shopping_Spending": 2.0,
    # Uu dai cho chi tieu hoat dong giai tri
    "Entertainment_Spending": 1.5,
    # Uu dai co ban cho chi tieu tap hoa
    "Grocery_Spending": 0.5,
    # Uu dai co ban cho hoa don dien nuoc
    "Utility_Bill_Spending": 0.5,
    # Uu dai co ban cho chi tieu do xang
    "Fuel_Spending": 0.5
# Ket thuc khoi danh sach hoac tu dien
}

# Cac nguong chi so tai chinh de xac dinh rui ro tin dung
RISK_THRESHOLDS = {
    # Nguong tran ty le su dung han muc tin dung cho phep
    "max_credit_utilization": 0.75,
    # Nguong san diem tin dung toi thieu an toan
    "min_credit_score": 550,
    # Nguong san ty le thanh toan no toi thieu an toan
    "min_payment_ratio": 0.40
# Ket thuc khoi danh sach hoac tu dien
}
