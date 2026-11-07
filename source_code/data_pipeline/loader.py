# Import thu vien os de kiem tra su ton tai cua tap tin tren o dia
import os
# Import thu vien pandas de doc va quan ly cau truc du lieu bang
import pandas as pd

# Ham nap tap du lieu giao dich the tin dung tu duong dan chi dinh
def load_credit_card_data(file_path: str) -> pd.DataFrame:
    # Kiem tra xem duong dan tap tin co ton tai tren he thong hay khong
    if not os.path.exists(file_path):
        # Thong bao loi va dung thuc thi neu khong tim thay tap tin
        raise FileNotFoundError(f"Khong tim thay tap tin tai: {file_path}")
    # In thong bao qua trinh bat dau doc du lieu
    print(f"[DATA LOADER] Dang nap du lieu tu: {file_path}")
    # Doc tap tin CSV vao bien DataFrame cua pandas
    df = pd.read_csv(file_path)
    # In thong tin so dong va so cot cua tap du lieu ra man hinh
    print(f"[DATA LOADER] Nap thanh cong! Kich thuoc du lieu: {df.shape[0]} dong, {df.shape[1]} cot.")
    # Dem tong so luong o du lieu bi khuyet thieu (missing values)
    total_nulls = df.isnull().sum().sum()
    # In tong so gia tri khuyet thieu duoc tim thay
    print(f"[DATA LOADER] Tong so gia tri Missing (Null): {total_nulls}")
    # Tra ve doi tuong DataFrame da nap
    return df
