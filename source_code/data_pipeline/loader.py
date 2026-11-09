# Import thư viện os để kiểm tra sự tồn tại của tập tin trên ổ đĩa
import os
# Import thư viện pandas để đọc và quản lý cấu trúc dữ liệu bảng
import pandas as pd

# Hàm nạp tập dữ liệu hành vi sử dụng thẻ tín dụng từ đường dẫn chỉ định
def load_credit_card_data(file_path: str) -> pd.DataFrame:
    # Kiểm tra xem đường dẫn tập tin có tồn tại trên hệ thống hay không
    if not os.path.exists(file_path):
        # Thông báo lỗi và dừng thực thi nếu không tìm thấy tập tin
        raise FileNotFoundError(f"Không tìm thấy tập tin tại: {file_path}")
    # In thông báo quá trình bắt đầu đọc dữ liệu
    print(f"[DATA LOADER] Đang nạp dữ liệu từ: {file_path}")
    # Đọc tập tin CSV vào biến DataFrame của pandas
    df = pd.read_csv(file_path)
    # In thông tin số dòng và số cột của tập dữ liệu ra màn hình
    print(f"[DATA LOADER] Nạp thành công! Kích thước dữ liệu: {df.shape[0]} dòng, {df.shape[1]} cột.")
    # Đếm tổng số lượng ô dữ liệu bị khuyết thiếu (missing values)
    total_nulls = df.isnull().sum().sum()
    # In tổng số giá trị khuyết thiếu được tìm thấy
    print(f"[DATA LOADER] Tổng số giá trị khuyết thiếu (Missing/Null): {total_nulls}")
    # Trả về đối tượng DataFrame đã nạp
    return df
