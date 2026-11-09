# Import thư viện os để thao tác với đường dẫn hệ thống
import os

# Đường dẫn tuyệt đối đến tệp dữ liệu gốc
DATA_PATH = r"D:\PTDL\dataset\synthetic_credit_card_customer_behavior_dataset.csv"

# Thư mục gốc chứa các kết quả đầu ra
OUTPUT_DIR = r"D:\PTDL\source_code\outputs"

# Thư mục lưu trữ các biểu đồ trực quan hóa dưới dạng hình ảnh
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# Thư mục lưu trữ các báo cáo thống kê dưới dạng tập tin CSV
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")

# Giá trị hạt giống ngẫu nhiên để cố định kết quả thực thi
RANDOM_STATE = 42

# Số lượng cụm lựa chọn cho thuật toán phân cụm K-Means
N_CLUSTERS = 4

# Danh sách các cột chi tiêu chi tiết theo 7 danh mục trong dữ liệu gốc
SPENDING_CATEGORIES = [
    # Cột chi tiêu mua sắm trực tuyến
    "Online_Shopping_Spending",
    # Cột chi tiêu tạp hóa và thực phẩm
    "Grocery_Spending",
    # Cột chi tiêu nhiên liệu xăng dầu
    "Fuel_Spending",
    # Cột chi tiêu ăn uống nhà hàng
    "Dining_Spending",
    # Cột chi tiêu du lịch khách sạn
    "Travel_Spending",
    # Cột chi tiêu giải trí xem phim
    "Entertainment_Spending",
    # Cột chi tiêu hóa đơn điện nước sinh hoạt
    "Utility_Bill_Spending"
# Kết thúc danh sách danh mục chi tiêu gốc
]

# Danh sách các đặc trưng tỷ trọng chi tiêu được tạo mới
RATIO_CATEGORIES = [
    # Tỷ trọng chi tiêu mua sắm trực tuyến
    "Pct_Online",
    # Tỷ trọng chi tiêu tạp hóa thực phẩm
    "Pct_Grocery",
    # Tỷ trọng chi tiêu nhiên liệu xăng dầu
    "Pct_Fuel",
    # Tỷ trọng chi tiêu ăn uống nhà hàng
    "Pct_Dining",
    # Tỷ trọng chi tiêu du lịch khách sạn
    "Pct_Travel",
    # Tỷ trọng chi tiêu giải trí
    "Pct_Entertainment",
    # Tỷ trọng chi tiêu hóa đơn sinh hoạt
    "Pct_Utility"
# Kết thúc danh sách đặc trưng tỷ trọng chi tiêu
]

# Hệ số tính điểm thưởng cơ bản theo từng hạng thẻ tín dụng
BASE_REWARD_MULTIPLIERS = {
    # Hạng thẻ Signature cao cấp nhất nhân hệ số gấp đôi
    "Signature": 2.0,
    # Hạng thẻ Platinum nhận hệ số cao
    "Platinum": 1.5,
    # Hạng thẻ Gold nhận hệ số tiêu chuẩn
    "Gold": 1.0,
    # Hạng thẻ Silver nhận hệ số cơ bản
    "Silver": 0.5,
    # Hạng thẻ Basic phổ thông nhận hệ số cơ bản
    "Basic": 0.5
# Kết thúc bảng hệ số hạng thẻ
}

# Hệ số thưởng thêm theo danh mục chi tiêu có giá trị lớn nhất
EXPENSE_BONUS_FACTORS = {
    # Ưu đãi đặc biệt cho chi tiêu du lịch
    "Travel_Spending": 3.0,
    # Ưu đãi đặc biệt cho chi tiêu nhà hàng ăn uống
    "Dining_Spending": 3.0,
    # Ưu đãi cho chi tiêu mua sắm trực tuyến
    "Online_Shopping_Spending": 2.0,
    # Ưu đãi cho chi tiêu hoạt động giải trí
    "Entertainment_Spending": 1.5,
    # Ưu đãi cơ bản cho chi tiêu tạp hóa
    "Grocery_Spending": 0.5,
    # Ưu đãi cơ bản cho hóa đơn điện nước
    "Utility_Bill_Spending": 0.5,
    # Ưu đãi cơ bản cho chi tiêu đổ xăng
    "Fuel_Spending": 0.5
# Kết thúc bảng hệ số danh mục chi tiêu
}

# Các ngưỡng chỉ số tài chính để xác định rủi ro tín dụng
RISK_THRESHOLDS = {
    # Ngưỡng trần tỷ lệ sử dụng hạn mức tín dụng cho phép
    "max_credit_utilization": 0.75,
    # Ngưỡng sàn điểm tín dụng tối thiểu an toàn
    "min_credit_score": 550,
    # Ngưỡng sàn tỷ lệ thanh toán nợ tối thiểu an toàn
    "min_payment_ratio": 0.40
# Kết thúc bảng ngưỡng rủi ro tín dụng
}
