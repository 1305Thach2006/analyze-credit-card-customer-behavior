# Import thư viện pandas để xử lý các cấu trúc dữ liệu bảng
import pandas as pd
# Import thư viện numpy để thực hiện các phép toán ma trận và điều kiện
import numpy as np
# Import thư viện NearestNeighbors từ scikit-learn để tìm kiếm láng giềng tương đồng
from sklearn.neighbors import NearestNeighbors
# Import thư viện typing để chú thích kiểu dữ liệu
from typing import Dict, Any, List

# Lớp tính điểm thưởng và gợi ý ưu đãi dựa trên hành vi khách hàng
class CustomerBehaviorScorer:
    # Hàm khởi tạo lớp CustomerBehaviorScorer với các bảng tra hệ số và ngưỡng
    def __init__(self, base_multipliers: Dict[str, float], expense_bonuses: Dict[str, float], risk_thresholds: Dict[str, float]):
        # Lưu trữ bảng hệ số tính điểm thưởng theo hạng thẻ tín dụng
        self.base_multipliers = base_multipliers
        # Lưu trữ bảng hệ số thưởng theo danh mục chi tiêu chủ đạo
        self.expense_bonuses = expense_bonuses
        # Lưu trữ các ngưỡng chỉ số an toàn tài chính
        self.risk_thresholds = risk_thresholds
        # Khởi tạo từ điển để lưu trữ mô hình k-NN riêng cho từng cụm
        self.cluster_knn_models = {}

    # Hàm tính toán điểm thưởng động đa biến dựa trên hành vi chi tiêu và hạng thẻ
    def calculate_dynamic_reward_points(self, df: pd.DataFrame) -> pd.Series:
        # In thông báo bắt đầu tính toán điểm thưởng
        print("[SCORING] Đang tính toán điểm thưởng động dựa trên hệ số hạng thẻ và chi tiêu...")
        # Tính hệ số điểm thưởng cơ bản dựa vào hạng thẻ bằng cách ánh xạ qua từ điển
        base_factor = df["Card_Type"].map(self.base_multipliers).fillna(1.0)
        # Tính hệ số thưởng danh mục dựa vào danh mục chi tiêu có giá trị lớn nhất
        category_bonus = df["Dominant_Expense_Category"].map(self.expense_bonuses).fillna(0.5)
        # Tính điểm thưởng khuyến khích cho khách hàng có mức độ tương tác ứng dụng di động cao
        app_engagement_bonus = np.where(df["Mobile_App_Login"] >= 10, 0.2, np.where(df["Mobile_App_Login"] >= 5, 0.1, 0.0))
        # Tính hệ số giảm trừ điểm thưởng đối với khách hàng có tỷ lệ thanh toán nợ thấp
        late_payment_penalty = np.where(df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"], 0.5, 0.0)
        # Tính tổng hệ số thưởng cuối cùng sau khi cộng thưởng và trừ phạt
        total_multiplier = (base_factor + category_bonus + app_engagement_bonus - late_payment_penalty).clip(lower=0.5)
        # Tính số điểm thưởng động tỷ lệ thuận với số tiền chi tiêu hàng tháng
        calculated_points = (df["Monthly_Spending"] / 500.0) * total_multiplier
        # Làm tròn điểm thưởng đến chữ số thập phân thứ nhất
        reward_points = calculated_points.round(2)
        # In thông báo đã hoàn thành tính toán điểm thưởng
        print(f"[SCORING] Hoàn tất tính điểm thưởng! Giá trị trung bình toàn bộ dữ liệu: {reward_points.mean():.2f} điểm.")
        # Trả về chuỗi điểm thưởng động của tất cả khách hàng
        return reward_points

    # Hàm khởi tạo và huấn luyện mô hình k-NN độc lập cho từng cụm khách hàng
    def fit_knn_by_cluster(self, df: pd.DataFrame, feature_cols: List[str], cluster_col: str = 'Cluster') -> None:
        # In thông báo huấn luyện k-NN tìm láng giềng
        print("[SCORING] Huấn luyện thuật toán k-NN tìm láng giềng tương đồng theo cụm...")
        # Lặp qua từng cụm trong tập dữ liệu
        for c in df[cluster_col].unique():
            # Lọc tập dữ liệu con thuộc cụm
            cluster_subset = df[df[cluster_col] == c]
            # Khởi tạo mô hình NearestNeighbors với k=5
            knn = NearestNeighbors(n_neighbors=min(5, len(cluster_subset)), metric='euclidean')
            # Huấn luyện mô hình k-NN trên tập đặc trưng của cụm
            knn.fit(cluster_subset[feature_cols].values)
            # Lưu trữ mô hình và danh sách mã khách hàng tương ứng vào từ điển
            self.cluster_knn_models[c] = {'model': knn, 'customer_ids': cluster_subset['Customer_ID'].values, 'indices': cluster_subset.index.values}
        # In thông báo đã hoàn thành huấn luyện k-NN
        print("[SCORING] Khởi tạo thuật toán k-NN hoàn tất!")

    # Hàm kiểm tra điều kiện rủi ro tín dụng dựa trên các ngưỡng chỉ số
    def check_risk_status(self, customer_row: pd.Series) -> str:
        # Kiểm tra nếu tỷ lệ sử dụng hạn mức vượt 75%
        if customer_row["Credit_Utilization"] > self.risk_thresholds["max_credit_utilization"]:
            # Trả về trạng thái cảnh báo rủi ro cao
            return "HIGH_RISK_UTILIZATION"
        # Kiểm tra nếu điểm tín dụng dưới 550
        if customer_row["Credit_Score"] < self.risk_thresholds["min_credit_score"]:
            # Trả về trạng thái cảnh báo điểm tín dụng thấp
            return "HIGH_RISK_CREDIT_SCORE"
        # Kiểm tra nếu tỷ lệ thanh toán dưới 40%
        if customer_row["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"]:
            # Trả về trạng thái cảnh báo chậm trả
            return "HIGH_RISK_PAYMENT_RATIO"
        # Trả về trạng thái an toàn nếu vượt qua tất cả các kiểm tra
        return "SAFE"

    # Hàm tổng hợp báo cáo phân tích chi tiết cho từng khách hàng trên toàn bộ tập dữ liệu
    def generate_customer_report(self, df: pd.DataFrame) -> pd.DataFrame:
        # In thông báo tạo báo cáo khách hàng
        print("[REPORT] Đang tổng hợp báo cáo phân tích chi tiết cho toàn bộ từng khách hàng...")
        # Tạo danh sách các điều kiện kiểm tra rủi ro tín dụng
        conditions = [df["Credit_Utilization"] > self.risk_thresholds["max_credit_utilization"], df["Credit_Score"] < self.risk_thresholds["min_credit_score"], df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"]]
        # Tạo danh sách các nhãn rủi ro tương ứng theo thứ tự ưu tiên
        choices = ["HIGH_RISK_UTILIZATION", "HIGH_RISK_CREDIT_SCORE", "HIGH_RISK_PAYMENT_RATIO"]
        # Gán nhãn trạng thái rủi ro sử dụng phép chọn theo điều kiện của numpy
        risk_status = np.select(conditions, choices, default="SAFE")
        # Danh sách các cột cần trình bày trong báo cáo chi tiết
        report_columns = [
            # Mã định danh khách hàng duy nhất
            "Customer_ID",
            # Nhãn cụm phân loại hành vi
            "Cluster",
            # Hạng thẻ tín dụng hiện tại
            "Card_Type",
            # Tổng số tiền chi tiêu hàng tháng
            "Monthly_Spending",
            # Danh mục chi tiêu có giá trị lớn nhất
            "Dominant_Expense_Category",
            # Hạn mức tín dụng được ngân hàng cấp
            "Credit_Limit",
            # Tỷ lệ sử dụng hạn mức tín dụng
            "Credit_Utilization",
            # Điểm uy tín tín dụng
            "Credit_Score",
            # Tỷ lệ thanh toán dư nợ hàng tháng
            "Payment_Ratio",
            # Điểm thưởng động đã được tính toán
            "Dynamic_Reward_Points",
            # Cờ nhị phân cảnh báo rủi ro
            "Risk_Flag",
            # Tỷ trọng chi tiêu cho các danh mục tùy ý
            "Discretionary_Ratio",
            # Tỷ trọng chi tiêu cho các danh mục thiết yếu
            "Essential_Ratio",
            # Tỷ lệ đổi điểm thưởng
            "Redemption_Rate"
        # Kết thúc danh sách cột báo cáo
        ]
        # Tạo DataFrame báo cáo chứa các cột đã chọn
        report_df = df[report_columns].copy()
        # Thêm cột trạng thái rủi ro chi tiết
        report_df["Risk_Status"] = risk_status
        # In thông báo hoàn tất báo cáo
        print(f"[REPORT] Đã tổng hợp thành công báo cáo cho toàn bộ {len(report_df)} khách hàng!")
        # Trả về bảng báo cáo chi tiết
        return report_df

    # Hàm phân tích chuyên sâu cho một khách hàng cụ thể
    def analyze_customer(self, customer_id: str, df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
        # Lọc thông tin khách hàng theo mã định danh Customer_ID
        cust_match = df[df['Customer_ID'] == customer_id]
        # Kiểm tra nếu không tìm thấy khách hàng
        if cust_match.empty:
            # Trả về thông báo lỗi
            return {"error": f"Không tìm thấy mã khách hàng: {customer_id}"}
        # Lấy bản ghi thông tin khách hàng
        customer_row = cust_match.iloc[0]
        # Lấy nhãn cụm của khách hàng
        cluster_id = customer_row['Cluster']
        # Kiểm tra trạng thái rủi ro tín dụng
        risk_status = self.check_risk_status(customer_row)
        # Danh sách mã khách hàng láng giềng tương đồng
        neighbor_ids = []
        # Kiểm tra nếu đã huấn luyện mô hình k-NN cho cụm này
        if cluster_id in self.cluster_knn_models:
            # Lấy thông tin mô hình k-NN của cụm
            knn_info = self.cluster_knn_models[cluster_id]
            # Lấy vector đặc trưng của khách hàng
            cust_vec = customer_row[feature_cols].values.reshape(1, -1)
            # Tìm kiếm 5 láng giềng gần nhất
            distances, indices = knn_info['model'].kneighbors(cust_vec)
            # Lấy danh sách mã khách hàng tương ứng với các chỉ số tìm được
            neighbor_ids = list(knn_info['customer_ids'][indices[0]])
            # Loại bỏ chính mã khách hàng hiện tại nếu có trong danh sách láng giềng
            if customer_id in neighbor_ids:
                # Xóa mã khách hàng khỏi danh sách láng giềng
                neighbor_ids.remove(customer_id)
        # Tạo hồ sơ phân tích tổng hợp của khách hàng
        profile = {
            # Mã khách hàng
            "Customer_ID": customer_id,
            # Nhãn cụm
            "Cluster": int(cluster_id),
            # Hạng thẻ tín dụng
            "Card_Type": customer_row["Card_Type"],
            # Chi tiêu hàng tháng
            "Monthly_Spending": float(customer_row["Monthly_Spending"]),
            # Danh mục chủ đạo
            "Dominant_Expense_Category": customer_row["Dominant_Expense_Category"],
            # Điểm thưởng động
            "Dynamic_Reward_Points": float(customer_row["Dynamic_Reward_Points"]),
            # Trạng thái rủi ro
            "Risk_Status": risk_status,
            # Cờ rủi ro nhị phân
            "Risk_Flag": int(customer_row["Risk_Flag"]),
            # Danh sách 5 láng giềng gần nhất
            "Similar_Neighbors": neighbor_ids[:5]
        # Kết thúc hồ sơ khách hàng
        }
        # Trả về hồ sơ khách hàng
        return profile
