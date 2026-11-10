# Import thư viện pandas để xử lý các cấu trúc dữ liệu bảng
import pandas as pd
# Import thư viện numpy để thực hiện các phép toán ma trận và điều kiện
import numpy as np
# Import thư viện typing để chú thích kiểu dữ liệu
from typing import Dict, Any, List

# Lớp tính điểm thưởng và gợi ý ưu đãi cá nhân hóa dựa trên hành vi và hiệu ứng nhân quả
class CustomerBehaviorScorer:
    # Hàm khởi tạo lớp CustomerBehaviorScorer với các bảng tra hệ số và ngưỡng rủi ro
    def __init__(self, base_multipliers: Dict[str, float], expense_bonuses: Dict[str, float], risk_thresholds: Dict[str, float]):
        # Lưu trữ bảng hệ số tính điểm thưởng theo hạng thẻ tín dụng
        self.base_multipliers = base_multipliers
        # Lưu trữ bảng hệ số thưởng theo danh mục chi tiêu chủ đạo
        self.expense_bonuses = expense_bonuses
        # Lưu trữ các ngưỡng chỉ số an toàn tài chính
        self.risk_thresholds = risk_thresholds

    # Hàm tính toán điểm thưởng động đa biến dựa trên hành vi chi tiêu và hạng thẻ
    def calculate_dynamic_reward_points(self, df: pd.DataFrame) -> pd.Series:
        # In thông báo bắt đầu tính toán điểm thưởng
        print("[SCORING] Đang tính toán điểm thưởng động dựa trên hệ số hạng thẻ và chi tiêu...")
        # Tính hệ số điểm thưởng cơ bản dựa vào hạng thẻ bằng cách ánh xạ qua từ điển
        base_factor = df["Card_Type"].map(self.base_multipliers).fillna(1.0)
        # Tính hệ số thưởng danh mục dựa vào danh mục chi tiêu có giá trị lớn nhất
        category_bonus = df["Dominant_Expense_Category"].map(self.expense_bonuses).fillna(0.5)
        # Tính điểm thưởng khuyến khích cho khách hàng có mức độ tương tác ứng dụng di động cao
        app_engagement_bonus = np.where(df["Mobile_App_Login"] >= 30, 0.2, np.where(df["Mobile_App_Login"] >= 15, 0.1, 0.0))
        # Tính hệ số giảm trừ điểm thưởng đối với khách hàng có tỷ lệ thanh toán nợ thấp
        late_payment_penalty = np.where(df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"], 0.5, 0.0)
        # Tính tổng hệ số thưởng cuối cùng sau khi cộng thưởng và trừ phạt
        total_multiplier = (base_factor + category_bonus + app_engagement_bonus - late_payment_penalty).clip(lower=0.5)
        # Tính số điểm thưởng động tỷ lệ thuận với số tiền chi tiêu hàng tháng
        calculated_points = (df["Monthly_Spending"] / 500.0) * total_multiplier
        # Làm tròn điểm thưởng đến 4 chữ số thập phân
        reward_points = calculated_points.round(4)
        # In thông báo đã hoàn thành tính toán điểm thưởng
        print(f"[SCORING] Hoàn tất tính điểm thưởng! Giá trị trung bình toàn bộ dữ liệu: {reward_points.mean():.4f} điểm.")
        # Trả về chuỗi điểm thưởng động của tất cả khách hàng
        return reward_points

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

    # Hàm tổng hợp báo cáo phân tích chi tiết cho từng khách hàng tích hợp Uplift Modeling
    def generate_customer_report(self, df: pd.DataFrame, uplift_scores: np.ndarray, uplift_segments: pd.Series) -> pd.DataFrame:
        # In thông báo tạo báo cáo khách hàng
        print("[REPORT] Đang tổng hợp báo cáo phân tích chi tiết tích hợp phân khúc Uplift cho toàn bộ khách hàng...")
        # Tạo danh sách các điều kiện kiểm tra rủi ro tín dụng
        conditions = [
            df["Credit_Utilization"] > self.risk_thresholds["max_credit_utilization"],
            df["Credit_Score"] < self.risk_thresholds["min_credit_score"],
            df["Payment_Ratio"] < self.risk_thresholds["min_payment_ratio"]
        ]
        # Tạo danh sách các nhãn rủi ro tương ứng theo thứ tự ưu tiên
        choices = ["HIGH_RISK_UTILIZATION", "HIGH_RISK_CREDIT_SCORE", "HIGH_RISK_PAYMENT_RATIO"]
        # Gán nhãn trạng thái rủi ro sử dụng phép chọn theo điều kiện của numpy
        risk_status = np.select(conditions, choices, default="SAFE")

        # Danh sách các cột cần trình bày trong báo cáo chi tiết
        report_columns = [
            # Mã định danh khách hàng duy nhất
            "Customer_ID",
            # Nhãn cụm phân loại hành vi K-Means
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
        ]
        # Tạo DataFrame báo cáo chứa các cột đã chọn
        report_df = df[report_columns].copy()
        # Làm tròn các cột tỷ lệ và số thực đến 4 chữ số thập phân
        report_df["Monthly_Spending"] = report_df["Monthly_Spending"].round(4)
        report_df["Credit_Utilization"] = report_df["Credit_Utilization"].round(4)
        report_df["Payment_Ratio"] = report_df["Payment_Ratio"].round(4)
        report_df["Dynamic_Reward_Points"] = report_df["Dynamic_Reward_Points"].round(4)
        report_df["Discretionary_Ratio"] = report_df["Discretionary_Ratio"].round(4)
        report_df["Essential_Ratio"] = report_df["Essential_Ratio"].round(4)
        report_df["Redemption_Rate"] = report_df["Redemption_Rate"].round(4)
        # Thêm cột trạng thái rủi ro chi tiết
        report_df["Risk_Status"] = risk_status
        # Tích hợp cột điểm hiệu ứng nhân quả Uplift Score làm tròn 4 chữ số thập phân
        report_df["Uplift_Score"] = np.round(uplift_scores, 4)
        # Tích hợp cột phân khúc nhân quả Uplift Segment chuẩn tiếng Anh
        report_df["Uplift_Segment"] = uplift_segments.values
        # In thông báo hoàn tất báo cáo
        print(f"[REPORT] Đã tổng hợp thành công báo cáo cho toàn bộ {len(report_df)} khách hàng tích hợp Uplift!")
        # Trả về bảng báo cáo chi tiết
        return report_df

    # Hàm phân tích chuyên sâu cho một khách hàng cụ thể
    def analyze_customer(self, customer_id: str, df: pd.DataFrame) -> Dict[str, Any]:
        # Lọc thông tin khách hàng theo mã định danh Customer_ID
        cust_match = df[df['Customer_ID'] == customer_id]
        # Kiểm tra nếu không tìm thấy khách hàng
        if cust_match.empty:
            # Trả về thông báo lỗi
            return {"error": f"Không tìm thấy mã khách hàng: {customer_id}"}
        # Lấy bản ghi thông tin khách hàng
        customer_row = cust_match.iloc[0]
        # Kiểm tra trạng thái rủi ro tín dụng
        risk_status = self.check_risk_status(customer_row)

        # Xây dựng đề xuất cá nhân hóa dựa trên phân khúc nhân quả
        uplift_segment = customer_row.get('Uplift_Segment', 'N/A')
        uplift_score = customer_row.get('Uplift_Score', 0.0)

        # Phân nhánh chiến lược can thiệp ưu đãi
        if uplift_segment == "Persuadables":
            recommendation_strategy = "Mục tiêu vàng: Cấp gói Voucher hoàn tiền và nhân 3x điểm thưởng danh mục chủ đạo."
        elif uplift_segment == "Sure Things":
            recommendation_strategy = "Khách hàng tự thân chi tiêu cao: Chăm sóc thẻ Signature/Platinum tự nhiên, tiết kiệm ngân sách voucher."
        elif uplift_segment == "Lost Causes":
            recommendation_strategy = "Rào cản an toàn (Risk Gate): Ngừng kích cầu mua sắm, đề xuất gói trả góp 0% (EMI) bảo toàn vốn."
        else:
            recommendation_strategy = "Tránh làm phiền: Duy trì kênh tương tác thụ động, không gửi tin nhắn quảng cáo trực tiếp."

        # Trả về từ điển kết quả phân tích cá nhân
        return {
            "customer_id": customer_id,
            "cluster": int(customer_row['Cluster']),
            "risk_status": risk_status,
            "uplift_score": round(float(uplift_score), 4),
            "uplift_segment": uplift_segment,
            "card_type": customer_row['Card_Type'],
            "spending": round(float(customer_row['Monthly_Spending']), 4),
            "recommendation_strategy": recommendation_strategy
        }
