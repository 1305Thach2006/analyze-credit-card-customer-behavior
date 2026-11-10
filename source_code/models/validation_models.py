# Import mô hình Hồi quy tuyến tính LinearRegression từ scikit-learn
from sklearn.linear_model import LinearRegression
# Import mô hình Random Forest Regressor tập hợp cây quyết định
from sklearn.ensemble import RandomForestRegressor
# Import mô hình học máy tập hợp xếp chồng StackingRegressor
from sklearn.ensemble import StackingRegressor
# Import mô hình Gradient Boosting tối ưu LightGBM Regressor
from lightgbm import LGBMRegressor
# Import các chỉ số đánh giá độ chính xác hồi quy: R2, RMSE, MAE
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
# Import hàm chia tách dữ liệu train_test_split từ scikit-learn
from sklearn.model_selection import train_test_split
# Import thư viện pandas để tạo bảng so sánh đối chuẩn
import pandas as pd
# Import thư viện numpy để tính toán ma trận và mảng số học
import numpy as np
# Import thư viện typing để chú thích kiểu trả về
from typing import Dict, Any, Tuple

# Lớp quản lý kiểm chứng và đối chuẩn hệ thống bằng các thuật toán Machine Learning và mô hình Lai
class RewardValidationEngine:
    # Hàm khởi tạo lớp kiểm chứng với hạt giống ngẫu nhiên
    def __init__(self, random_state: int = 42):
        # Lưu giá trị random_state để cố định kết quả
        self.random_state = random_state
        # Khởi tạo mô hình Random Forest cơ sở cho bộ ensemble
        rf_base = RandomForestRegressor(n_estimators=50, max_depth=12, random_state=self.random_state, n_jobs=-1)
        # Khởi tạo mô hình LightGBM cơ sở cho bộ ensemble
        lgbm_base = LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=self.random_state, verbose=-1, n_jobs=-1)
        # Khởi tạo mô hình lai Stacking Hybrid Ensemble kết hợp Bagging và Boosting qua Meta-Learner
        stacking_hybrid = StackingRegressor(
            # Danh sách 2 mô hình học máy nền tảng mạnh nhất
            estimators=[
                # Mô hình Random Forest đại diện trường phái Bagging
                ('Random Forest', rf_base),
                # Mô hình LightGBM đại diện trường phái Boosting
                ('LightGBM', lgbm_base)
            # Kết thúc danh sách bộ ước lượng cơ sở
            ],
            # Mô hình Hồi quy tuyến tính đóng vai trò trọng tài tối ưu hóa trọng số phần dư
            final_estimator=LinearRegression(),
            # Thực hiện kiểm định chéo 3-fold để sinh dự đoán out-of-fold cho Meta-Learner
            cv=3,
            # Chạy đa luồng song song để tối ưu thời gian tính toán
            n_jobs=-1
        # Kết thúc khởi tạo StackingRegressor
        )
        # Khởi tạo từ điển chứa 4 mô hình Machine Learning và mô hình Lai để so sánh đối chuẩn
        self.models = {
            # Mô hình Hồi quy tuyến tính đóng vai trò mô hình cơ sở
            'Linear Regression': LinearRegression(),
            # Mô hình Random Forest tập hợp cây song song triệt tiêu phương sai
            'Random Forest Regressor': rf_base,
            # Mô hình LightGBM Gradient Boosting tối ưu chuỗi phần dư
            'LightGBM Regressor': lgbm_base,
            # Mô hình lai Stacking Hybrid Ensemble kết hợp Bagging và Boosting
            'Stacking Hybrid Ensemble': stacking_hybrid
        # Kết thúc từ điển danh sách mô hình
        }
        # Bảng lưu kết quả đánh giá đối chuẩn
        self.benchmark_results = None

    # Hàm thực hiện huấn luyện và đánh giá đối chuẩn 4 mô hình Machine Learning
    def run_benchmark(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        # In thông báo bắt đầu quá trình đối chuẩn mô hình
        print('[VALIDATION] Bắt đầu kiểm chứng hệ thống bằng Linear Regression vs Random Forest vs LightGBM vs Stacking Hybrid...')
        # Chia tập dữ liệu thành tập Huấn luyện Train (80%) và tập Kiểm thử Test (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=self.random_state)
        # Danh sách lưu kết quả đo lường của từng mô hình
        results_list = []
        # Lặp qua từng mô hình trong danh sách
        for model_name, model in self.models.items():
            # In thông báo đang huấn luyện mô hình hiện tại
            print(f'[VALIDATION] Đang huấn luyện mô hình: {model_name}...')
            # Huấn luyện mô hình trên tập Train (80% dữ liệu)
            model.fit(X_train, y_train)
            # Thực hiện dự đoán trên tập Test độc lập (20% dữ liệu)
            y_pred = model.predict(X_test)
            # Tính hệ số xác định R2 Score (càng gần 1.0 mô hình càng chính xác)
            r2 = float(r2_score(y_test, y_pred))
            # Tính sai số căn quân phương trung bình RMSE
            rmse = float(root_mean_squared_error(y_test, y_pred))
            # Tính sai số tuyệt đối trung bình MAE
            mae = float(mean_absolute_error(y_test, y_pred))
            # In kết quả đo lường của mô hình ra màn hình console với độ chính xác 4 chữ số thập phân
            print(f'[VALIDATION] -> {model_name}: R2 = {r2:.4f} | RMSE = {rmse:.4f} | MAE = {mae:.4f}')
            # Lưu thông số đánh giá vào danh sách kết quả làm tròn 4 chữ số thập phân
            results_list.append({
                # Tên mô hình đang đánh giá
                'Model': model_name,
                # Hệ số xác định R2 Score
                'R2_Score': round(r2, 4),
                # Sai số căn quân phương trung bình RMSE
                'RMSE': round(rmse, 4),
                # Sai số tuyệt đối trung bình MAE
                'MAE': round(mae, 4)
            # Kết thúc phần tử kết quả đánh giá
            })
        # Chuyển danh sách kết quả thành DataFrame của pandas
        self.benchmark_results = pd.DataFrame(results_list)
        # In thông báo hoàn tất quá trình đối chuẩn
        print('[VALIDATION] Hoàn tất đối chuẩn mô hình Machine Learning thành công!')
        # Trả về bảng kết quả so sánh đối chuẩn
        return self.benchmark_results
