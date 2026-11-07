# Import mo hinh Hoi quy tuyen tinh LinearRegression tu scikit-learn
from sklearn.linear_model import LinearRegression
# Import mo hinh Random Forest Regressor tap hop cay quyet dinh
from sklearn.ensemble import RandomForestRegressor
# Import mo hinh Gradient Boosting toi uu LightGBM Regressor
from lightgbm import LGBMRegressor
# Import cac chi so danh gia do chinh xac hoi quy: R2, RMSE, MAE
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
# Import ham chia tach du lieu train_test_split tu scikit-learn
from sklearn.model_selection import train_test_split
# Import thu vien pandas de tao bang so sanh doi chuan
import pandas as pd
# Import thu vien numpy de tinh toan ma tran va mang so hoc
import numpy as np
# Import thu vien typing de chu thich kieu tra ve
from typing import Dict, Any, Tuple

# Lop quan ly kiem chung he thong tinh diem thuong bang cac thuat toan Machine Learning thuan tuy
class RewardValidationEngine:
    # Ham khoi tao lop kiem chung voi hat giong ngau nhien
    def __init__(self, random_state: int = 42):
        # Luu gia tri random_state de co dinh ket qua
        self.random_state = random_state
        # Khoi tao tu dien chua 3 mo hinh Machine Learning co dien va tang tien de so sanh doi chuan
        self.models = {
            # Mo hinh Hoi quy tuyen tinh dong vai tro mo hinh co so
            'Linear Regression': LinearRegression(),
            # Mo hinh Random Forest tap hop cay song song triet tieu phuong sai
            'Random Forest Regressor': RandomForestRegressor(n_estimators=50, max_depth=12, random_state=self.random_state, n_jobs=-1),
            # Mo hinh LightGBM Gradient Boosting toi uu chuoi phan du
            'LightGBM Regressor': LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=self.random_state, verbose=-1, n_jobs=-1)
        # Ket thuc khoi danh sach hoac tu dien
        }
        # Bang luu ket qua danh gia doi chuan
        self.benchmark_results = None

    # Ham thuc hien huan luyen va danh gia doi chuan 3 mo hinh Machine Learning
    def run_benchmark(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        # In thong bao bat dau qua trinh doi chuan mo hinh
        print('[VALIDATION] Bat dau kiem chung he thong bang Linear Regression vs Random Forest vs LightGBM...')
        # Chia tap du lieu thanh tap Huan luyen Train (80%) va tap Kiem thu Test (20%)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=self.random_state)
        # Danh sach luu ket qua do luong cua tung mo hinh
        results_list = []
        # Lap qua tung mo hinh trong danh sach
        for model_name, model in self.models.items():
            # In thong bao dang huan luyen mo hinh hien tai
            print(f'[VALIDATION] Dang huan luyen mo hinh: {model_name}...')
            # Huan luyen mo hinh tren tap Train (80% du lieu)
            model.fit(X_train, y_train)
            # Thuc hien du doan tren tap Test doc lap (20% du lieu)
            y_pred = model.predict(X_test)
            # Tinh he so xac dinh R2 Score (cang gan 1.0 mo hinh cang chinh xac)
            r2 = float(r2_score(y_test, y_pred))
            # Tinh sai so can quan phuong trung binh RMSE
            rmse = float(root_mean_squared_error(y_test, y_pred))
            # Tinh sai so tuyet doi trung binh MAE
            mae = float(mean_absolute_error(y_test, y_pred))
            # In ket qua do luong cua mo hinh ra man hinh console
            print(f'[VALIDATION] -> {model_name}: R2 = {r2:.4f} | RMSE = {rmse:.2f} | MAE = {mae:.2f}')
            # Luu thong so danh gia vao danh sach ket qua
            results_list.append({
                # Ten mo hinh dang danh gia
                'Model': model_name,
                # He so xac dinh R2 Score
                'R2_Score': round(r2, 4),
                # Sai so can quan phuong trung binh RMSE
                'RMSE': round(rmse, 2),
                # Sai so tuyet doi trung binh MAE
                'MAE': round(mae, 2)
            # Ket thuc khoi danh sach hoac tu dien
            })
        # Chuyen danh sach ket qua thanh DataFrame cua pandas
        self.benchmark_results = pd.DataFrame(results_list)
        # In thong bao hoan tat qua trinh doi chuan
        print('[VALIDATION] Hoan tat doi chuan mo hinh Machine Learning thanh cong!')
        # Tra ve bang ket qua so sanh doi chuan
        return self.benchmark_results
