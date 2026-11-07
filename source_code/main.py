# Import thu vien os de thao tac voi thu muc va duong dan tep
import os
# Import thu vien sys de cau hinh duong dan module trong he thong
import sys
# Import thu vien pandas de thao tac va quan ly bang du lieu
import pandas as pd
# Import thu vien numpy de tinh toan ma tran va mang so hoc
import numpy as np
# Import thu vien StandardScaler tu scikit-learn de chuan hoa du lieu
from sklearn.preprocessing import StandardScaler

# Them thu muc hien tai vao duong dan tim kiem module cua Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import cac thiet lap duong dan tu config/settings.py
from config.settings import DATA_PATH, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR
# Import cac tham so thuat toan va trong so tu config/settings.py
from config.settings import RANDOM_STATE, N_CLUSTERS, SPENDING_CATEGORIES, RATIO_CATEGORIES
# Import cac he so thuong va nguong rui ro tu config/settings.py
from config.settings import BASE_REWARD_MULTIPLIERS, EXPENSE_BONUS_FACTORS, RISK_THRESHOLDS

# Import ham nap du lieu tu data_pipeline/loader.py
from data_pipeline.loader import load_credit_card_data
# Import ham tao dac trung nghiep vu tu data_pipeline/feature_engineering.py
from data_pipeline.feature_engineering import engineer_credit_features
# Import ham chuan hoa du lieu tu data_pipeline/preprocessor.py
from data_pipeline.preprocessor import preprocess_features

# Import cac ham ve bieu do tu eda/visualizer.py
from eda.visualizer import plot_correlation_heatmap, plot_category_spending_boxplot, plot_pca_clusters, plot_model_benchmark

# Import bo phan cum khach hang K-Means va PCA tu models/clustering.py
from models.clustering import CustomerClustering
# Import dong co tinh diem thuong va goi y tu models/recommender.py
from models.recommender import CustomerBehaviorScorer
# Import lop doi chuan cac mo hinh Machine Learning tu models/validation_models.py
from models.validation_models import RewardValidationEngine

# Ham luu DataFrame an toan tranh loi bi khoa tep khi mo tren Excel
def save_dataframe_safely(df: pd.DataFrame, file_path: str) -> str:
    # Khoi lenh thu luu tep CSV theo duong dan goc
    try:
        # Ghi du lieu DataFrame ra tep CSV khong kem cot chi muc
        df.to_csv(file_path, index=False)
        # In thong bao luu tep thanh cong
        print(f'[REPORT] Da luu thanh cong bao cao tai: {file_path}')
        # Tra ve duong dan tep da luu
        return file_path
    # Bat ngoai le PermissionError khi tep dang bi khoa boi chuong trinh khac
    except PermissionError:
        # Tach ten tep va phan mo rong
        base, ext = os.path.splitext(file_path)
        # Tao duong dan tep thay the voi hau to _updated
        alt_path = f'{base}_updated{ext}'
        # Ghi du lieu DataFrame sang tep thay the
        df.to_csv(alt_path, index=False)
        # In canh bao nguoi dung ve viec tep bi khoa va luu sang duong dan moi
        print(f'[CANH BAO] Tap tin {os.path.basename(file_path)} dang mo trong Excel. Da luu vao: {alt_path}')
        # Tra ve duong dan tep moi
        return alt_path

# Ham dieu phoi toan bo luong phan tich du lieu va mo hinh hoa
def main():
    # In duong ke ngan cach tieu de
    print('=' * 80)
    # In tieu de quy trinh phan tich
    print('  MO HINH PHAN TICH HANH VI TIEU DUNG THE TIN DUNG VA DE XUAT CHIEN LUOC UU DAI CA NHAN')
    # In duong ke ngan cach
    print('=' * 80)

    # -------------------------------------------------------------
    # BUOC 1: NAP DU LIEU (DATA INGESTION)
    # -------------------------------------------------------------
    # Goi ham doc du lieu tu tep CSV goc
    raw_df = load_credit_card_data(DATA_PATH)

    # -------------------------------------------------------------
    # BUOC 2: CHE TAC DAC TRUNG (FEATURE ENGINEERING)
    # -------------------------------------------------------------
    # Tao cac ty trong chi tieu va gan co rui ro
    df_engineered = engineer_credit_features(raw_df)

    # -------------------------------------------------------------
    # BUOC 3: PHAN TICH THONG KE MO TA VA TRUC QUAN HOA (EDA)
    # -------------------------------------------------------------
    # In thong bao buoc truc quan hoa
    print('\n[BUOC 3] Thuc hien truc quan hoa du lieu EDA...')
    # Danh sach cac cot duoc dua vao ma tran tuong quan
    corr_cols = [
        # Thu nhap hang nam
        'Annual_Income',
        # Tong chi tieu hang thang
        'Monthly_Spending',
        # Han muc tin dung
        'Credit_Limit',
        # Diem tin dung
        'Credit_Score',
        # Ty le thanh toan
        'Payment_Ratio',
        # Ty trong chi tieu tuy y
        'Discretionary_Ratio',
        # Ty trong chi tieu thiet yeu
        'Essential_Ratio'
    # Ket thuc khoi danh sach hoac tu dien
    ]
    # Duong dan tap tin anh heatmap tuong quan
    heatmap_file = os.path.join(FIGURES_DIR, 'eda_correlation_heatmap.png')
    # Ve va luu bieu do ma tran he so tuong quan
    plot_correlation_heatmap(df_engineered, corr_cols, heatmap_file)

    # Duong dan tap tin bieu do hop phan phoi chi tieu
    boxplot_file = os.path.join(FIGURES_DIR, 'category_spending_boxplot.png')
    # Ve va luu bieu do hop cho 7 danh muc chi tieu
    plot_category_spending_boxplot(df_engineered, SPENDING_CATEGORIES, boxplot_file)

    # -------------------------------------------------------------
    # BUOC 4: TIEN XU LY VA CHUAN HOA DU LIEU (PREPROCESSING)
    # -------------------------------------------------------------
    # In thong bao buoc tien xu ly chuan hoa
    print('\n[BUOC 4] Chuan hoa dac trung hanh vi bang StandardScaler...')
    # Danh sach cac cot dac trung duoc lua chon de dua vao phan cum
    cluster_features = RATIO_CATEGORIES + ['Credit_Utilization', 'Payment_Ratio', 'Discretionary_Ratio', 'Redemption_Rate']
    # Thuc hien chuan hoa du lieu bang Z-score thong qua StandardScaler
    X_scaled, scaler, scaled_df = preprocess_features(df_engineered, cluster_features)

    # -------------------------------------------------------------
    # BUOC 5: PHAN CUM K-MEANS VA CHIEU KHONG GIAN PCA 2D
    # -------------------------------------------------------------
    # In thong bao buoc phan cum va giam chieu
    print('\n[BUOC 5] Phan cum K-Means va chieu khong gian PCA 2D...')
    # Khoi tao bo phan cum K-Means voi k=4
    clustering_model = CustomerClustering(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    # Huan luyen va gan nhan cum cho 50.000 khach hang
    cluster_labels = clustering_model.fit_predict(X_scaled)
    # Gan cot nhan cum vao bang du lieu da qua xu ly dac trung
    df_engineered['Cluster'] = cluster_labels

    # Danh gia chat luong phan cum qua Silhouette Score va Davies-Bouldin Index
    eval_metrics = clustering_model.evaluate(X_scaled)

    # Chieu ma tran dac trung xuong 2 chieu PCA de truc quan hoa
    pca_coords = clustering_model.fit_transform_pca(X_scaled)
    # Duong dan luu tap tin anh bieu do phan cum tren khong gian PCA
    pca_img_path = os.path.join(FIGURES_DIR, 'pca_customer_clusters.png')
    # Ve va xuat bieu do phan cum 2D
    plot_pca_clusters(pca_coords, cluster_labels, pca_img_path)

    # Tong hop bang chi so dinh luong cua 4 cum de xay dung persona
    personas_summary_df = clustering_model.summarize_personas(df_engineered, cluster_labels)
    # Duong dan luu bao cao thong ke persona
    personas_csv_path = os.path.join(REPORTS_DIR, 'cluster_personas_summary.csv')
    # Luu bang thong ke persona ra CSV an toan
    save_dataframe_safely(personas_summary_df, personas_csv_path)

    # -------------------------------------------------------------
    # BUOC 6: TINH DIEM THUONG DONG VA GOI Y UU DAI THEO HANH VI
    # -------------------------------------------------------------
    # In thong bao buoc tinh diem thuong va goi y
    print('\n[BUOC 6] Tinh toan chi so diem thuong va tong hop bao cao cho tung khach hang...')
    # Khoi tao bo tinh diem va goi y dua tren he so va nguong rui ro
    scorer = CustomerBehaviorScorer(BASE_REWARD_MULTIPLIERS, EXPENSE_BONUS_FACTORS, RISK_THRESHOLDS)

    # Tinh toan chuoi diem thuong dong cho toan bo khach hang
    df_engineered['Dynamic_Reward_Points'] = scorer.calculate_dynamic_reward_points(df_engineered)

    # Huan luyen thuat toan k-NN tim lang gieng rieng biet tren tung cum
    scorer.fit_knn_by_cluster(df_engineered, cluster_features, cluster_col='Cluster')

    # Tong hop bao cao phan tich chi tiet cho tung khach hang tren toan bo 50.000 dong
    customer_report_df = scorer.generate_customer_report(df_engineered)
    # Duong dan luu tep CSV bao cao chi tiet tung khach hang
    customer_report_path = os.path.join(REPORTS_DIR, 'customer_level_analysis_report.csv')
    # Xuat toan bo bao cao tung khach hang ra tep CSV (luu truc tiep khong in chi tiet ra man hinh)
    save_dataframe_safely(customer_report_df, customer_report_path)

    # -------------------------------------------------------------
    # BUOC 7: KIEM CHUNG MO HINH MACHINE LEARNING
    # -------------------------------------------------------------
    # In thong bao buoc kiem chung mo hinh
    print('\n[BUOC 7] Kiem chung mo hinh Machine Learning (Linear Regression vs Random Forest vs LightGBM)...')
    # Danh sach cac dac trung dau vao duoc dung de du bao diem thuong
    feature_predictors = ['Monthly_Spending', 'Annual_Income', 'Credit_Limit', 'Mobile_App_Login', 'Payment_Ratio', 'Discretionary_Spending', 'Essential_Spending']
    # Chuan hoa tap bien doc lap bang StandardScaler
    X_val = StandardScaler().fit_transform(df_engineered[feature_predictors].values)
    # Lay mang gia tri bien phu thuoc muc tieu
    y_val = df_engineered['Dynamic_Reward_Points'].values

    # Khoi tao dong co kiem chung mo hinh Machine Learning
    val_engine = RewardValidationEngine(random_state=RANDOM_STATE)
    # Chay doi chuan hieu nang tren 3 mo hinh
    benchmark_df = val_engine.run_benchmark(X_val, y_val)

    # Duong dan luu bao cao doi chuan dang CSV
    benchmark_csv_path = os.path.join(REPORTS_DIR, 'model_benchmark_metrics.csv')
    # Xuat bang ket qua doi chuan ra CSV an toan
    save_dataframe_safely(benchmark_df, benchmark_csv_path)

    # Duong dan luu bieu do so sanh hieu nang
    benchmark_img_path = os.path.join(FIGURES_DIR, 'model_benchmark_comparison.png')
    # Ve va luu bieu do cot so sanh R2
    plot_model_benchmark(benchmark_df, benchmark_img_path)

    # In thong bao hoan tat toan bo quy trinh
    print('\n' + '=' * 80)
    # In dong tong ket
    print('  QUY TRINH PHAN TICH DU LIEU VA MO HINH HOA DA HOAN TAT XUAT SAC!')
    # In thu muc luu tru bieu do
    print(f'  - Bieu do truc quan hoa da duoc luu tai: {FIGURES_DIR}')
    # In thu muc luu tru bao cao du lieu
    print(f'  - Toan bo bao cao CSV da duoc luu tai  : {REPORTS_DIR}')
    # In duong ke ket thuc
    print('=' * 80)

# Diem bat dau thuc thi chuong trinh
if __name__ == '__main__':
    # Goi ham main dieu phoi thuc thi
    main()
