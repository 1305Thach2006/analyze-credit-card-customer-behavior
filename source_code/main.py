# Import thư viện os để thao tác với thư mục và đường dẫn tệp
import os
# Import thư viện sys để cấu hình đường dẫn module và bảng mã ký tự hệ thống
import sys

# Cấu hình luồng xuất chuẩn tiêu chuẩn hỗ trợ tiếng Việt có dấu UTF-8 trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    # Thiết lập lại bộ mã hóa đầu ra sang UTF-8
    sys.stdout.reconfigure(encoding='utf-8')

# Import thư viện warnings để ẩn các cảnh báo phiên bản tương lai của scikit-learn
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Import thư viện pandas để thao tác trên DataFrame
import pandas as pd
# Import thư viện numpy để thực hiện các phép toán ma trận
import numpy as np
# Import bộ chuẩn hóa StandardScaler để chuẩn hóa đặc trưng
from sklearn.preprocessing import StandardScaler

# Thêm thư mục hiện tại vào đường dẫn tìm kiếm module của Python
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import các thiết lập đường dẫn từ config/settings.py
from config.settings import DATA_PATH, OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR
# Import các tham số thuật toán và trọng số từ config/settings.py
from config.settings import RANDOM_STATE, N_CLUSTERS, SPENDING_CATEGORIES, RATIO_CATEGORIES
# Import các hệ số thưởng và ngưỡng rủi ro từ config/settings.py
from config.settings import BASE_REWARD_MULTIPLIERS, EXPENSE_BONUS_FACTORS, RISK_THRESHOLDS

# Import hàm nạp dữ liệu từ data_pipeline/loader.py
from data_pipeline.loader import load_credit_card_data
# Import hàm kỹ thuật chế tác đặc trưng từ data_pipeline/feature_engineering.py
from data_pipeline.feature_engineering import engineer_credit_features
# Import hàm tiền xử lý và chuẩn hóa từ data_pipeline/preprocessor.py
from data_pipeline.preprocessor import preprocess_features

# Import các hàm vẽ biểu đồ trực quan hóa từ eda/visualizer.py
from eda.visualizer import (
    plot_correlation_heatmap,
    plot_category_spending_boxplot,
    plot_pca_clusters,
    plot_kmeans_k_optimization,
    plot_model_benchmark,
    plot_uplift_qini_curve
)

# Import bộ phân cụm khách hàng K-Means và PCA từ models/clustering.py
from models.clustering import CustomerClustering
# Import động cơ tính điểm thưởng và gợi ý từ models/recommender.py
from models.recommender import CustomerBehaviorScorer
# Import lớp đối chuẩn các mô hình Machine Learning từ models/validation_models.py
from models.validation_models import RewardValidationEngine
# Import mô-đun học máy nhân quả Uplift Modeling từ models/uplift_engine.py
from models.uplift_engine import UpliftPersonalizationEngine

# Hàm lưu DataFrame an toàn tránh lỗi bị khóa tệp khi người dùng mở trên Excel
def save_dataframe_safely(df: pd.DataFrame, file_path: str) -> str:
    # Khối lệnh thử lưu tệp CSV theo đường dẫn gốc
    try:
        # Ghi dữ liệu DataFrame ra tệp CSV không kèm cột chỉ mục
        df.to_csv(file_path, index=False)
        # In thông báo lưu tệp thành công
        print(f'[REPORT] Đã lưu thành công báo cáo tại: {file_path}')
        # Trả về đường dẫn tệp đã lưu
        return file_path
    # Bắt ngoại lệ PermissionError khi tệp đang bị khóa bởi chương trình khác
    except PermissionError:
        # Tách tên tệp và phần mở rộng
        base, ext = os.path.splitext(file_path)
        # Tạo đường dẫn tệp thay thế với hậu tố _updated
        alt_path = f'{base}_updated{ext}'
        # Ghi dữ liệu DataFrame sang tệp thay thế
        df.to_csv(alt_path, index=False)
        # In cảnh báo người dùng về việc tệp bị khóa và đã lưu sang đường dẫn mới
        print(f'[CẢNH BÁO] Tập tin {os.path.basename(file_path)} đang mở trong Excel. Đã lưu vào: {alt_path}')
        # Trả về đường dẫn tệp mới
        return alt_path

# Hàm điều phối toàn bộ quy trình phân tích dữ liệu và huấn luyện mô hình
def main():
    # In dòng phân cách tiêu đề chương trình
    print('=' * 80)
    # In tiêu đề đề tài nghiên cứu bằng tiếng Việt có dấu chuẩn
    print('  MÔ HÌNH PHÂN TÍCH HÀNH VI TIÊU DÙNG THẺ TÍN DỤNG VÀ ĐỀ XUẤT CHIẾN LƯỢC ƯU ĐÃI CÁ NHÂN')
    # In dòng phân cách tiêu đề
    print('=' * 80)

    # -------------------------------------------------------------
    # BƯỚC 1: NẠP DỮ LIỆU (DATA INGESTION)
    # -------------------------------------------------------------
    # Gọi hàm nạp dữ liệu từ file CSV gốc
    raw_df = load_credit_card_data(DATA_PATH)

    # -------------------------------------------------------------
    # BƯỚC 2: CHẾ TÁC ĐẶC TRƯNG (FEATURE ENGINEERING)
    # -------------------------------------------------------------
    # Tạo các tỷ trọng chi tiêu và gắn cờ cảnh báo rủi ro
    df_engineered = engineer_credit_features(raw_df)

    # -------------------------------------------------------------
    # BƯỚC 3: TRỰC QUAN HÓA KHÁM PHÁ DỮ LIỆU (EDA)
    # -------------------------------------------------------------
    # In thông báo bước trực quan hóa khám phá dữ liệu
    print('\n[BƯỚC 3] Thực hiện trực quan hóa dữ liệu khám phá (EDA)...')
    # Danh sách 7 cột đặc trưng chi tiêu và tài chính để vẽ ma trận tương quan gốc
    corr_features = ['Annual_Income', 'Monthly_Spending', 'Credit_Limit', 'Credit_Score', 'Payment_Ratio', 'Discretionary_Ratio', 'Essential_Ratio']
    # Đường dẫn lưu tệp ảnh biểu đồ nhiệt tương quan
    heatmap_path = os.path.join(FIGURES_DIR, 'eda_correlation_heatmap.png')
    # Vẽ và lưu biểu đồ heatmap tương quan
    plot_correlation_heatmap(df_engineered, corr_features, heatmap_path)

    # Đường dẫn lưu tệp ảnh biểu đồ hộp phân phối chi tiêu
    boxplot_path = os.path.join(FIGURES_DIR, 'category_spending_boxplot.png')
    # Vẽ và lưu biểu đồ boxplot 7 danh mục chi tiêu
    plot_category_spending_boxplot(df_engineered, SPENDING_CATEGORIES, boxplot_path)

    # -------------------------------------------------------------
    # BƯỚC 4: TIỀN XỬ LÝ VÀ CHUẨN HÓA DỮ LIỆU (PREPROCESSING)
    # -------------------------------------------------------------
    # In thông báo bước chuẩn hóa dữ liệu
    print('\n[BƯỚC 4] Chuẩn hóa đặc trưng hành vi bằng StandardScaler...')
    # Danh sách các cột đặc trưng được lựa chọn để đưa vào phân cụm
    cluster_features = RATIO_CATEGORIES + ['Credit_Utilization', 'Payment_Ratio', 'Discretionary_Ratio', 'Redemption_Rate']
    # Thực hiện chuẩn hóa dữ liệu bằng Z-score thông qua StandardScaler
    X_scaled, scaler, scaled_df = preprocess_features(df_engineered, cluster_features)

    # -------------------------------------------------------------
    # BƯỚC 5: TỐI ƯU HÓA SỐ CỤM K (2 → 6), PHÂN CỤM K-MEANS VÀ PCA 2D
    # -------------------------------------------------------------
    # In thông báo bước thực nghiệm quét k và phân cụm
    print('\n[BƯỚC 5] Thực nghiệm quét số cụm k từ 2 đến 6, phân cụm K-Means và chiếu không gian PCA 2D...')
    # Khởi tạo bộ phân cụm K-Means với k=4 mặc định
    clustering_model = CustomerClustering(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE)
    # Thực hiện thực nghiệm quét k từ 2 đến 6 để tính Elbow, Silhouette và Davies-Bouldin
    k_eval_df = clustering_model.evaluate_k_range(X_scaled, k_range=range(2, 7))
    # Đường dẫn lưu báo cáo CSV chỉ số đánh giá theo k
    k_eval_csv_path = os.path.join(REPORTS_DIR, 'k_selection_metrics.csv')
    # Lưu bảng thống kê chỉ số theo k ra tệp CSV an toàn
    save_dataframe_safely(k_eval_df, k_eval_csv_path)
    # Đường dẫn lưu biểu đồ tối ưu hóa số cụm k
    k_opt_img_path = os.path.join(FIGURES_DIR, 'kmeans_k_optimization_curve.png')
    # Vẽ và lưu biểu đồ đánh giá tối ưu số cụm k với chú thích tiếng Việt có dấu
    plot_kmeans_k_optimization(k_eval_df, k_opt_img_path)

    # Huấn luyện và gán nhãn cụm cho 50.000 khách hàng với số cụm k=4 tối ưu
    cluster_labels = clustering_model.fit_predict(X_scaled)
    # Gán cột nhãn cụm vào bảng dữ liệu đã qua xử lý đặc trưng
    df_engineered['Cluster'] = cluster_labels

    # Đánh giá chất lượng phân cụm qua Silhouette Score và Davies-Bouldin Index tại k=4
    eval_metrics = clustering_model.evaluate(X_scaled)

    # Chiếu ma trận đặc trưng xuống 2 chiều PCA để trực quan hóa
    pca_coords = clustering_model.fit_transform_pca(X_scaled)
    # Đường dẫn lưu tập tin ảnh biểu đồ phân cụm trên không gian PCA
    pca_img_path = os.path.join(FIGURES_DIR, 'pca_customer_clusters.png')
    # Vẽ và xuất biểu đồ phân cụm 2D với chú thích tiếng Việt có dấu
    plot_pca_clusters(pca_coords, cluster_labels, pca_img_path)

    # Tổng hợp bảng chỉ số định lượng của 4 cụm để xây dựng chân dung persona
    personas_summary_df = clustering_model.summarize_personas(df_engineered, cluster_labels)
    # Đường dẫn lưu báo cáo thống kê persona
    personas_csv_path = os.path.join(REPORTS_DIR, 'cluster_personas_summary.csv')
    # Lưu bảng thống kê persona ra CSV an toàn
    save_dataframe_safely(personas_summary_df, personas_csv_path)

    # -------------------------------------------------------------
    # BƯỚC 6: HỌC MÁY NHÂN QUẢ UPLIFT MODELING & TỐI ƯU HÓA ƯU ĐÃI
    # -------------------------------------------------------------
    # In thông báo bước huấn luyện Uplift Modeling
    print('\n[BƯỚC 6] Huấn luyện học máy nhân quả Uplift Modeling (T-Learner vs S-Learner) và phân khúc khách hàng...')
    # Khởi tạo bộ tính điểm thưởng động và kiểm tra rủi ro
    scorer = CustomerBehaviorScorer(BASE_REWARD_MULTIPLIERS, EXPENSE_BONUS_FACTORS, RISK_THRESHOLDS)
    # Tính toán chuỗi điểm thưởng động cho toàn bộ khách hàng
    df_engineered['Dynamic_Reward_Points'] = scorer.calculate_dynamic_reward_points(df_engineered)

    # Khởi tạo động cơ Uplift Modeling
    uplift_engine = UpliftPersonalizationEngine(random_state=RANDOM_STATE)
    # Thiết lập biến can thiệp Treatment và biến mục tiêu Outcome
    treatment, outcome = uplift_engine.prepare_uplift_data(df_engineered)

    # Tập hợp các đặc trưng đầu vào phục vụ mô hình Uplift
    uplift_features = cluster_features + ['Monthly_Spending', 'Annual_Income', 'Credit_Limit', 'Credit_Score']
    # Chuẩn hóa tập đặc trưng đầu vào cho Uplift
    X_uplift = StandardScaler().fit_transform(df_engineered[uplift_features].values)

    # Huấn luyện và đối chuẩn T-Learner vs S-Learner trên 80% Train / 20% Test
    uplift_metrics = uplift_engine.fit_and_evaluate(X_uplift, outcome, treatment)

    # Đường dẫn lưu biểu đồ đường cong Qini
    qini_img_path = os.path.join(FIGURES_DIR, 'uplift_qini_curve.png')
    # Vẽ và lưu biểu đồ đường cong Qini với chú thích tiếng Việt có dấu
    plot_uplift_qini_curve(
        uplift_metrics['y_test'],
        uplift_metrics['uplift_preds_t'],
        uplift_metrics['uplift_preds_s'],
        uplift_metrics['treatment_test'],
        qini_img_path
    )

    # Dự báo điểm Uplift Score cho toàn bộ 50.000 khách hàng bằng mô hình T-Learner
    full_uplift_scores = uplift_engine.predict_full_dataset(X_uplift)

    # Phân loại 50.000 khách hàng vào 4 nhóm nhân quả kinh điển
    uplift_segments = uplift_engine.assign_uplift_quadrants(df_engineered, full_uplift_scores)

    # Tổng hợp báo cáo định lượng 4 nhóm nhân quả
    uplift_summary_df = uplift_engine.generate_uplift_summary_report(df_engineered, full_uplift_scores, uplift_segments)
    # Đường dẫn lưu báo cáo phân khúc nhân quả
    uplift_summary_csv_path = os.path.join(REPORTS_DIR, 'uplift_segments_summary.csv')
    # Lưu báo cáo phân khúc nhân quả ra tệp CSV an toàn
    save_dataframe_safely(uplift_summary_df, uplift_summary_csv_path)

    # Tổng hợp báo cáo phân tích chi tiết cho từng khách hàng tích hợp Uplift
    customer_report_df = scorer.generate_customer_report(df_engineered, full_uplift_scores, uplift_segments)
    # Đường dẫn lưu tệp CSV báo cáo chi tiết từng khách hàng
    customer_report_path = os.path.join(REPORTS_DIR, 'customer_level_analysis_report.csv')
    # Xuất toàn bộ báo cáo từng khách hàng ra tệp CSV
    save_dataframe_safely(customer_report_df, customer_report_path)

    # Trích xuất bảng minh họa mẫu 4 khách hàng đại diện cho 4 nhóm nhân quả Uplift
    sample_recommendations_df = scorer.generate_sample_recommendations(customer_report_df)
    # Đường dẫn lưu tệp CSV bảng mẫu đề xuất ưu đãi
    sample_recommendations_path = os.path.join(REPORTS_DIR, 'sample_recommendations.csv')
    # Lưu bảng mẫu ra CSV an toàn phục vụ thuyết trình slide và báo cáo
    save_dataframe_safely(sample_recommendations_df, sample_recommendations_path)

    # -------------------------------------------------------------
    # BƯỚC 7: KIỂM CHỨNG MÔ HÌNH MACHINE LEARNING VÀ MÔ HÌNH LAI STACKING
    # -------------------------------------------------------------
    # In thông báo bước kiểm chứng mô hình
    print('\n[BƯỚC 7] Kiểm chứng mô hình Machine Learning (Linear Regression vs Random Forest vs LightGBM vs Stacking Hybrid)...')
    # Danh sách các đặc trưng đầu vào được dùng để dự báo điểm thưởng
    feature_predictors = ['Monthly_Spending', 'Annual_Income', 'Credit_Limit', 'Mobile_App_Login', 'Payment_Ratio', 'Discretionary_Spending', 'Essential_Spending']
    # Chuẩn hóa tập biến độc lập bằng StandardScaler(Z)
    X_val = StandardScaler().fit_transform(df_engineered[feature_predictors].values)
    # Lấy mảng giá trị biến phụ thuộc mục tiêu
    y_val = df_engineered['Dynamic_Reward_Points'].values

    # Khởi tạo động cơ kiểm chứng mô hình Machine Learning và mô hình Lai
    val_engine = RewardValidationEngine(random_state=RANDOM_STATE)
    # Chạy đối chuẩn hiệu năng trên 4 mô hình
    benchmark_df = val_engine.run_benchmark(X_val, y_val)

    # Đường dẫn lưu báo cáo đối chuẩn dạng CSV
    benchmark_csv_path = os.path.join(REPORTS_DIR, 'model_benchmark_metrics.csv')
    # Xuất bảng kết quả đối chuẩn ra CSV an toàn
    save_dataframe_safely(benchmark_df, benchmark_csv_path)

    # Đường dẫn lưu biểu đồ so sánh hiệu năng 4 mô hình
    benchmark_img_path = os.path.join(FIGURES_DIR, 'model_benchmark_comparison.png')
    # Vẽ và lưu biểu đồ cột so sánh R2 với chú thích tiếng Việt có dấu
    plot_model_benchmark(benchmark_df, benchmark_img_path)

    # In thông báo hoàn tất toàn bộ quy trình
    print('\n' + '=' * 80)
    # In dòng tổng kết
    print('  QUY TRÌNH PHÂN TÍCH DỮ LIỆU VÀ MÔ HÌNH HÓA ĐÃ HOÀN TẤT XUẤT SẮC!')
    # In thư mục lưu trữ biểu đồ
    print(f'  - Biểu đồ trực quan hóa đã được lưu tại: {FIGURES_DIR}')
    # In thư mục lưu trữ báo cáo dữ liệu
    print(f'  - Toàn bộ báo cáo CSV đã được lưu tại  : {REPORTS_DIR}')
    # In đường kẻ kết thúc
    print('=' * 80)

# Điểm bắt đầu thực thi chương trình
if __name__ == '__main__':
    # Gọi hàm main điều phối thực thi
    main()
