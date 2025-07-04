import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, roc_curve, auc)

st.set_page_config(page_title="산불 예측 AI 대시보드", layout='wide')

st.title("🔥 환경을 지키는 AI - 산불 예측 대시보드")

# 1. 데이터 업로드 & 미리보기
st.header("1️. 데이터 미리보기")
uploaded_file = st.file_uploader("fire.csv.xlsx 파일을 업로드하세요", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.write("데이터 샘플", df.head())

    # 변수 선택
    numeric_cols = ['top_tprt', 'avg_hmd', 'ave_wdsp', 'de_rnfl_qy']
    target_col = 'frfire_ocrn_nt'

    # 결측치 처리
    df = df.dropna(subset=[target_col])
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())

    # 타겟 이진화
    df[target_col] = (df[target_col] > 0).astype(int)

    st.subheader("타겟(산불 발생) 분포")
    st.bar_chart(df[target_col].value_counts())

    # 2. 변수별 히스토그램
    st.header("2️. 주요 변수별 분포")
    fig, axs = plt.subplots(2, 2, figsize=(12, 6))
    for i, col in enumerate(numeric_cols):
        ax = axs[i//2, i%2]
        sns.histplot(df[col], bins=20, kde=True, ax=ax)
        ax.set_title(col)
    st.pyplot(fig)

    # 3. 상관관계 히트맵
    st.header("3️. 주요 변수 간 상관관계")
    corr_matrix = df[numeric_cols].corr()
    fig_corr, ax_corr = plt.subplots()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

    # 4. 모델 학습 및 평가
    st.header("4️. 산불 예측 AI 모델 비교")

    features = numeric_cols
    X = df[features]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 모델 선언/학습
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    xgb_model = xgb.XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    # 혼동행렬 함수
    def plot_confusion(y_true, y_pred, title):
        labels = np.unique([*y_true, *y_pred])
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig_cm, ax_cm = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap='Blues', ax=ax_cm)
        ax_cm.set_title(title)
        st.pyplot(fig_cm)

    # 모델별 혼동행렬 및 성능지표
    st.subheader("모델별 혼동행렬 및 평가")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**RandomForest**")
        plot_confusion(y_test, y_pred_rf, "RandomForest 혼동행렬")
        st.text(classification_report(y_test, y_pred_rf))
    with col2:
        st.markdown("**LogisticRegression**")
        plot_confusion(y_test, y_pred_lr, "LogisticRegression 혼동행렬")
        st.text(classification_report(y_test, y_pred_lr))
    with col3:
        st.markdown("**XGBoost**")
        plot_confusion(y_test, y_pred_xgb, "XGBoost 혼동행렬")
        st.text(classification_report(y_test, y_pred_xgb))

    # ROC Curve 비교
    st.subheader("모델별 ROC Curve & AUC")
    fig_roc, ax_roc = plt.subplots(figsize=(8,6))

    def plot_roc(model, X, y, label):
        y_proba = model.predict_proba(X)[:,1]
        fpr, tpr, _ = roc_curve(y, y_proba)
        auc_score = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{label} (AUC={auc_score:.2f})")
        return auc_score

    auc_rf = plot_roc(rf, X_test, y_test, "RandomForest")
    auc_lr = plot_roc(lr, X_test, y_test, "LogisticRegression")
    auc_xgb = plot_roc(xgb_model, X_test, y_test, "XGBoost")

    ax_roc.plot([0,1],[0,1],'k--',label='Random (AUC=0.5)')
    ax_roc.set_xlabel('False Positive Rate')
    ax_roc.set_ylabel('True Positive Rate')
    ax_roc.set_title('모델별 ROC Curve 및 AUC')
    ax_roc.legend()
    ax_roc.grid(True)
    st.pyplot(fig_roc)

    # 변수 중요도
    st.header("5️. 랜덤포레스트 변수 중요도")
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig_imp, ax_imp = plt.subplots()
    ax_imp.bar(range(X.shape[1]), importances[indices], align='center')
    ax_imp.set_xticks(range(X.shape[1]))
    ax_imp.set_xticklabels(np.array(X.columns)[indices], rotation=45)
    ax_imp.set_title("Feature Importances (RandomForest)")
    st.pyplot(fig_imp)

    st.markdown("---")
    st.success("AI 기반 산불 예측 결과와 데이터 EDA를 확인했습니다! \n(실제 정책적 활용 시, FN(실제 발생인데 미발생으로 예측한 경우) 감소 전략에 주목하세요.)")
else:
    st.info("먼저 데이터를 업로드하세요.")
