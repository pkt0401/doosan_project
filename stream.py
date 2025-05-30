import streamlit as st
import pandas as pd
import numpy as np
import faiss
import openai
import re
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split

# ------------- 시스템 다국어 텍스트 -----------------
system_texts = {
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "시스템 개요",
        "tab_phase1": "위험성 평가 (Phase 1)",
        "tab_phase2": "개선대책 생성 (Phase 2)",
        "overview_header": "LLM 기반 위험성평가 시스템",
        "overview_text": "두산에너빌리티 AI Risk Assessment는 국내 및 해외 건설현장 '수시위험성평가' 및 '노동부 중대재해 사례'를 학습하여 개발된 자동 위험성평가 프로그램 입니다. 생성된 위험성평가는 반드시 수시 위험성평가 심의회를 통해 검증 후 사용하시기 바랍니다.",
        "features_title": "시스템 특징 및 구성요소",
        "phase1_features": """
        #### Phase 1: 위험성 평가 자동화
        - 공정별 작업활동에 따른 위험성평가 데이터 학습
        - 작업활동 입력 시 유해위험요인 자동 예측
        - 유사 위험요인 사례 검색 및 표시
        - 대규모 언어 모델(LLM) 기반 위험도(빈도, 강도, T) 측정
        - Excel 기반 공정별 위험성평가 데이터 자동 분석
        - 위험등급(A-E) 자동 산정
        """,
        "phase2_features": """
        #### Phase 2: 개선대책 자동 생성
        - 위험요소별 맞춤형 개선대책 자동 생성
        - 다국어(한/영/중) 개선대책 생성 지원
        - 개선 전후 위험도(T) 자동 비교 분석
        - 위험 감소율(RRR) 정량적 산출
        - 공종/공정별 최적 개선대책 데이터베이스 구축
        """,
        "phase1_header": "위험성 평가 자동화 (Phase 1)",
        "api_key_label": "OpenAI API 키를 입력하세요:",
        "dataset_label": "데이터셋 선택",
        "load_data_label": "데이터 로드 및 인덱스 구성",
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "계속하려면 OpenAI API 키를 입력하세요.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "데모 목적으로 {max_texts}개의 텍스트만 임베딩합니다. 실제 환경에서는 전체 데이터를 처리해야 합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! (총 {max_texts}개 항목 처리)",
        "hazard_prediction_header": "유해위험요인 예측",
        "load_first_warning": "먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.",
        "activity_label": "작업활동:",
        "predict_hazard_btn": "유해위험요인 예측하기",
        "activity_warning": "작업활동을 입력하세요.",
        "predicting_hazard": "유해위험요인을 예측하는 중...",
        "similar_cases_header": "유사한 사례",
        "similar_case_text": """
        <div class="similar-case">
            <strong>사례 {i}</strong><br>
            <strong>작업활동:</strong> {activity}<br>
            <strong>유해위험요인:</strong> {hazard}<br>
            <strong>위험도:</strong> 빈도 {freq}, 강도 {intensity}, T값 {t_value} (등급 {grade})
        </div>
        """,
        "prediction_result_header": "예측 결과",
        "activity_result": "작업활동: {activity}",
        "hazard_result": "예측된 유해위험요인: {hazard}",
        "result_table_columns": ["항목", "값"],
        "result_table_rows": ["빈도", "강도", "T 값", "위험등급"],
        "parsing_error": "위험성 평가 결과를 파싱할 수 없습니다.",
        "gpt_response": "GPT 원문 응답: {response}",
        "phase2_header": "개선대책 자동 생성 (Phase 2)",
        "language_select_label": "개선대책 언어 선택:",
        "input_method_label": "입력 방식 선택:",
        "input_methods": ["Phase 1 평가 결과 사용", "직접 입력"],
        "phase1_results_header": "Phase 1 평가 결과",
        "risk_level_text": "위험도: 빈도 {freq}, 강도 {intensity}, T값 {t_value} (등급 {grade})",
        "phase1_first_warning": "먼저 Phase 1에서 위험성 평가를 수행하세요.",
        "hazard_label": "유해위험요인:",
        "frequency_label": "빈도 (1-5):",
        "intensity_label": "강도 (1-5):",
        "t_value_text": "T값: {t_value} (등급: {grade})",
        "generate_improvement_btn": "개선대책 생성",
        "generating_improvement": "개선대책을 생성하는 중...",
        "no_data_warning": "Phase 1에서 데이터 로드 및 인덱스 구성을 완료하지 않았습니다. 기본 예시를 사용합니다.",
        "improvement_result_header": "개선대책 생성 결과",
        "improvement_plan_header": "개선대책",
        "risk_improvement_header": "위험도 개선 결과",
        "comparison_columns": ["항목", "개선 전", "개선 후"],
        "risk_reduction_label": "위험 감소율 (RRR)",
        "t_value_change_header": "위험도(T값) 변화",
        "before_improvement": "개선 전 T값:",
        "after_improvement": "개선 후 T값:",
        "parsing_error_improvement": "위험성 평가 결과를 파싱할 수 없습니다.",
        "excel_export": "📥 결과 Excel 다운로드",
        "risk_classification": "위험도 분류",
        "supported_languages": "지원 언어",
        "languages_count": "3개",
        "languages_detail": "한/영/중",
        "assessment_phases": "평가 단계",
        "phases_count": "2단계",
        "phases_detail": "평가+개선",
        "risk_grades": "위험등급",
        "grades_count": "5등급",
        "grades_detail": "A~E"
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_phase1": "Risk Assessment (Phase 1)",
        "tab_phase2": "Improvement Measures (Phase 2)",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": "Doosan Enerbility AI Risk Assessment is an automated program trained on both on-demand risk-assessment reports from domestic and overseas construction sites and major-accident cases compiled by Korea's Ministry of Employment and Labor. Please ensure that every generated assessment is reviewed and approved by the On-Demand Risk Assessment Committee before it is used.",
        "features_title": "System Features and Components",
        "phase1_features": """
        #### Phase 1: Risk Assessment Automation
        - Learning risk assessment data according to work activities by process
        - Automatic hazard prediction when work activities are entered
        - Similar case search and display
        - Risk level (frequency, intensity, T) measurement based on large language models (LLM)
        - Automatic analysis of Excel-based process-specific risk assessment data
        - Automatic risk grade (A-E) calculation
        """,
        "phase2_features": """
        #### Phase 2: Automatic Generation of Improvement Measures
        - Automatic generation of customized improvement measures for risk factors
        - Multilingual (Korean/English/Chinese) improvement measure generation support
        - Automatic comparative analysis of risk level (T) before and after improvement
        - Quantitative calculation of Risk Reduction Rate (RRR)
        - Building a database of optimal improvement measures by work type/process
        """,
        "phase1_header": "Risk Assessment Automation (Phase 1)",
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset",
        "load_data_label": "Load Data and Configure Index",
        "load_data_btn": "Load Data and Configure Index",
        "api_key_warning": "Please enter an OpenAI API key to continue.",
        "data_loading": "Loading data and configuring index...",
        "demo_limit_info": "For demo purposes, only embedding {max_texts} texts. In a real environment, all data should be processed.",
        "data_load_success": "Data load and index configuration complete! (Total {max_texts} items processed)",
        "hazard_prediction_header": "Hazard Prediction",
        "load_first_warning": "Please click the [Load Data and Configure Index] button first.",
        "activity_label": "Work Activity:",
        "predict_hazard_btn": "Predict Hazards",
        "activity_warning": "Please enter a work activity.",
        "predicting_hazard": "Predicting hazards...",
        "similar_cases_header": "Similar Cases",
        "similar_case_text": """
        <div class="similar-case">
            <strong>Case {i}</strong><br>
            <strong>Work Activity:</strong> {activity}<br>
            <strong>Hazard:</strong> {hazard}<br>
            <strong>Risk Level:</strong> Frequency {freq}, Intensity {intensity}, T-value {t_value} (Grade {grade})
        </div>
        """,
        "prediction_result_header": "Prediction Results",
        "activity_result": "Work Activity: {activity}",
        "hazard_result": "Predicted Hazard: {hazard}", 
        "result_table_columns": ["Item", "Value"],
        "result_table_rows": ["Frequency", "Intensity", "T Value", "Risk Grade"],
        "parsing_error": "Unable to parse risk assessment results.",
        "gpt_response": "Original GPT Response: {response}",
        "phase2_header": "Automatic Generation of Improvement Measures (Phase 2)",
        "language_select_label": "Select Language for Improvement Measures:",
        "input_method_label": "Select Input Method:",
        "input_methods": ["Use Phase 1 Assessment Results", "Direct Input"],
        "phase1_results_header": "Phase 1 Assessment Results",
        "risk_level_text": "Risk Level: Frequency {freq}, Intensity {intensity}, T-value {t_value} (Grade {grade})",
        "phase1_first_warning": "Please perform a risk assessment in Phase 1 first.",
        "hazard_label": "Hazard:",
        "frequency_label": "Frequency (1-5):",
        "intensity_label": "Intensity (1-5):",
        "t_value_text": "T-value: {t_value} (Grade: {grade})",
        "generate_improvement_btn": "Generate Improvement Measures",
        "generating_improvement": "Generating improvement measures...",
        "no_data_warning": "Data loading and index configuration was not completed in Phase 1. Using basic examples.",
        "improvement_result_header": "Improvement Measure Generation Results",
        "improvement_plan_header": "Improvement Measures",
        "risk_improvement_header": "Risk Level Improvement Results",
        "comparison_columns": ["Item", "Before Improvement", "After Improvement"],
        "risk_reduction_label": "Risk Reduction Rate (RRR)",
        "t_value_change_header": "Risk Level (T-value) Change",
        "before_improvement": "T-value Before Improvement:",
        "after_improvement": "T-value After Improvement:",
        "parsing_error_improvement": "Unable to parse improvement measure generation results.",
        "excel_export": "📥 Download Excel Results",
        "risk_classification": "Risk Classification",
        "supported_languages": "Supported Languages",
        "languages_count": "3 Languages",
        "languages_detail": "KOR/ENG/CHN",
        "assessment_phases": "Assessment Phases",
        "phases_count": "2 Phases",
        "phases_detail": "Assessment+Improvement",
        "risk_grades": "Risk Grades",
        "grades_count": "5 Grades", 
        "grades_detail": "A~E"
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "系统概述",
        "tab_phase1": "风险评估 (第1阶段)",
        "tab_phase2": "改进措施 (第2阶段)",
        "overview_header": "基于LLM的风险评估系统",
        "overview_text": "Doosan Enerbility AI 风险评估系统是一款自动化风险评估程序，基于国内外施工现场的'临时风险评估'数据以及韩国雇佣劳动部的重大事故案例进行训练开发而成。生成的风险评估结果必须经过临时风险评估审议委员会的审核后方可使用",
        "features_title": "系统特点和组件",
        "phase1_features": """
        #### 第1阶段：风险评估自动化
        - 按工序学习与工作活动相关的风险评估数据
        - 输入工作活动时自动预测危害因素
        - 相似案例搜索和显示
        - 基于大型语言模型(LLM)的风险等级（频率、强度、T值）测量
        - 自动分析基于Excel的特定工序风险评估数据
        - 自动计算风险等级(A-E)
        """,
        "phase2_features": """
        #### 第2阶段：自动生成改进措施
        - 为风险因素自动生成定制的改进措施
        - 多语言（韩语/英语/中文）改进措施生成支持
        - 改进前后风险等级（T值）的自动比较分析
        - 风险降低率(RRR)的定量计算
        - 建立按工作类型/工序的最佳改进措施数据库
        """,
        "phase1_header": "风险评估自动化 (第1阶段)",
        "api_key_label": "输入OpenAI API密钥：",
        "dataset_label": "选择数据集",
        "load_data_label": "加载数据和配置索引",
        "load_data_btn": "加载数据和配置索引",
        "api_key_warning": "请输入OpenAI API密钥以继续。",
        "data_loading": "正在加载数据和配置索引...",
        "demo_limit_info": "出于演示目的，仅嵌入{max_texts}个文本。在实际环境中，应处理所有数据。",
        "data_load_success": "数据加载和索引配置完成！（共处理{max_texts}个项目）",
        "hazard_prediction_header": "危害预测",
        "load_first_warning": "请先点击[加载数据和配置索引]按钮。",
        "activity_label": "工作活动：",
        "predict_hazard_btn": "预测危害",
        "activity_warning": "请输入工作活动。",
        "predicting_hazard": "正在预测危害...",
        "similar_cases_header": "相似案例",
        "similar_case_text": """
        <div class="similar-case">
            <strong>案例 {i}</strong><br>
            <strong>工作活动：</strong> {activity}<br>
            <strong>危害：</strong> {hazard}<br>
            <strong>风险等级：</strong> 频率 {freq}, 强度 {intensity}, T值 {t_value} (等级 {grade})
        </div>
        """,
        "prediction_result_header": "预测结果",
        "activity_result": "工作活动: {activity}",
        "hazard_result": "预测的危害: {hazard}",
        "result_table_columns": ["项目", "值"],
        "result_table_rows": ["频率", "强度", "T值", "风险等级"],
        "parsing_error": "无法解析风险评估结果。",
        "gpt_response": "原始GPT响应: {response}",
        "phase2_header": "自动生成改进措施 (第2阶段)",
        "language_select_label": "选择改进措施的语言：",
        "input_method_label": "选择输入方法：",
        "input_methods": ["使用第1阶段评估结果", "直接输入"],
        "phase1_results_header": "第1阶段评估结果",
        "risk_level_text": "风险等级: 频率 {freq}, 强度 {intensity}, T值 {t_value} (等级 {grade})",
        "phase1_first_warning": "请先在第1阶段进行风险评估。",
        "hazard_label": "危害：",
        "frequency_label": "频率 (1-5)：",
        "intensity_label": "强度 (1-5)：",
        "t_value_text": "T值: {t_value} (等级: {grade})",
        "generate_improvement_btn": "生成改进措施",
        "generating_improvement": "正在生成改进措施...",
        "no_data_warning": "在第1阶段未完成数据加载和索引配置。使用基本示例。",
        "improvement_result_header": "改进措施生成结果",
        "improvement_plan_header": "改进措施",
        "risk_improvement_header": "风险等级改进结果",
        "comparison_columns": ["项目", "改进前", "改进后"],
        "risk_reduction_label": "风险降低率 (RRR)",
        "t_value_change_header": "风险等级 (T值) 变化",
        "before_improvement": "改进前T值：",
        "after_improvement": "改进后T值：",
        "parsing_error_improvement": "无法解析改进措施生成结果。",
        "excel_export": "📥 下载Excel结果",
        "risk_classification": "风险分类",
        "supported_languages": "支持语言",
        "languages_count": "3种语言",
        "languages_detail": "韩/英/中",
        "assessment_phases": "评估阶段", 
        "phases_count": "2个阶段",
        "phases_detail": "评估+改进",
        "risk_grades": "风险等级",
        "grades_count": "5个等级",
        "grades_detail": "A~E"
    }
}

# ----------------- 페이지 / 스타일 -----------------
st.set_page_config(page_title="Artificial Intelligence Risk Assessment", page_icon="🛠️", layout="wide")

st.markdown("""
<style>
/* 기존 스타일에 추가 개선 */
.main-header{font-size:2.5rem;color:#1E88E5;text-align:center;margin-bottom:1rem}
.sub-header{font-size:1.8rem;color:#0D47A1;margin-top:2rem;margin-bottom:1rem}
.metric-container{background-color:#f0f2f6;border-radius:10px;padding:20px;box-shadow:2px 2px 5px rgba(0,0,0,0.1)}
.result-box{background-color:#f8f9fa;border-radius:10px;padding:15px;margin-top:10px;margin-bottom:10px;border-left:5px solid #4CAF50}
.phase-badge{background-color:#4CAF50;color:white;padding:5px 10px;border-radius:15px;font-size:0.8rem;margin-right:10px}
.similar-case{background-color:#f1f8e9;border-radius:8px;padding:12px;margin-bottom:8px;border-left:4px solid #689f38}
.language-selector{position:absolute;top:10px;right:10px;z-index:1000}
.calculation-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:15px;margin:20px 0}
.risk-grade-a{background-color:#ff1744;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
.risk-grade-b{background-color:#ff9800;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
.risk-grade-c{background-color:#ffc107;color:black;padding:5px 10px;border-radius:5px;font-weight:bold}
.risk-grade-d{background-color:#4caf50;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
.risk-grade-e{background-color:#2196f3;color:white;padding:5px 10px;border-radius:5px;font-weight:bold}
</style>
""", unsafe_allow_html=True)

# ----------------- 세션 상태 -----------------
ss = st.session_state
for key, default in {
    "language":"Korean","index":None,"embeddings":None,
    "retriever_pool_df":None,"last_assessment":None
}.items():
    if key not in ss: ss[key]=default

# ----------------- 언어 선택 -----------------
col0, colLang = st.columns([6,1])
with colLang:
    lang = st.selectbox("", list(system_texts.keys()), index=list(system_texts.keys()).index(ss.language))
    ss.language = lang
texts = system_texts[ss.language]

# ----------------- 헤더 -----------------
st.markdown(f'<div class="main-header">{texts["title"]}</div>', unsafe_allow_html=True)

# ----------------- 탭 구성 -----------------
tabs = st.tabs([texts["tab_overview"], "Risk Assessment ✨"])

# -----------------------------------------------------------------------------
# ---------------------------  공용 유틸리티 -----------------------------------
# -----------------------------------------------------------------------------

def determine_grade(value: int):
    """위험도 등급 분류 개선"""
    if 16<=value<=25: return 'A'
    if 10<=value<=15: return 'B'
    if 5<=value<=9: return 'C'
    if 3<=value<=4: return 'D'
    if 1<=value<=2: return 'E'
    return 'Unknown' if ss.language!='Korean' else '알 수 없음'

def get_grade_color(grade):
    """위험등급별 색상 반환"""
    colors = {
        'A': '#ff1744',  # 빨강
        'B': '#ff9800',  # 주황
        'C': '#ffc107',  # 노랑
        'D': '#4caf50',  # 초록
        'E': '#2196f3',  # 파랑
    }
    return colors.get(grade, '#gray')

def compute_rrr(original_t, improved_t):
    """위험 감소율 계산"""
    if original_t == 0:
        return 0
    return ((original_t - improved_t) / original_t) * 100

def _extract_improvement_info(row):
    """
    유사 사례 한 건에서 - 개선대책 / 개선 후 빈도·강도·T 값을 최대한 찾아 반환
    """
    # ① 개선대책
    plan_cols = [c for c in row.index if re.search(r'개선대책|Improvement|改进', c, re.I)]
    plan = row[plan_cols[0]] if plan_cols else ""

    # ② 개선 후 빈도·강도·T
    cand_sets = [
        ('개선 후 빈도', '개선 후 강도', '개선 후 T'),
        ('개선빈도', '개선강도', '개선T'),
        ('improved_frequency', 'improved_intensity', 'improved_T'),
        ('改进后频率', '改进后强度', '改进后T值'),
    ]
    imp_f, imp_i, imp_t = None, None, None
    for f,i,t in cand_sets:
        if f in row and i in row and t in row:
            imp_f, imp_i, imp_t = int(row[f]), int(row[i]), int(row[t])
            break

    # 값이 없으면 원래 값에서 감소된 값으로 추정
    if imp_f is None:
        orig_f, orig_i = int(row['빈도']), int(row['강도'])
        # 개선 후 추정: 원래 값의 50-70% 범위
        imp_f = max(1, orig_f - 1)
        imp_i = max(1, orig_i - 1)
        imp_t = imp_f * imp_i

    return plan, imp_f, imp_i, imp_t

@st.cache_data(show_spinner=False)
def load_data(selected_dataset_name: str):
    """향상된 데이터 로딩 함수"""
    try:
        # Excel 파일 읽기 시 여러 시트 처리 가능
        if os.path.exists(f"{selected_dataset_name}.xlsx"):
            df = pd.read_excel(f"{selected_dataset_name}.xlsx")
        else:
            # 파일이 없으면 샘플 데이터 생성
            return create_sample_data()

        # 1) 불필요 열 제거
        if "삭제 Del" in df.columns:
            df.drop(["삭제 Del"], axis=1, inplace=True)

        # 2) 빈 행 제거 (모든 값이 NaN인 행)
        df = df.dropna(how='all')

        # 3) 핵심 열 이름 통일 및 정리
        column_mapping = {
            "작업활동 및 내용\nWork & Contents": "작업활동 및 내용",
            "유해위험요인 및 환경측면 영향\nHazard & Risk": "유해위험요인 및 환경측면 영향",
            "피해형태 및 환경영향\nDamage & Effect": "피해형태 및 환경영향",
            "개선대책 및 세부관리방안\nCorrective Action": "개선대책"
        }
        
        df.rename(columns=column_mapping, inplace=True)
        
        # 빈도, 강도 열 식별 및 변환
        numeric_columns = ['빈도', '강도']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # 4) 필수 열이 없는 경우 기본값 설정
        if '빈도' not in df.columns:
            df['빈도'] = 3
        if '강도' not in df.columns:
            df['강도'] = 3

        # 5) T, 등급 계산
        df["T"] = df["빈도"] * df["강도"]
        df["등급"] = df["T"].apply(determine_grade)

        # 6) 개선대책 열이 없는 경우 기본값 설정
        if "개선대책" not in df.columns:
            alt_cols = [c for c in df.columns if "개선" in c or "대책" in c or "Corrective" in c]
            if alt_cols:
                df.rename(columns={alt_cols[0]: "개선대책"}, inplace=True)
            else:
                df["개선대책"] = "안전 교육 실시 및 보호구 착용"

        # 7) 최종 열 정리
        required_cols = [
            "작업활동 및 내용",
            "유해위험요인 및 환경측면 영향", 
            "피해형태 및 환경영향",
            "빈도",
            "강도", 
            "T",
            "등급",
            "개선대책"
        ]
        
        # 존재하는 열만 선택
        final_cols = [col for col in required_cols if col in df.columns]
        df = df[final_cols]

        # 8) 빈 값 처리
        df = df.fillna({
            "작업활동 및 내용": "일반 작업",
            "유해위험요인 및 환경측면 영향": "일반적 위험",
            "피해형태 및 환경영향": "부상",
            "개선대책": "안전 조치 수행"
        })

        return df

    except Exception as e:
        st.warning(f"데이터 로딩 중 오류가 발생했습니다: {str(e)}")
        st.info("샘플 데이터를 사용합니다.")
        return create_sample_data()

def create_sample_data():
    """샘플 데이터 생성"""
    data = {
        "작업활동 및 내용": [
            "임시 현장 저장소에서 포크리프트를 이용한 철골 구조재 하역작업",
            "콘크리트/CMU 블록 설치 작업",
            "굴착 및 되메우기 작업",
            "고소 작업대를 이용한 외벽 작업",
            "용접 작업"
        ],
        "유해위험요인 및 환경측면 영향": [
            "다중 인양으로 인한 적재물 낙하",
            "불충분한 작업 발판으로 인한 추락",
            "굴착벽 붕괴로 인한 매몰",
            "안전대 미착용으로 인한 추락",
            "용접 흄 및 화재 위험"
        ],
        "피해형태 및 환경영향": [
            "타박상",
            "골절",
            "매몰",
            "추락사",
            "화상"
        ],
        "빈도": [3, 3, 2, 4, 2],
        "강도": [5, 4, 5, 5, 3],
        "개선대책": [
            "1) 다수의 철골재를 함께 인양하지 않도록 관리 2) 치수, 중량, 형상이 다른 재료를 함께 인양하지 않도록 관리",
            "1) 비계대 누락된 목판 설치 2) 안전대 부착설비 설치 및 사용 3) 비계 변경 시 타공종 외 작업자 작업 금지",
            "1) 적절한 사면 기울기 유지 2) 굴착면 보강 3) 정기적 지반 상태 점검",
            "1) 안전대 착용 의무화 2) 작업 전 안전교육 실시 3) 추락방지망 설치",
            "1) 적절한 환기시설 설치 2) 화재 예방 조치 3) 보호구 착용"
        ]
    }
    df = pd.DataFrame(data)
    df["T"] = df["빈도"] * df["강도"]
    df["등급"] = df["T"].apply(determine_grade)
    return df

def embed_texts_with_openai(texts, model="text-embedding-3-large", api_key=None):
    """OpenAI를 이용한 텍스트 임베딩 생성"""
    if api_key: 
        openai.api_key = api_key
    
    embeddings = []
    batch_size = 10  # 배치 처리로 효율성 향상
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        try:
            # 텍스트 전처리
            processed_texts = [str(txt).replace("\n", " ").strip() for txt in batch_texts]
            
            resp = openai.Embedding.create(
                model=model, 
                input=processed_texts
            )
            
            for j, item in enumerate(resp["data"]):
                embeddings.append(item["embedding"])
                
        except Exception as e:
            st.error(f"임베딩 생성 중 오류: {str(e)}")
            # 오류 시 더미 임베딩 추가
            for _ in batch_texts:
                embeddings.append([0] * 1536)
    
    return embeddings

def generate_with_gpt(prompt, api_key, language, max_retries=3):
    """향상된 GPT 생성 함수"""
    if api_key: 
        openai.api_key = api_key
    
    sys_prompts = {
        "Korean": "당신은 건설 현장 위험성 평가 전문가입니다. 정확하고 실용적인 한국어 답변을 제공하세요.",
        "English": "You are a construction site risk assessment expert. Provide accurate and practical responses in English.",
        "Chinese": "您是建筑工地风险评估专家。请用中文提供准确实用的回答。"
    }
    
    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model="gpt-4", 
                messages=[
                    {"role": "system", "content": sys_prompts.get(language, sys_prompts['Korean'])},
                    {"role": "user", "content": prompt}
                ], 
                temperature=0.1,  # 일관성을 위해 낮은 temperature
                max_tokens=500,   # 토큰 제한 증가
                top_p=0.9
            )
            return resp['choices'][0]['message']['content'].strip()
            
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"GPT 호출 오류 (시도 {attempt + 1}/{max_retries}): {str(e)}")
                return ""
            else:
                st.warning(f"GPT 호출 재시도 중... ({attempt + 1}/{max_retries})")
                continue

# ----------------- 프롬프트 생성 함수들 (개선됨) -----------------

def construct_prompt_phase1_hazard(retrieved_docs, activity_text, language="Korean"):
    """향상된 유해위험요인 예측 프롬프트"""
    prompt_templates = {
        "Korean": {
            "intro": "건설 현장에서 다음과 같은 작업활동과 유해위험요인 사례들이 있습니다:\n\n",
            "example_format": "사례 {i}:\n- 작업활동: {activity}\n- 유해위험요인: {hazard}\n\n",
            "query_format": "위 사례들을 참고하여 다음 작업활동의 주요 유해위험요인을 구체적으로 예측해주세요:\n\n작업활동: {activity}\n\n예측된 유해위험요인: "
        },
        "English": {
            "intro": "Here are examples of work activities and associated hazards at construction sites:\n\n",
            "example_format": "Case {i}:\n- Work Activity: {activity}\n- Hazard: {hazard}\n\n", 
            "query_format": "Based on the above cases, please predict the main hazards for the following work activity:\n\nWork Activity: {activity}\n\nPredicted Hazard: "
        },
        "Chinese": {
            "intro": "以下是建筑工地工作活动和相关危害的例子:\n\n",
            "example_format": "案例 {i}:\n- 工作活动: {activity}\n- 危害: {hazard}\n\n",
            "query_format": "根据上述案例，请预测以下工作活动的主要危害:\n\n工作活动: {activity}\n\n预测的危害: "
        }
    }
    
    template = prompt_templates.get(language, prompt_templates["Korean"])
    
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            activity = doc['작업활동 및 내용']
            hazard = doc['유해위험요인 및 환경측면 영향']
            if pd.notna(activity) and pd.notna(hazard):
                retrieved_examples.append((activity, hazard))
        except:
            continue
    
    prompt = template["intro"]
    for i, (activity, hazard) in enumerate(retrieved_examples[:5], 1):  # 최대 5개 예시
        prompt += template["example_format"].format(i=i, activity=activity, hazard=hazard)
    
    prompt += template["query_format"].format(activity=activity_text)
    return prompt

def construct_prompt_phase1_risk(retrieved_docs, activity_text, hazard_text, language="Korean"):
    """향상된 위험도 평가 프롬프트"""
    prompt_templates = {
        "Korean": {
            "intro": "건설 현장 위험성 평가 기준:\n- 빈도(1-5): 1=매우드물게, 2=드물게, 3=가끔, 4=자주, 5=매우자주\n- 강도(1-5): 1=경미한부상, 2=가벼운부상, 3=중간부상, 4=심각한부상, 5=사망\n- T값 = 빈도 × 강도\n\n참고 사례들:\n\n",
            "example_format": "사례 {i}:\n입력: {input}\n평가: 빈도={freq}, 강도={intensity}, T값={t_value}\n\n",
            "query_format": "위 기준과 사례를 참고하여 다음을 평가해주세요:\n\n작업활동: {activity}\n유해위험요인: {hazard}\n\n다음 JSON 형식으로 정확히 답변하세요:\n{json_format}"
        },
        "English": {
            "intro": "Construction site risk assessment criteria:\n- Frequency(1-5): 1=Very Rare, 2=Rare, 3=Occasional, 4=Frequent, 5=Very Frequent\n- Intensity(1-5): 1=Minor Injury, 2=Light Injury, 3=Moderate Injury, 4=Serious Injury, 5=Fatality\n- T-value = Frequency × Intensity\n\nReference cases:\n\n",
            "example_format": "Case {i}:\nInput: {input}\nAssessment: Frequency={freq}, Intensity={intensity}, T-value={t_value}\n\n",
            "query_format": "Based on the above criteria and cases, please assess the following:\n\nWork Activity: {activity}\nHazard: {hazard}\n\nRespond in the following JSON format:\n{json_format}"
        },
        "Chinese": {
            "intro": "建筑工地风险评估标准:\n- 频率(1-5): 1=非常罕见, 2=罕见, 3=偶尔, 4=频繁, 5=非常频繁\n- 强度(1-5): 1=轻微伤害, 2=轻伤, 3=中度伤害, 4=严重伤害, 5=死亡\n- T值 = 频率 × 强度\n\n参考案例:\n\n",
            "example_format": "案例 {i}:\n输入: {input}\n评估: 频率={freq}, 强度={intensity}, T值={t_value}\n\n",
            "query_format": "根据上述标准和案例，请评估以下内容:\n\n工作活动: {activity}\n危害: {hazard}\n\n请以以下JSON格式回答:\n{json_format}"
        }
    }
    
    json_formats = {
        "Korean": '{"빈도": 숫자, "강도": 숫자, "T": 숫자}',
        "English": '{"frequency": number, "intensity": number, "T": number}',
        "Chinese": '{"频率": 数字, "强度": 数字, "T": 数字}'
    }
    
    template = prompt_templates.get(language, prompt_templates["Korean"])
    json_format = json_formats.get(language, json_formats["Korean"])
    
    retrieved_examples = []
    for _, doc in retrieved_docs.iterrows():
        try:
            example_input = f"{doc['작업활동 및 내용']} - {doc['유해위험요인 및 환경측면 영향']}"
            frequency = int(doc['빈도'])
            intensity = int(doc['강도'])
            T_value = frequency * intensity
            retrieved_examples.append((example_input, frequency, intensity, T_value))
        except:
            continue
    
    prompt = template["intro"]
    for i, (example_input, freq, intensity, t_val) in enumerate(retrieved_examples[:3], 1):
        prompt += template["example_format"].format(
            i=i, input=example_input, freq=freq, intensity=intensity, t_value=t_val
        )
    
    prompt += template["query_format"].format(
        activity=activity_text, hazard=hazard_text, json_format=json_format
    )
    
    return prompt

def parse_gpt_output_phase1(gpt_output, language="Korean"):
    """향상된 GPT 출력 파싱"""
    json_patterns = {
        "Korean": r'\{"빈도":\s*([1-5]),\s*"강도":\s*([1-5]),\s*"T":\s*([0-9]+)\}',
        "English": r'\{"frequency":\s*([1-5]),\s*"intensity":\s*([1-5]),\s*"T":\s*([0-9]+)\}',
        "Chinese": r'\{"频率":\s*([1-5]),\s*"强도":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    }
    
    # 현재 언어 패턴부터 시도
    pattern = json_patterns.get(language, json_patterns["Korean"])
    match = re.search(pattern, gpt_output)
    
    if match:
        pred_frequency = int(match.group(1))
        pred_intensity = int(match.group(2))
        pred_T = int(match.group(3))
        return pred_frequency, pred_intensity, pred_T
    
    # 다른 언어 패턴들도 시도
    for lang, pattern in json_patterns.items():
        if lang != language:
            match = re.search(pattern, gpt_output)
            if match:
                pred_frequency = int(match.group(1))
                pred_intensity = int(match.group(2))
                pred_T = int(match.group(3))
                return pred_frequency, pred_intensity, pred_T
    
    # 대체 파싱 시도 (숫자만 추출)
    numbers = re.findall(r'\b([1-5])\b', gpt_output)
    if len(numbers) >= 2:
        freq, intensity = int(numbers[0]), int(numbers[1])
        return freq, intensity, freq * intensity
    
    return None

def construct_prompt_phase2(retrieved_docs, activity_text, hazard_text, freq, intensity, T, target_language="Korean"):
    """향상된 개선대책 생성 프롬프트"""
    # 기존 construct_prompt_phase2 함수 내용을 그대로 유지하되, 
    # 더 구체적이고 실용적인 개선대책을 생성하도록 개선
    
    example_section = ""
    examples_added = 0
    
    # 언어별 필드명 (기존과 동일)
    field_names = {
        "Korean": {
            "improvement_fields": ['개선대책 및 세부관리방안', '개선대책', '개선방안'],
            "activity": "작업활동 및 내용",
            "hazard": "유해위험요인 및 환경측면 영향",
            "freq": "빈도",
            "intensity": "강도",
        },
        "English": {
            "improvement_fields": ['Improvement Measures', 'Improvement Plan', 'Countermeasures'],
            "activity": "작업활동 및 내용",
            "hazard": "유해위험요인 및 환경측면 영향",
            "freq": "빈도",
            "intensity": "강도",
        },
        "Chinese": {
            "improvement_fields": ['改进措施', '改进计划', '对策'],
            "activity": "작업활동 및 내용", 
            "hazard": "유해위험요인 및 환경측면 영향",
            "freq": "빈도",
            "intensity": "강도",
        }
    }
    
    fields = field_names.get(target_language, field_names["Korean"])
    
    # 예시 데이터 수집
    for _, row in retrieved_docs.iterrows():
        try:
            improvement_plan = ""
            for field in fields["improvement_fields"]:
                if field in row and pd.notna(row[field]):
                    improvement_plan = row[field]
                    break
            
            if not improvement_plan:
                continue
                
            original_freq = int(row[fields["freq"]]) if fields["freq"] in row else 3
            original_intensity = int(row[fields["intensity"]]) if fields["intensity"] in row else 3
            original_T = original_freq * original_intensity
                
            # 개선 후 데이터 (추정)
            improved_freq = max(1, original_freq - 1)
            improved_intensity = max(1, original_intensity - 1) 
            improved_T = improved_freq * improved_intensity
            
            if target_language == "Korean":
                example_section += f"""
예시 {examples_added + 1}:
입력 작업활동: {row[fields['activity']]}
입력 유해위험요인: {row[fields['hazard']]}
입력 원래 빈도: {original_freq}
입력 원래 강도: {original_intensity}
입력 원래 T값: {original_T}
출력 (개선계획 및 위험감소) JSON 형식:
{{
  "개선대책": "{improvement_plan}",
  "개선 후 빈도": {improved_freq},
  "개선 후 강도": {improved_intensity}, 
  "개선 후 T": {improved_T},
  "T 감소율": {compute_rrr(original_T, improved_T):.2f}
}}

"""
            elif target_language == "English":
                example_section += f"""
Example {examples_added + 1}:
Input Work Activity: {row[fields['activity']]}
Input Hazard: {row[fields['hazard']]}
Input Original Frequency: {original_freq}
Input Original Intensity: {original_intensity}
Input Original T-value: {original_T}
Output (Improvement Plan and Risk Reduction) JSON format:
{{
  "improvement_plan": "{improvement_plan}",
  "improved_frequency": {improved_freq},
  "improved_intensity": {improved_intensity},
  "improved_T": {improved_T},
  "reduction_rate": {compute_rrr(original_T, improved_T):.2f}
}}

"""
            else:  # Chinese
                example_section += f"""
示例 {examples_added + 1}:
输入工作活动: {row[fields['activity']]}
输入危害: {row[fields['hazard']]}
输入原频率: {original_freq}
输入原强度: {original_intensity}
输入原T值: {original_T}
输出 (改进计划和风险降低) JSON格式:
{{
  "改进措施": "{improvement_plan}",
  "改进后频率": {improved_freq},
  "改进后强度": {improved_intensity},
  "改进后T值": {improved_T},
  "T值降低率": {compute_rrr(original_T, improved_T):.2f}
}}

"""
            
            examples_added += 1
            if examples_added >= 3:
                break
                
        except Exception as e:
            continue
    
    # 기본 예시 (예시가 없는 경우)
    if examples_added == 0:
        if target_language == "Korean":
            example_section = """
예시 1:
입력 작업활동: 굴착 및 되메우기 작업
입력 유해위험요인: 굴착벽 붕괴로 인한 매몰
입력 원래 빈도: 3
입력 원래 강도: 4
입력 원래 T값: 12
출력 (개선계획 및 위험감소) JSON 형식:
{
  "개선대책": "1) 토양 분류에 따른 적절한 경사 유지 2) 굴착 벽면 보강 3) 정기적인 지반 상태 검사 실시",
  "개선 후 빈도": 1,
  "개선 후 강도": 2,
  "개선 후 T": 2,
  "T 감소율": 83.33
}

"""
        elif target_language == "English":
            example_section = """
Example 1:
Input Work Activity: Excavation and backfilling
Input Hazard: Collapse of excavation wall
Input Original Frequency: 3
Input Original Intensity: 4
Input Original T-value: 12
Output (Improvement Plan and Risk Reduction) JSON format:
{
  "improvement_plan": "1) Maintain proper slope according to soil classification 2) Reinforce excavation walls 3) Conduct regular ground condition inspections",
  "improved_frequency": 1,
  "improved_intensity": 2,
  "improved_T": 2,
  "reduction_rate": 83.33
}

"""
        else:  # Chinese
            example_section = """
示例 1:
输入工作活动: 挖掘和回填作业
输入危害: 挖掘墙壁倒塌
输入原频率: 3
输入原强度: 4
输入原T值: 12
输出 (改进计划和风险降低) JSON格式:
{
  "改进措施": "1) 根据土壤分类维持适当的斜坡 2) 加固挖掘墙壁 3) 定期进行地面状况检查",
  "改进后频率": 1,
  "改进后强度": 2,
  "改进后T值": 2,
  "T值降低率": 83.33
}

"""
    
    # 언어별 JSON 키 및 안내문
    json_keys = {
        "Korean": {
            "improvement": "개선대책",
            "improved_freq": "개선 후 빈도",
            "improved_intensity": "개선 후 강도",
            "improved_t": "개선 후 T",
            "reduction_rate": "T 감소율"
        },
        "English": {
            "improvement": "improvement_plan",
            "improved_freq": "improved_frequency", 
            "improved_intensity": "improved_intensity",
            "improved_t": "improved_T",
            "reduction_rate": "reduction_rate"
        },
        "Chinese": {
            "improvement": "改进措施",
            "improved_freq": "改进后频率",
            "improved_intensity": "改进后强度",
            "improved_t": "改进后T值", 
            "reduction_rate": "T值降低率"
        }
    }
    
    instructions = {
        "Korean": {
            "new_input": "이제 새로운 입력입니다:",
            "output_format": "다음 JSON 형식으로 실용적이고 구체적인 개선대책을 제공하세요:",
            "requirements": "개선대책은 실제 현장에서 적용 가능한 구체적인 방법을 3개 이상 번호를 매겨 제시하세요. 개선 후 빈도와 강도는 합리적으로 감소된 값으로 설정하세요.",
            "output": "출력:"
        },
        "English": {
            "new_input": "Now here is a new input:",
            "output_format": "Please provide practical and specific improvement measures in the following JSON format:",
            "requirements": "Improvement measures should include at least 3 specific, field-applicable methods in a numbered list. Set improved frequency and intensity to reasonably reduced values.",
            "output": "Output:"
        },
        "Chinese": {
            "new_input": "现在是新的输入:",
            "output_format": "请以以下JSON格式提供实用且具体的改进措施:",
            "requirements": "改进措施应包括至少3个具体的、现场可应用的方法，以编号列表形式。将改进后的频率和强度设置为合理降低的值。",
            "output": "输出:"
        }
    }
    
    keys = json_keys.get(target_language, json_keys["Korean"])
    instr = instructions.get(target_language, instructions["Korean"])
    
    # 최종 프롬프트 구성
    prompt = f"""{example_section}
{instr['new_input']}
입력 작업활동: {activity_text}
입력 유해위험요인: {hazard_text}
입력 원래 빈도: {freq}
입력 원래 강도: {intensity}
입력 원래 T값: {T}

{instr['output_format']}
{{
  "{keys["improvement"]}": "번호가 매겨진 구체적 개선대책 리스트",
  "{keys["improved_freq"]}": (1-5 사이의 정수),
  "{keys["improved_intensity"]}": (1-5 사이의 정수),
  "{keys["improved_t"]}": (개선 후 빈도 × 개선 후 강도),
  "{keys["reduction_rate"]}": (위험 감소 백분율)
}}

{instr['requirements']}
{instr['output']}
"""
    
    return prompt

def parse_gpt_output_phase2(gpt_output, language="Korean"):
    """향상된 Phase 2 출력 파싱"""
    try:
        # JSON 블록 추출
        json_match = re.search(r'```json\s*(.*?)\s*```', gpt_output, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # JSON 객체 직접 추출
            json_match = re.search(r'\{.*\}', gpt_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = gpt_output

        import json
        result = json.loads(json_str)
        
        # 언어별 키 매핑
        key_mappings = {
            "Korean": {
                "improvement": ["개선대책"],
                "improved_freq": ["개선 후 빈도", "개선빈도"],
                "improved_intensity": ["개선 후 강도", "개선강도"],
                "improved_t": ["개선 후 T", "개선T"],
                "reduction_rate": ["T 감소율", "감소율", "위험 감소율"]
            },
            "English": {
                "improvement": ["improvement_plan", "improvement_measures"],
                "improved_freq": ["improved_frequency", "new_frequency"],
                "improved_intensity": ["improved_intensity", "new_intensity"],
                "improved_t": ["improved_T", "new_T"],
                "reduction_rate": ["reduction_rate", "risk_reduction_rate"]
            },
            "Chinese": {
                "improvement": ["改进措施", "改进计划"],
                "improved_freq": ["改进后频率", "新频率"],
                "improved_intensity": ["改进后强度", "新强度"],
                "improved_t": ["改进后T值", "新T值"],
                "reduction_rate": ["T值降低率", "降低率"]
            }
        }
        
        mappings = key_mappings.get(language, key_mappings["Korean"])
        mapped_result = {}
        
        # 키 매핑 수행
        for result_key, possible_keys in mappings.items():
            for key in possible_keys:
                if key in result:
                    mapped_result[result_key] = result[key]
                    break
        
        # 필수 값 검증 및 기본값 설정
        if "improved_freq" not in mapped_result:
            mapped_result["improved_freq"] = 1
        if "improved_intensity" not in mapped_result:
            mapped_result["improved_intensity"] = 1
        if "improved_t" not in mapped_result:
            mapped_result["improved_t"] = mapped_result["improved_freq"] * mapped_result["improved_intensity"]
        if "reduction_rate" not in mapped_result:
            mapped_result["reduction_rate"] = 50.0  # 기본값
            
        return mapped_result
        
    except Exception as e:
        st.error(f"개선대책 파싱 중 오류 발생: {str(e)}")
        st.write("원본 GPT 응답:", gpt_output)
        return {
            "improvement": "안전 교육 실시 및 보호구 착용 의무화",
            "improved_freq": 1,
            "improved_intensity": 1, 
            "improved_t": 1,
            "reduction_rate": 50.0
        }

# -----------------------------------------------------------------------------
# ---------------------------  Overview 탭 ------------------------------------
# -----------------------------------------------------------------------------
with tabs[0]:
    st.markdown(f'<div class="sub-header">{texts["overview_header"]}</div>', unsafe_allow_html=True)
    
    col_overview, col_features = st.columns([3, 2])
    
    with col_overview:
        st.markdown(f"<div class='info-text'>{texts['overview_text']}</div>", unsafe_allow_html=True)
        
        # 시스템 메트릭 표시
        col_metric1, col_metric2, col_metric3 = st.columns(3)
        
        with col_metric1:
            st.metric(texts["supported_languages"], texts["languages_count"], texts["languages_detail"])
        with col_metric2:
            st.metric(texts["assessment_phases"], texts["phases_count"], texts["phases_detail"])
        with col_metric3:
            st.metric(texts["risk_grades"], texts["grades_count"], texts["grades_detail"])
    
    with col_features:
        st.markdown(f"**{texts['features_title']}**")
        st.markdown(texts['phase1_features'])
        st.markdown(texts['phase2_features'])

# -----------------------------------------------------------------------------
# --------------  Risk Assessment 통합 탭 --------------------------------------
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown(f'<div class="sub-header">{texts["tab_phase1"]} & {texts["tab_phase2"]}</div>', unsafe_allow_html=True)

    # ① API Key & Dataset Setup
    col_api, col_dataset = st.columns([2, 1])
    
    with col_api:
        api_key = st.text_input(texts['api_key_label'], type='password', key='api_key_all')
    
    with col_dataset:
        dataset_name = st.selectbox(texts['dataset_label'], [
            "SWRO 건축공정 (건축)",
            "Civil (토목)", 
            "Marine (토목)",
            "SWRO 기계공사 (플랜트)",
            "SWRO 전기작업표준 (플랜트)"
        ], key='dataset_all')

    # ② Data Loading Section
    if ss.retriever_pool_df is None or st.button(texts['load_data_btn'], type="primary"):
        if not api_key:
            st.warning(texts['api_key_warning'])
        else:
            with st.spinner(texts['data_loading']):
                try:
                    df = load_data(dataset_name)
                    
                    # 데이터 분할 및 처리
                    if len(df) > 10:
                        train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
                    else:
                        train_df = df.copy()
                    
                    pool_df = train_df.copy()
                    pool_df['content'] = pool_df.apply(lambda r: ' '.join(r.values.astype(str)), axis=1)
                    
                    # 임베딩 처리
                    to_embed = pool_df['content'].tolist()
                    max_texts = min(len(to_embed), 30)  # 처리 개수 증가
                    
                    st.info(texts['demo_limit_info'].format(max_texts=max_texts))
                    
                    # 배치 처리로 임베딩 생성
                    embeds = embed_texts_with_openai(to_embed[:max_texts], api_key=api_key)
                    
                    # FAISS 인덱스 구성
                    vecs = np.array(embeds, dtype='float32')
                    dim = vecs.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(vecs)
                    
                    # 세션 상태 업데이트
                    ss.index = index
                    ss.embeddings = vecs
                    ss.retriever_pool_df = pool_df.iloc[:max_texts]
                    
                    st.success(texts['data_load_success'].format(max_texts=max_texts))
                    
                    # 데이터 미리보기
                    with st.expander("📊 로드된 데이터 미리보기"):
                        st.dataframe(df.head(), use_container_width=True)
                        
                except Exception as e:
                    st.error(f"데이터 로딩 중 오류가 발생했습니다: {str(e)}")

    st.divider()

    # ③ Risk Assessment Interface
    st.markdown("### 🔍 위험성 평가 수행")
    
    # 작업활동 입력
    activity = st.text_area(
        texts['activity_label'], 
        placeholder="예: 임시 현장 저장소에서 포크리프트를 이용한 철골 구조재 하역작업",
        height=100,
        key='user_activity'
    )
    
    # 추가 옵션
    col_options1, col_options2 = st.columns(2)
    
    with col_options1:
        include_similar_cases = st.checkbox("유사 사례 포함", value=True)
        
    with col_options2:
        result_language = st.selectbox("결과 언어", ["Korean", "English", "Chinese"], 
                                     index=["Korean", "English", "Chinese"].index(ss.language))

    # 실행 버튼
    run_button = st.button("🚀 위험성 평가 실행", type="primary", use_container_width=True)

    if run_button and activity:
        if not api_key:
            st.warning(texts['api_key_warning'])
        elif ss.index is None:
            st.warning(texts['load_first_warning'])
        else:
            with st.spinner("위험성 평가를 수행하는 중..."):
                try:
                    # === Phase 1: Risk Assessment ===
                    
                    # 1) 유사 사례 검색
                    q_emb = embed_texts_with_openai([activity], api_key=api_key)[0]
                    D, I = ss.index.search(np.array([q_emb], dtype='float32'), 
                                         k=min(10, len(ss.retriever_pool_df)))
                    sim_docs = ss.retriever_pool_df.iloc[I[0]]

                    # 2) 유해위험요인 예측
                    hazard_prompt = construct_prompt_phase1_hazard(sim_docs, activity, result_language)
                    hazard = generate_with_gpt(hazard_prompt, api_key, result_language)

                    # 3) 빈도·강도 예측
                    risk_prompt = construct_prompt_phase1_risk(sim_docs, activity, hazard, result_language)
                    risk_json = generate_with_gpt(risk_prompt, api_key, result_language)
                    
                    parse_result = parse_gpt_output_phase1(risk_json, result_language)
                    if not parse_result:
                        st.error(texts['parsing_error'])
                        st.expander("GPT 원문 응답").write(risk_json)
                        st.stop()
                    
                    freq, intensity, T = parse_result
                    grade = determine_grade(T)

                    # === Phase 2: Improvement Measures ===
                    
                    # 4) 개선대책 생성
                    improvement_prompt = construct_prompt_phase2(
                        sim_docs, activity, hazard, freq, intensity, T, result_language
                    )
                    improvement_response = generate_with_gpt(improvement_prompt, api_key, result_language)
                    
                    parsed_improvement = parse_gpt_output_phase2(improvement_response, result_language)
                    if not parsed_improvement:
                        st.error(texts['parsing_error_improvement'])
                        st.expander("GPT 원문 응답").write(improvement_response)
                        st.stop()

                    # 개선대책 결과 추출
                    improvement_plan = parsed_improvement.get('improvement', '')
                    improved_freq = parsed_improvement.get('improved_freq', 1)
                    improved_intensity = parsed_improvement.get('improved_intensity', 1)
                    improved_T = parsed_improvement.get('improved_t', improved_freq * improved_intensity)
                    rrr = compute_rrr(T, improved_T)

                    # === Results Display ===
                    
                    # Phase 1 결과 표시
                    st.markdown("## 📋 Phase 1: 위험성 평가 결과")
                    
                    col_result1, col_result2 = st.columns([2, 1])
                    
                    with col_result1:
                        st.markdown(f"**작업활동:** {activity}")
                        st.markdown(f"**예측된 유해위험요인:** {hazard}")
                        
                        # 위험도 테이블
                        result_df = pd.DataFrame({
                            texts['result_table_columns'][0]: texts['result_table_rows'],
                            texts['result_table_columns'][1]: [freq, intensity, T, grade]
                        })
                        st.dataframe(result_df, use_container_width=True, hide_index=True)
                    
                    with col_result2:
                        # 위험등급 시각화
                        grade_color = get_grade_color(grade)
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; background-color:{grade_color}; 
                                    color:white; border-radius:10px; margin:10px 0;">
                            <h2 style="margin:0;">위험등급</h2>
                            <h1 style="margin:10px 0; font-size:3rem;">{grade}</h1>
                            <p style="margin:0;">T값: {T}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # 유사 사례 표시 (옵션)
                    if include_similar_cases:
                        st.markdown("### 🔍 유사한 사례")
                        
                        similar_records = []
                        for i, (_, doc) in enumerate(sim_docs.iterrows(), 1):
                            plan, imp_f, imp_i, imp_t = _extract_improvement_info(doc)
                            
                            # 사례 표시
                            with st.expander(f"사례 {i}: {doc['작업활동 및 내용'][:50]}..."):
                                col_case1, col_case2 = st.columns(2)
                                
                                with col_case1:
                                    st.write(f"**작업활동:** {doc['작업활동 및 내용']}")
                                    st.write(f"**유해위험요인:** {doc['유해위험요인 및 환경측면 영향']}")
                                    st.write(f"**위험도:** 빈도 {doc['빈도']}, 강도 {doc['강도']}, T값 {doc['T']} (등급 {doc['등급']})")
                                
                                with col_case2:
                                    st.write(f"**개선대책:** {plan if plan else '미제공'}")
                                    if plan:
                                        st.write(f"**개선 후:** F={imp_f}, I={imp_i}, T={imp_t}")
                            
                            # 엑셀용 레코드
                            similar_records.append({
                                "No": i,
                                "작업활동": doc['작업활동 및 내용'],
                                "유해위험요인": doc['유해위험요인 및 환경측면 영향'],
                                "빈도": doc['빈도'],
                                "강도": doc['강도'],
                                "T": doc['T'],
                                "위험등급": doc['등급'],
                                "개선대책": plan,
                            })

                    # Phase 2 결과 표시
                    st.markdown("## 🛠️ Phase 2: 개선대책 생성 결과")
                    
                    col_improvement1, col_improvement2 = st.columns([3, 2])
                    
                    with col_improvement1:
                        st.markdown(f"### {texts['improvement_plan_header']}")
                    
                        # 1) 2) 등 번호 뒤에 줄바꿈 넣기
                        import re
                        # 예: "1) 첫번째 2) 두번째 3) 세번째" -> 
                        #    "1) 첫번째\n2) 두번째\n3) 세번째"
                        plan_md = re.sub(r'(\d\))\s*', r'\1  \n', improvement_plan.strip())
                    
                        st.markdown(plan_md)
                    
                    with col_improvement2:
                        st.markdown(f"### {texts['risk_improvement_header']}")
                        
                        # 개선 전후 비교 테이블
                        comparison_df = pd.DataFrame({
                            texts['comparison_columns'][0]: texts['result_table_rows'],
                            texts['comparison_columns'][1]: [freq, intensity, T, grade],
                            texts['comparison_columns'][2]: [improved_freq, improved_intensity, improved_T, determine_grade(improved_T)]
                        })
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                        
                        # 위험 감소율 표시
                        st.metric(
                            label=texts['risk_reduction_label'],
                            value=f"{rrr:.1f}%",
                            delta=f"-{T-improved_T} T값"
                        )

                    # 위험도 변화 시각화
                    st.markdown("### 📊 위험도 변화 시각화")
                    
                    col_vis1, col_vis2 = st.columns(2)
                    
                    with col_vis1:
                        st.markdown("**개선 전 위험도**")
                        progress_before = min(T / 25, 1.0)  # 최대 25로 정규화
                        st.progress(progress_before)
                        st.caption(f"T값: {T} (등급: {grade})")
                    
                    with col_vis2:
                        st.markdown("**개선 후 위험도**")
                        progress_after = min(improved_T / 25, 1.0)
                        st.progress(progress_after)
                        st.caption(f"T값: {improved_T} (등급: {determine_grade(improved_T)})")

                    # 결과를 세션에 저장
                    ss.last_assessment = {
                        'activity': activity,
                        'hazard': hazard,
                        'freq': freq,
                        'intensity': intensity,
                        'T': T,
                        'grade': grade,
                        'improvement_plan': improvement_plan,
                        'improved_freq': improved_freq,
                        'improved_intensity': improved_intensity,
                        'improved_T': improved_T,
                        'rrr': rrr,
                        'similar_cases': similar_records if include_similar_cases else []
                    }

                    # Excel 다운로드 기능
                    st.markdown("### 💾 결과 다운로드")
                    
                    def create_excel_download():
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                            # ─── Phase 1 결과 시트 ─────────────────────────────
                            phase1_df = pd.DataFrame({
                                "항목": ["작업활동", "유해위험요인", "빈도", "강도", "T값", "위험등급"],
                                "값": [activity, hazard, freq, intensity, T, grade]
                            })
                            phase1_df.to_excel(writer, sheet_name="Phase1_결과", index=False)
                    
                            # ─── Phase 2 결과 시트 ─────────────────────────────
                            phase2_df = pd.DataFrame({
                                "항목": ["개선대책", "개선 후 빈도", "개선 후 강도", "개선 후 T값", "개선 후 등급", "위험 감소율"],
                                "값": [improvement_plan,
                                        improved_freq,
                                        improved_intensity,
                                        improved_T,
                                        determine_grade(improved_T),
                                        f"{rrr:.2f}%"]
                            })
                            phase2_df.to_excel(writer, sheet_name="Phase2_결과", index=False)
                    
                            # ─── 비교 분석 시트 ───────────────────────────────
                            comparison_detail_df = pd.DataFrame({
                                "항목": ["빈도", "강도", "T값", "위험등급"],
                                "개선 전": [freq, intensity, T, grade],
                                "개선 후": [improved_freq,
                                          improved_intensity,
                                          improved_T,
                                          determine_grade(improved_T)],
                                "개선율": [
                                    f"{(freq-improved_freq)/freq*100:.1f}%" if freq > 0 else "0%",
                                    f"{(intensity-improved_intensity)/intensity*100:.1f}%" if intensity > 0 else "0%",
                                    f"{rrr:.1f}%",
                                    f"{grade} → {determine_grade(improved_T)}"
                                ]
                            })
                            comparison_detail_df.to_excel(writer, sheet_name="비교분석", index=False)
                    
                            # ─── 유사사례 시트 ─────────────────────────────────
                            # 이미 similar_records 에는 10건이 들어와 있다고 가정
                            if similar_records:
                                sim_df = pd.DataFrame(similar_records)
                    
                                # 개선 후 빈도·강도 계산
                                sim_df["개선 후 빈도"] = sim_df["빈도"].astype(int).apply(lambda x: max(1, x - 1))
                                sim_df["개선 후 강도"] = sim_df["강도"].astype(int).apply(lambda x: max(1, x - 1))
                    
                                # 내보낼 컬럼 구성
                                export_df = pd.DataFrame({
                                    "작업활동 및 내용 Work Sequence":      sim_df["작업활동"],
                                    "유해위험요인 및 환경측면 영향 Hazardous Factors": sim_df["유해위험요인"],
                                    "위험성 Risk – 빈도 likelihood":     sim_df["빈도"],
                                    "위험성 Risk – 강도 severity":      sim_df["강도"],
                                    "개선대책 및 세부관리방안 Control Measures":    sim_df["개선대책"],
                                    "위험성 Risk (개선 후) – 빈도 likelihood": sim_df["개선 후 빈도"],
                                    "위험성 Risk (개선 후) – 강도 severity":  sim_df["개선 후 강도"],
                                })
                    
                                export_df.to_excel(writer, sheet_name="유사사례", index=False)
                    
                                # 전체 컬럼 빨간색, 줄바꿈 적용
                                workbook = writer.book
                                worksheet = writer.sheets["유사사례"]
                                red_fmt = workbook.add_format({
                                    "font_color": "#FF0000",
                                    "text_wrap": True
                                })
                                for col_idx in range(len(export_df.columns)):
                                    # 폭 20, 서식 적용
                                    worksheet.set_column(col_idx, col_idx, 20, red_fmt)
                    
                        return output.getvalue()


# ------------------- 푸터 ------------------------
st.markdown('<hr style="margin-top: 3rem;">', unsafe_allow_html=True)

footer_col1, footer_col2, footer_col3 = st.columns([1, 1, 1])

with footer_col1:
    if os.path.exists('cau.png'):
        st.image('cau.png', width=140)

with footer_col2:
    st.markdown("""
    <div style="text-align: center; padding: 20px;">
        <h4>두산에너빌리티</h4>
        <p>AI 기반 위험성 평가 시스템</p>
        <p style="font-size: 0.8rem; color: #666;">
            © 2025 Doosan Enerbility. All rights reserved.
        </p>
    </div>
    """, unsafe_allow_html=True)

with footer_col3:
    if os.path.exists('doosan.png'):
        st.image('doosan.png', width=160)
