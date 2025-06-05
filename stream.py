import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import io
from sklearn.model_selection import train_test_split
from openai import OpenAI

# ----------------- 시스템 다국어 텍스트 -----------------
system_texts = {
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "시스템 개요",
        "tab_phase": "위험성 평가 & 개선대책",
        "overview_header": "LLM 기반 위험성평가 시스템",
        "overview_text": (
            "Doosan Enerbility AI Risk Assessment는 국내 및 해외 건설현장의 '수시 위험성 평가' 및 "
            "'노동부 중대재해 사례'를 학습하여 개발된 자동 위험성평가 프로그램입니다. "
            "생성된 위험성 평가는 반드시 수시 위험성평가 심의회를 통해 검증 후 사용하시기 바랍니다."
        ),
        "features_title": "시스템 특징 및 구성요소",
        "phase_features": (
            "#### Phase 1: 위험성 평가 자동화\n"
            "- 공정별 작업활동에 따른 위험성평가 데이터 학습\n"
            "- 작업활동 입력 시 유해위험요인 자동 예측 (영어 내부 실행)\n"
            "- 유사 사례 검색 및 표시 (영어 내부 처리 → 최종 출력 번역)\n"
            "- LLM 기반 위험도(빈도, 강도, T) 측정 (영어 내부 실행)\n"
            "- 위험등급(A–E) 자동 산정\n\n"
            "#### Phase 2: 개선대책 자동 생성\n"
            "- 맞춤형 개선대책 자동 생성 (영어 내부 실행)\n"
            "- 다국어(한국어/영어/중국어) 개선대책 생성 지원\n"
            "- 개선 전후 위험도(T) 자동 비교 분석\n"
            "- 공종/공정별 최적 개선대책 데이터베이스 구축"
        ),
        "supported_languages_label": "지원 언어",
        "supported_languages_value": "3개",
        "supported_languages_detail": "한/영/중",
        "assessment_phases_label": "평가 단계",
        "assessment_phases_value": "2단계",
        "assessment_phases_detail": "Phase1+Phase2",
        "risk_grades_label": "위험등급",
        "risk_grades_value": "5등급",
        "risk_grades_detail": "A~E",
        "api_key_label": "OpenAI API 키를 입력하세요:",
        "dataset_label": "데이터셋 선택",
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "계속하려면 OpenAI API 키를 입력하세요.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "데모 목적으로 {max_texts}개의 텍스트만 임베딩합니다. 실제 환경에서는 전체 데이터를 처리해야 합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! (총 {max_texts}개 항목 처리)",
        "load_first_warning": "먼저 [데이터 로드 및 인덱스 구성] 버튼을 클릭하세요.",
        "activity_label": "작업활동:",
        "include_similar_cases": "유사 사례 포함",
        "run_assessment": "🚀 위험성 평가 실행",
        "activity_warning": "작업활동을 입력하세요.",
        "performing_assessment": "위험성 평가를 수행하는 중...",
        "phase1_results": "📋 Phase 1: 위험성 평가 결과",
        "work_activity": "작업활동",
        "predicted_hazard": "예측된 유해위험요인",
        "risk_grade_display": "위험등급",
        "t_value_display": "T값",
        "risk_level_text": "위험도 : 빈도 {freq}, 강도 {intensity}, T {T} ({grade})",
        "similar_cases_section": "🔍 유사한 사례",
        "case_number": "사례",
        "phase2_results": "🛠️ Phase 2: 개선대책 생성 결과",
        "improvement_plan_header": "개선대책",
        "risk_improvement_header": "위험도 개선 결과",
        "comparison_columns": ["항목", "개선 전", "개선 후"],
        "risk_reduction_label": "위험 감소율 (RRR)",
        "risk_visualization": "📊 위험도 변화 시각화",
        "before_improvement": "개선 전",
        "after_improvement": "개선 후",
        "grade_label": "등급",
        "download_results": "💾 결과 다운로드",
        "excel_export": "📥 결과 Excel 다운로드",
        "col_activity_header": "작업활동 및 내용 Work Sequence",
        "col_hazard_header": "유해위험요인 및 환경측면 영향 Hazarous Factors",
        "col_ehs_header": "EHS",
        "col_risk_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_risk_severity_header": "위험성 Risk – 강도 severity",
        "col_control_header": "개선대책 및 세부관리방안 Control Measures",
        "col_incharge_header": "개선담당자 In Charge",
        "col_duedate_header": "개선일자 Correction Due Date",
        "col_after_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_after_severity_header": "위험성 Risk – 강도 severity"
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_phase": "Risk Assessment & Improvement",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": (
            "Doosan Enerbility AI Risk Assessment is an automated program trained on on-demand risk-assessment reports "
            "from domestic and overseas construction sites and major-accident cases compiled by Korea's Ministry of Employment "
            "and Labor. Please ensure that every generated assessment is reviewed and approved by the On-Demand Risk Assessment "
            "Committee before it is used."
        ),
        "features_title": "System Features and Components",
        "phase_features": (
            "#### Phase 1: Risk Assessment Automation\n"
            "- Learning risk assessment data per work activity\n"
            "- Automatic hazard prediction when work activities are entered (internal: English)\n"
            "- Similar case search & display (internal: English → final: translation)\n"
            "- LLM-based risk level (frequency, intensity, T) measurement (internal: English)\n"
            "- Automatic risk grade (A–E) calculation\n\n"
            "#### Phase 2: Automatic Generation of Improvement Measures\n"
            "- Customized improvement measures generation (internal: English)\n"
            "- Multilingual (Korean/English/Chinese) improvement measure support\n"
            "- Automatic comparative analysis before/after improvement\n"
            "- Database of optimal improvement measures per process"
        ),
        "supported_languages_label": "Supported Languages",
        "supported_languages_value": "3",
        "supported_languages_detail": "KOR/ENG/CHN",
        "assessment_phases_label": "Assessment Phases",
        "assessment_phases_value": "2",
        "assessment_phases_detail": "Phase1+Phase2",
        "risk_grades_label": "Risk Grades",
        "risk_grades_value": "5",
        "risk_grades_detail": "A–E",
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset",
        "load_data_btn": "Load Data and Configure Index",
        "api_key_warning": "Please enter an OpenAI API key to continue.",
        "data_loading": "Loading data and configuring index...",
        "demo_limit_info": "For demo purposes, only embedding {max_texts} texts. In a real environment, process all data.",
        "data_load_success": "Data load and index configuration complete! (Total {max_texts} items processed)",
        "load_first_warning": "Please click [Load Data and Configure Index] first.",
        "activity_label": "Work Activity:",
        "include_similar_cases": "Include Similar Cases",
        "run_assessment": "🚀 Run Risk Assessment",
        "activity_warning": "Please enter a work activity.",
        "performing_assessment": "Performing risk assessment...",
        "phase1_results": "📋 Phase 1: Risk Assessment Results",
        "work_activity": "Work Activity",
        "predicted_hazard": "Predicted Hazard",
        "risk_grade_display": "Risk Grade",
        "t_value_display": "T Value",
        "risk_level_text": "Risk Level : Frequency {freq}, Intensity {intensity}, T {T} (Grade {grade})",
        "similar_cases_section": "🔍 Similar Cases",
        "case_number": "Case",
        "phase2_results": "🛠️ Phase 2: Improvement Measures Results",
        "improvement_plan_header": "Improvement Plan",
        "risk_improvement_header": "Risk Improvement Results",
        "comparison_columns": ["Item", "Before Improvement", "After Improvement"],
        "risk_reduction_label": "Risk Reduction Rate (RRR)",
        "risk_visualization": "📊 Risk Level Change Visualization",
        "before_improvement": "Before Improvement",
        "after_improvement": "After Improvement",
        "grade_label": "Grade",
        "download_results": "💾 Download Results",
        "excel_export": "📥 Download Excel Report",
        "col_activity_header": "작업활동 및 내용 Work Sequence",
        "col_hazard_header": "유해위험요인 및 환경측면 영향 Hazarous Factors",
        "col_ehs_header": "EHS",
        "col_risk_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_risk_severity_header": "위험성 Risk – 강도 severity",
        "col_control_header": "개선대책 및 세부관리방안 Control Measures",
        "col_incharge_header": "개선담당자 In Charge",
        "col_duedate_header": "개선일자 Correction Due Date",
        "col_after_likelihood_header": "위험성 Risk – 빈도 likelihood",
        "col_after_severity_header": "위험성 Risk – 강도 severity"
    },
    "Chinese": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "系统概述",
        "tab_phase": "风险评估 & 改进措施",
        "overview_header": "基于LLM的风险评估系统",
        "overview_text": (
            "Doosan Enerbility AI 风险评估系统是一款自动化风险评估程序，基于国内外施工现场的'临时风险评估'数据及韩国劳工部 "
            "重大事故案例训练开发而成。生成的风险评估结果必须经过临时风险评估审议委员会的审核后方可使用。"
        ),
        "features_title": "系统特点和组件",
        "phase_features": (
            "#### 第1阶段：风险评估自动化\n"
            "- 按工作活动学习风险评估数据\n"
            "- 输入工作活动时自动预测危害 (内部：英语)\n"
            "- 相似案例搜索与显示 (内部：英语 → 最终：翻译)\n"
            "- 基于LLM的风险等级（频率、强度、T）测量 (内部：英语)\n"
            "- 自动计算风险等级(A–E)\n\n"
            "#### 第2阶段：自动生成改进措施\n"
            "- 定制化改进措施自动生成 (内部：英语)\n"
            "- 多语言 (韩/英/中) 改进措施支持\n"
            "- 自动比较改进前后风险等级\n"
            "- 按工序管理最优改进措施数据库"
        ),
        "supported_languages_label": "支持语言",
        "supported_languages_value": "3 种",
        "supported_languages_detail": "韩/英/中",
        "assessment_phases_label": "评估阶段",
        "assessment_phases_value": "2 阶段",
        "assessment_phases_detail": "Phase1+Phase2",
        "risk_grades_label": "风险等级",
        "risk_grades_value": "5 等级",
        "risk_grades_detail": "A–E",
        "api_key_label": "输入 OpenAI API 密钥：",
        "dataset_label": "选择数据集",
        "load_data_btn": "加载数据并配置索引",
        "api_key_warning": "请输入 OpenAI API 密钥以继续。",
        "data_loading": "正在加载数据并配置索引...",
        "demo_limit_info": "演示用途仅嵌入 {max_texts} 个文本。实际环境应处理所有数据。",
        "data_load_success": "数据加载与索引配置完成！(共处理 {max_texts} 项目)",
        "load_first_warning": "请先点击 [加载数据并配置索引]。",
        "activity_label": "工作活动：",
        "include_similar_cases": "包括相似案例",
        "run_assessment": "🚀 运行风险评估",
        "activity_warning": "请输入工作活动。",
        "performing_assessment": "正在进行风险评估...",
        "phase1_results": "📋 第1阶段：风险评估结果",
        "work_activity": "工作活动",
        "predicted_hazard": "预测危害",
        "risk_grade_display": "风险等级",
        "t_value_display": "T 值",
        "risk_level_text": "风险等级 : 频率 {freq}, 强度 {intensity}, T {T} (等级 {grade})",
        "similar_cases_section": "🔍 相似案例",
        "case_number": "案例",
        "phase2_results": "🛠️ 第2阶段：改进措施结果",
        "improvement_plan_header": "改进措施",
        "risk_improvement_header": "风险改进结果",
        "comparison_columns": ["项目", "改进前", "改进后"],
        "risk_reduction_label": "风险降低率 (RRR)",
        "risk_visualization": "📊 风险等级变化可视化",
        "before_improvement": "改进前",
        "after_improvement": "改进后",
        "grade_label": "等级",
        "download_results": "💾 下载结果",
        "excel_export": "📥 下载 Excel 报表",
        "col_activity_header": "작업활동 및 내용 Work Sequence",
        "col_hazard_header": "유해위험요인 및 환경측면 영향 Hazarous Factors",
        "col_ehs_header": "EHS",
        "col_risk_likelihood_header": "위험性 Risk – 빈도 likelihood",
        "col_risk_severity_header": "위험性 Risk – 강도 severity",
        "col_control_header": "개선대책 및 세부관리방안 Control Measures",
        "col_incharge_header": "개선담당자 In Charge",
        "col_duedate_header": "개선일자 Correction Due Date",
        "col_after_likelihood_header": "위험性 Risk – 빈도 likelihood",
        "col_after_severity_header": "위험性 Risk – 강도 severity"
    }
}

# ----------------- 페이지 스타일 -----------------
st.set_page_config(page_title="AI Risk Assessment", page_icon="🛠️", layout="wide")
st.markdown(
    """
    <style>
    .main-header{font-size:2.5rem;color:#1E88E5;text-align:center;margin-bottom:1rem}
    .sub-header{font-size:1.8rem;color:#0D47A1;margin-top:2rem;margin-bottom:1rem}
    .similar-case{background-color:#f1f8e9;border-radius:8px;padding:12px;margin-bottom:8px;border-left:4px solid #689f38}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------- 세션 상태 초기화 -----------------
ss = st.session_state
for key, default in {
    "language": "Korean",            # 화면 표시 및 결과 언어
    "index": None,                   # FAISS 인덱스
    "embeddings": None,              # 임베딩 행렬
    "retriever_pool_df": None,       # 유사 사례 후보 데이터프레임 (한국어 원본)
    "last_assessment": None          # 마지막 평가 결과 저장용
}.items():
    if key not in ss:
        ss[key] = default

# ----------------- 언어 선택 -----------------
col0, colLang = st.columns([6, 1])
with colLang:
    lang = st.selectbox(
        "언어 선택",
        list(system_texts.keys()),
        index=list(system_texts.keys()).index(ss.language),
        label_visibility="hidden"
    )
    ss.language = lang
texts = system_texts[ss.language]
result_language = ss.language   # UI 언어와 동일하게 결과 언어로 사용

# ----------------- 헤더 -----------------
st.markdown(f'<div class="main-header">{texts["title"]}</div>', unsafe_allow_html=True)

# ----------------- 탭 구성 -----------------
tabs = st.tabs([texts["tab_overview"], texts["tab_phase"]])

# -----------------------------------------------------------------------------  
# ---------------- Utility Functions ------------------------------------------
# -----------------------------------------------------------------------------  

def determine_grade(value: int) -> str:
    if 16 <= value <= 25:
        return 'A'
    if 10 <= value <= 15:
        return 'B'
    if 5 <= value <= 9:
        return 'C'
    if 3 <= value <= 4:
        return 'D'
    if 1 <= value <= 2:
        return 'E'
    return 'Unknown' if ss.language != 'Korean' else '알 수 없음'

def get_grade_color(grade: str) -> str:
    colors = {
        'A': '#ff1744',
        'B': '#ff9800',
        'C': '#ffc107',
        'D': '#4caf50',
        'E': '#2196f3',
    }
    return colors.get(grade, '#808080')

def compute_rrr(original_t: int, improved_t: int) -> float:
    if original_t == 0:
        return 0.0
    return ((original_t - improved_t) / original_t) * 100

# ─── 신규 추가: 개선대책 번호 기준 줄바꿈 함수 ─────────────────
def format_improvement_plan_for_display(plan_text: str) -> str:
    """
    개선대책(plan_text)이 '1) ... 2) ... 3) ...' 식 번호 매겨져 있을 때,
    각 번호 앞에 줄바꿈을 추가하여 반환합니다.
    """
    if not plan_text:
        return ""

    # (1) 기존 개행문자 모두 제거하고 한 줄로 만든 뒤
    single_line = plan_text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")

    # (2) 숫자) 패턴(예: '2)') 앞에 '\n' 삽입
    formatted = re.sub(r"(?<!\n)(\d\))", r"\n\1", single_line)

    # (3) 첫 글자가 불필요한 \n 인 경우 제거
    if formatted.startswith("\n"):
        formatted = formatted[1:]

    return formatted
# ─────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data(selected_dataset_name: str) -> pd.DataFrame:
    try:
        dataset_mapping = {
            "건축": "건축", "Architecture": "건축",
            "토목": "토목", "Civil": "토목",
            "플랜트": "플랜트", "Plant": "플랜트"
        }
        actual_filename = dataset_mapping.get(selected_dataset_name, selected_dataset_name)

        if os.path.exists(f"{actual_filename}.xlsx"):
            try:
                df = pd.read_excel(f"{actual_filename}.xlsx", engine='openpyxl')
            except Exception:
                df = pd.read_excel(f"{actual_filename}.xlsx", engine='xlrd')
        elif os.path.exists(f"{actual_filename}.xls"):
            df = pd.read_excel(f"{actual_filename}.xls", engine='xlrd')
        else:
            st.info(f"파일을 찾을 수 없습니다: {actual_filename}.xlsx 또는 {actual_filename}.xls")
            st.info("샘플 데이터를 사용합니다.")
            return create_sample_data()

        if "삭제 Del" in df.columns:
            df.drop(["삭제 Del"], axis=1, inplace=True)
        df = df.dropna(how='all')

        column_mapping = {
            "작업활동 및 내용\nWork & Contents": "작업활동 및 내용",
            "유해위험요인 및 환경측면 영향\nHazard & Risk": "유해위험요인 및 환경측면 영향",
            "피해형태 및 환경영향\nDamage & Effect": "피해형태 및 환경영향",
            "개선대책 및 세부관리방안\nCorrective Action": "개선대책"
        }
        df.rename(columns=column_mapping, inplace=True)

        for col in ["빈도", "강도"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if '빈도' not in df.columns:
            df['빈도'] = 3
        if '강도' not in df.columns:
            df['강도'] = 3

        df["T"] = df["빈도"] * df["강도"]
        df["등급"] = df["T"].apply(determine_grade)

        if "개선대책" not in df.columns:
            alt_cols = [c for c in df.columns if "개선" in c or "Corrective" in c]
            if alt_cols:
                df.rename(columns={alt_cols[0]: "개선대책"}, inplace=True)
            else:
                df["개선대책"] = "안전 교육 실시 및 보호구 착용"

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
        final_cols = [col for col in required_cols if col in df.columns]
        df = df[final_cols]

        df = df.fillna({
            "작업활동 및 내용": "일반 작업",
            "유해위험요인 및 환경측면 영향": "일반적 위험",
            "피해형태 및 환경영향": "부상",
            "개선대책": "안전 조치 수행"
        })
        return df

    except Exception as e:
        st.warning(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        st.info("샘플 데이터를 사용합니다.")
        return create_sample_data()

def create_sample_data() -> pd.DataFrame:
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
            "타박상", "골절", "매몰", "추락사", "화상"
        ],
        "빈도": [3, 3, 2, 4, 2],
        "강도": [5, 4, 5, 5, 3],
        "개선대책": [
            "1) 다수의 철골재를 함께 인양하지 않도록 관리\n"
            "2) 치수, 중량, 형상이 다른 재료를 함께 인양하지 않도록 관리",
            "1) 비계대 누락된 목판 설치\n"
            "2) 안전대 부착설비 설치 및 사용\n"
            "3) 비계 변경 시 타공종 외 작업자 작업 금지",
            "1) 적절한 사면 기울기 유지\n"
            "2) 굴착면 보강\n"
            "3) 정기적 지반 상태 점검",
            "1) 안전대 착용 의무화\n"
            "2) 작업 전 안전교육 실시\n"
            "3) 추락방지망 설치",
            "1) 적절한 환기시설 설치\n"
            "2) 화재 예방 조치\n"
            "3) 보호구 착용"
        ]
    }
    df = pd.DataFrame(data)
    df["T"] = df["빈도"] * df["강도"]
    df["등급"] = df["T"].apply(determine_grade)
    return df

def embed_texts_with_openai(texts: list[str], api_key: str, model: str="text-embedding-3-large") -> list[list[float]]:
    if not api_key:
        st.error("API 키가 설정되어 있지 않습니다.")
        return []
    client = OpenAI(api_key=api_key)
    embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        processed = [str(t).replace("\n", " ").strip() for t in batch]
        try:
            resp = client.embeddings.create(
                model=model,
                input=processed
            )
            for item in resp.data:
                embeddings.append(item.embedding)
        except Exception as e:
            st.error(f"임베딩 생성 실패 (배치 {i}): {e}")
            for _ in processed:
                embeddings.append([0.0] * 1536)
    return embeddings

def generate_with_gpt(prompt: str, api_key: str, model: str="gpt-4o", max_retries: int=3) -> str:
    if not api_key:
        st.error("API 키가 설정되어 있지 않습니다.")
        return ""
    client = OpenAI(api_key=api_key)
    sys_prompt = "You are a construction site risk assessment expert. Provide accurate and practical responses in English."
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",  "content": sys_prompt},
                    {"role": "user",    "content": prompt}
                ],
                temperature=0.1,
                max_tokens=700,
                top_p=0.9
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"GPT 호출 오류 ({attempt+1}/{max_retries}): {e}")
                return ""
            else:
                st.warning(f"GPT 재시도 중... ({attempt+1}/{max_retries})")
                continue

def translate_similar_cases(sim_docs: pd.DataFrame, api_key: str) -> pd.DataFrame:
    sim_docs_en = sim_docs.copy().reset_index(drop=True)
    sim_docs_en["activity_en"] = sim_docs_en["작업활동 및 내용"]
    sim_docs_en["hazard_en"] = sim_docs_en["유해위험요인 및 환경측면 영향"]
    sim_docs_en["plan_en"] = sim_docs_en["개선대책"]
    for idx, row in sim_docs_en.iterrows():
        act_ko = row["작업활동 및 내용"]
        prompt_act = (
            "Translate the following construction work activity into English. "
            "Only provide the translation:\n\n" + act_ko
        )
        act_en = generate_with_gpt(prompt_act, api_key)
        if act_en:
            sim_docs_en.at[idx, "activity_en"] = act_en

        haz_ko = row["유해위험요인 및 환경측면 영향"]
        prompt_haz = (
            "Translate the following construction hazard into English. "
            "Only provide the translation:\n\n" + haz_ko
        )
        haz_en = generate_with_gpt(prompt_haz, api_key)
        if haz_en:
            sim_docs_en.at[idx, "hazard_en"] = haz_en

        plan_ko = row["개선대책"]
        prompt_plan = (
            "Translate the following safety improvement measures into English. "
            "Keep the numbered format. Only provide the translation:\n\n" + plan_ko
        )
        plan_en = generate_with_gpt(prompt_plan, api_key)
        if plan_en:
            sim_docs_en.at[idx, "plan_en"] = plan_en

    return sim_docs_en

def translate_output(content: str, target_language: str, api_key: str, max_retries: int=2) -> str:
    if target_language == "English" or not api_key:
        return content
    for attempt in range(max_retries):
        try:
            prompt = f"Translate the following into {target_language}. Only provide the translation:\n\n{content}"
            translated = generate_with_gpt(prompt, api_key)
            if translated:
                return translated
        except Exception:
            if attempt == max_retries - 1:
                return content
            else:
                continue
    return content

def construct_prompt_phase1_hazard(sim_docs_en: pd.DataFrame, activity_en: str) -> str:
    intro = "Below are examples of work activities and associated hazards at construction sites:\n\n"
    example_fmt = "Example {i}:\n- Work Activity: {act}\n- Hazard: {haz}\n\n"
    query_fmt = (
        "Based on the above examples, predict the main hazards for the following work activity:\n\n"
        f"Work Activity: {activity_en}\n\nPredicted Hazard: "
    )
    prompt = intro
    for i, (_, row) in enumerate(sim_docs_en.head(10).iterrows(), start=1):
        act = row["activity_en"]
        haz = row["hazard_en"]
        if pd.notna(act) and pd.notna(haz):
            prompt += example_fmt.format(i=i, act=act, haz=haz)
    prompt += query_fmt
    return prompt

def construct_prompt_phase1_risk(sim_docs_en: pd.DataFrame, activity_en: str, hazard_en: str) -> str:
    intro = (
        "Construction site risk assessment criteria:\n"
        "- Frequency(1-5): 1=Very Rare, 2=Rare, 3=Occasional, 4=Frequent, 5=Very Frequent\n"
        "- Intensity(1-5): 1=Minor Injury, 2=Light Injury, 3=Moderate Injury, 4=Serious Injury, 5=Fatality\n"
        "- T-value = Frequency × Intensity\n\n"
        "Reference Cases:\n\n"
    )
    example_fmt = "Case {i}:\nInput: {inp}\nAssessment: Frequency={freq}, Intensity={intensity}, T-value={t}\n\n"
    json_format = '{"frequency": number, "intensity": number, "T": number}'
    query_fmt = (
        "Based on the above criteria and cases, assess the following:\n\n"
        f"Work Activity: {activity_en}\n"
        f"Hazard: {hazard_en}\n\n"
        f"Respond exactly in this JSON format:\n{json_format}"
    )
    prompt = intro
    count = 0
    for _, row in sim_docs_en.head(10).iterrows():
        try:
            inp = f"{row['activity_en']} - {row['hazard_en']}"
            freq = int(row["빈도"])
            intensity = int(row["강도"])
            t_val = freq * intensity
            count += 1
            prompt += example_fmt.format(i=count, inp=inp, freq=freq, intensity=intensity, t=t_val)
            if count >= 3:
                break
        except Exception:
            continue
    prompt += query_fmt
    return prompt

def parse_gpt_output_phase1(gpt_output: str) -> tuple[int, int, int]:
    pattern = r'\{"frequency":\s*([1-5]),\s*"intensity":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    match = re.search(pattern, gpt_output)
    if match:
        freq = int(match.group(1))
        intensity = int(match.group(2))
        t_val = int(match.group(3))
        return freq, intensity, t_val
    nums = re.findall(r'\b([1-5])\b', gpt_output)
    if len(nums) >= 2:
        freq = int(nums[0])
        intensity = int(nums[1])
        return freq, intensity, freq * intensity
    return None

def construct_prompt_phase2(sim_docs_en: pd.DataFrame, activity_en: str, hazard_en: str,
                             freq: int, intensity: int, t_val: int, api_key: str) -> str:
    example_section = ""
    count = 0
    for _, row in sim_docs_en.head(10).iterrows():
        try:
            plan_en = row["plan_en"]
            orig_freq = int(row["빈도"])
            orig_intensity = int(row["강도"])
            orig_t = orig_freq * orig_intensity
            new_freq = max(1, orig_freq - 1)
            new_intensity = max(1, orig_intensity - 1)
            new_t = new_freq * new_intensity

            count += 1
            example_section += (
                f"Example {count}:\n"
                f"Input Work Activity: {row['activity_en']}\n"
                f"Input Hazard: {row['hazard_en']}\n"
                f"Original Frequency: {orig_freq}\n"
                f"Original Intensity: {orig_intensity}\n"
                f"Original T-value: {orig_t}\n"
                "Output (Improvement Plan and Risk Reduction) in JSON:\n"
                "{\n"
                f'  "improvement_plan": "{plan_en}",\n'
                f'  "improved_frequency": {new_freq},\n'
                f'  "improved_intensity": {new_intensity},\n'
                f'  "improved_T": {new_t},\n'
                f'  "reduction_rate": {compute_rrr(orig_t, new_t):.2f}\n'
                "}\n\n"
            )
            if count >= 2:
                break
        except Exception:
            continue

    if count == 0:
        example_section = (
            "Example 1:\n"
            "Input Work Activity: Excavation and backfilling\n"
            "Input Hazard: Collapse of excavation wall\n"
            "Original Frequency: 3\n"
            "Original Intensity: 4\n"
            "Original T-value: 12\n"
            "Output (Improvement Plan and Risk Reduction) in JSON:\n"
            "{\n"
            '  "improvement_plan": "1) Maintain proper slope according to soil classification\\n'
            '2) Reinforce excavation walls\\n3) Conduct regular ground condition inspections",\n'
            '  "improved_frequency": 1,\n'
            '  "improved_intensity": 2,\n'
            '  "improved_T": 2,\n'
            '  "reduction_rate": 83.33\n'
            "}\n\n"
        )

    prompt = (
        example_section +
        "Now here is a new input:\n"
        f"Work Activity: {activity_en}\n"
        f"Hazard: {hazard_en}\n"
        f"Original Frequency: {freq}\n"
        f"Original Intensity: {intensity}\n"
        f"Original T-value: {t_val}\n\n"
        "Please provide practical and specific improvement measures in the following JSON format:\n"
        "{\n"
        '  "improvement_plan": "numbered list of specific measures",\n'
        '  "improved_frequency": (integer 1-5),\n'
        '  "improved_intensity": (integer 1-5),\n'
        '  "improved_T": (improved_frequency × improved_intensity),\n'
        '  "reduction_rate": (percentage)\n'
        "}\n\n"
        "Improvement measures should include at least 3 field-applicable methods."
    )
    return prompt

def parse_gpt_output_phase2(gpt_output: str) -> dict:
    try:
        json_match = re.search(r'\{.*\}', gpt_output, re.DOTALL)
        if not json_match:
            raise ValueError("JSON match not found")
        import json
        json_str = json_match.group(0)
        parsed = json.loads(json_str)
        if "improvement_plan" in parsed:
            return {
                "improvement_plan": parsed.get("improvement_plan", ""),
                "improved_freq": parsed.get("improved_frequency", 1),
                "improved_intensity": parsed.get("improved_intensity", 1),
                "improved_T": parsed.get("improved_T", parsed.get("improved_frequency", 1) * parsed.get("improved_intensity", 1)),
                "reduction_rate": parsed.get("reduction_rate", 0.0)
            }
        chinese_keys = {
            "improvement": ["改进措施", "改进计划"],
            "improved_freq": ["改进后频率", "新频率"],
            "improved_intensity": ["改进后强度", "新强度"],
            "improved_T": ["改进后T值", "新T值"],
            "reduction_rate": ["T值降低率", "降低率"]
        }
        if any(key in parsed for key in chinese_keys["improvement"]):
            im_plan = ""
            for k in chinese_keys["improvement"]:
                if k in parsed:
                    im_plan = parsed[k]
                    break
            imp_freq = 1
            for k in chinese_keys["improved_freq"]:
                if k in parsed:
                    imp_freq = int(parsed[k])
                    break
            imp_int = 1
            for k in chinese_keys["improved_intensity"]:
                if k in parsed:
                    imp_int = int(parsed[k])
                    break
            imp_t = imp_freq * imp_int
            for k in chinese_keys["improved_T"]:
                if k in parsed:
                    imp_t = int(parsed[k])
                    break
            r_rate = 0.0
            for k in chinese_keys["reduction_rate"]:
                if k in parsed:
                    try:
                        r_rate = float(parsed[k])
                    except:
                        r_rate = 0.0
                    break
            return {
                "improvement_plan": im_plan,
                "improved_freq": imp_freq,
                "improved_intensity": imp_int,
                "improved_T": imp_t,
                "reduction_rate": r_rate
            }
        raise ValueError("No recognized keys found in JSON")
    except Exception:
        plan = ""
        m_plan_en = re.search(r'"improvement_plan"\s*:\s*"(?P<plan>.*?)"', gpt_output, re.DOTALL)
        if m_plan_en:
            raw = m_plan_en.group("plan")
            plan = raw.replace('\n', '\\n').strip()
        else:
            m_plan_cn = re.search(r'"(改进措施|改进计划)"\s*:\s*"(?P<plan>.*?)"', gpt_output, re.DOTALL)
            if m_plan_cn:
                raw = m_plan_cn.group("plan")
                plan = raw.replace('\n', '\\n').strip()
            else:
                plan = (
                    "1) Educate workers and mandate PPE usage\n"
                    "2) Install pedestrian walkways\n"
                    "3) Provide high-visibility vests"
                )
        def extract_int(keys: list[str]) -> int:
            for key in keys:
                m = re.search(rf'"{key}"\s*:\s*(\d+)', gpt_output)
                if m:
                    return int(m.group(1))
            return 1
        def extract_float(keys: list[str]) -> float:
            for key in keys:
                m = re.search(rf'"{key}"\s*:\s*([\d\.]+)', gpt_output)
                if m:
                    try:
                        return float(m.group(1))
                    except:
                        return 0.0
            return 0.0
        english_keys = {
            "improved_freq": ["improved_frequency"],
            "improved_intensity": ["improved_intensity"],
            "improved_T": ["improved_T"],
            "reduction_rate": ["reduction_rate"]
        }
        chinese_keys = {
            "improved_freq": ["改进后频率", "新频率"],
            "improved_intensity": ["改进后强度", "新强度"],
            "improved_T": ["改进后T值", "新T值"],
            "reduction_rate": ["T值降低率", "降低率"]
        }
        imp_freq = extract_int(english_keys["improved_freq"] + chinese_keys["improved_freq"])
        imp_int = extract_int(english_keys["improved_intensity"] + chinese_keys["improved_intensity"])
        imp_t   = extract_int(english_keys["improved_T"] + chinese_keys["improved_T"])
        r_rate  = extract_float(english_keys["reduction_rate"] + chinese_keys["reduction_rate"])
        return {
            "improvement_plan": plan,
            "improved_freq": imp_freq,
            "improved_intensity": imp_int,
            "improved_T": imp_t,
            "reduction_rate": r_rate
        }

def create_excel_download(result_dict: dict, similar_records: list[dict]) -> bytes:
    output = io.BytesIO()
    try:
        # 현재 날짜 가져오기
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            workbook = writer.book

def create_excel_download(result_dict: dict, similar_records: list[dict]) -> bytes:
    output = io.BytesIO()
    try:
        # 현재 날짜 가져오기
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            workbook = writer.book

            # ─── 메인 결과를 데이터셋 형식에 맞춰 단일 시트로 생성 ─────────────────
            # 원본 데이터셋의 정확한 컬럼 구조와 동일하게 생성
            main_result_df = pd.DataFrame({
                "작업활동 및 내용\nWork & Contents": [result_dict["activity"]],
                "유해위험요인 및 환경측면 영향\nHazard & Risk": [result_dict["hazard"]],
                "EHS": ["S"],  # EHS는 "S"로 설정
                "피해형태 및 환경영향\nDamage & Effect": [""],  # 빈 값으로 설정
                "빈도\nRisk Rate": [result_dict["freq"]],  # 첫 번째 위험성 Risk Rate 하위
                "강도\nSeverity": [result_dict["intensity"]],
                "T": [result_dict["T"]],
                "등급\nRate": [result_dict["grade"]],
                "개선대책 및 세부관리방안\nCorrective Action": [result_dict["improvement_plan"]],
                "개선담당자\nResponsibility": [""],  # 개선담당자는 빈칸
                "개선일자\nAction Date": [current_date],  # 개선일자는 현재 날짜
                "빈도\nLikelihood": [result_dict["improved_freq"]],  # 두 번째 위험성 Risk Rate 하위
                "강도\nSeverity ": [result_dict["improved_intensity"]],  # 공백으로 구분
                "T ": [result_dict["improved_T"]],  # 공백으로 구분
                "등급\nRate ": [determine_grade(result_dict["improved_T"])]  # 공백으로 구분
            })
            
            main_result_df.to_excel(writer, sheet_name="위험성평가결과", index=False)
            ws_main = writer.sheets["위험성평가결과"]
            
            # 컬럼 너비 자동 조정
            for col_idx, column in enumerate(main_result_df.columns):
                max_length = max(
                    len(str(column)),
                    main_result_df[column].astype(str).str.len().max() if not main_result_df[column].empty else 0
                )
                ws_main.set_column(col_idx, col_idx, min(max_length + 2, 50))

            # ─── 유사사례 시트 (기존 데이터셋 형식) ─────────────────
            if similar_records:
                sim_df = pd.DataFrame(similar_records)
                sim_df["개선 후 빈도"] = sim_df["빈도"].astype(int).apply(lambda x: max(1, x - 1))
                sim_df["개선 후 강도"] = sim_df["강도"].astype(int).apply(lambda x: max(1, x - 1))
                sim_df["개선 후 T"] = sim_df["개선 후 빈도"] * sim_df["개선 후 강도"]
                sim_df["개선 후 등급"] = sim_df["개선 후 T"].apply(determine_grade)

                export_df = pd.DataFrame({
                    "작업활동 및 내용\nWork & Contents": sim_df["작업활동"],
                    "유해위험요인 및 환경측면 영향\nHazard & Risk": sim_df["유해위험요인"],
                    "EHS": ["S" for _ in range(len(sim_df))],  # EHS는 "S"로 설정
                    "피해형태 및 환경영향\nDamage & Effect": ["" for _ in range(len(sim_df))],
                    "빈도\nRisk Rate": sim_df["빈도"],
                    "강도\nSeverity": sim_df["강도"],
                    "T": sim_df["T"],
                    "등급\nRate": sim_df["등급"],
                    "개선대책 및 세부관리방안\nCorrective Action": sim_df["개선대책"],
                    "개선담당자\nResponsibility": ["" for _ in range(len(sim_df))],  # 개선담당자는 빈칸
                    "개선일자\nAction Date": [current_date for _ in range(len(sim_df))],  # 개선일자는 현재 날짜
                    "빈도\nLikelihood": sim_df["개선 후 빈도"],
                    "강도\nSeverity ": sim_df["개선 후 강도"],  # 공백으로 구분
                    "T ": sim_df["개선 후 T"],
                    "등급\nRate ": sim_df["개선 후 등급"]
                })
                export_df.to_excel(writer, sheet_name="유사사례", index=False)
                ws_sim = writer.sheets["유사사례"]
                
                # 유사사례 시트 컬럼 너비 조정
                for col_idx, column in enumerate(export_df.columns):
                    max_length = max(
                        len(str(column)),
                        export_df[column].astype(str).str.len().max() if not export_df[column].empty else 0
                    )
                    ws_sim.set_column(col_idx, col_idx, min(max_length + 2, 50))

        return output.getvalue()
        
    except ImportError:
        st.warning("Excel 다운로드를 위한 라이브러리가 없습니다. CSV로 다운로드합니다.")
        # CSV 백업도 동일한 형식으로 생성
        from datetime import datetime
        csv_buffer = io.StringIO()
        
        # 메인 결과를 CSV로 생성
        main_result_df = pd.DataFrame({
            "작업활동 및 내용\nWork & Contents": [result_dict["activity"]],
            "유해위험요인 및 환경측면 영향\nHazard & Risk": [result_dict["hazard"]],
            "EHS": ["S"],  # EHS는 "S"로 설정
            "피해형태 및 환경영향\nDamage & Effect": [""],
            "빈도\nRisk Rate": [result_dict["freq"]],
            "강도\nSeverity": [result_dict["intensity"]],
            "T": [result_dict["T"]],
            "등급\nRate": [result_dict["grade"]],
            "개선대책 및 세부관리방안\nCorrective Action": [result_dict["improvement_plan"]],
            "개선담당자\nResponsibility": [""],  # 개선담당자는 빈칸
            "개선일자\nAction Date": [datetime.now().strftime("%Y-%m-%d")],  # 개선일자는 현재 날짜
            "빈도\nLikelihood": [result_dict["improved_freq"]],
            "강도\nSeverity ": [result_dict["improved_intensity"]],  # 공백으로 구분
            "T ": [result_dict["improved_T"]],
            "등급\nRate ": [determine_grade(result_dict["improved_T"])]
        })
        
        main_result_df.to_csv(csv_buffer, index=False, encoding="utf-8-sig")
        return csv_buffer.getvalue().encode("utf-8-sig")
# -----------------------------------------------------------------------------  
# ---------------------- Overview 탭 ------------------------------------------  
# -----------------------------------------------------------------------------  
with tabs[0]:
    st.markdown(f'<div class="sub-header">{texts["overview_header"]}</div>', unsafe_allow_html=True)
    col_overview, col_features = st.columns([3, 2])
    with col_overview:
        st.markdown(texts["overview_text"])
        st.markdown(f"**{texts['features_title']}**")
        st.markdown(texts["phase_features"])
    with col_features:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(
                label=texts["supported_languages_label"],
                value=texts["supported_languages_value"],
                delta=texts["supported_languages_detail"]
            )
        with col2:
            st.metric(
                label=texts["assessment_phases_label"],
                value=texts["assessment_phases_value"],
                delta=texts["assessment_phases_detail"]
            )
        with col3:
            st.metric(
                label=texts["risk_grades_label"],
                value=texts["risk_grades_value"],
                delta=texts["risk_grades_detail"]
            )

# -----------------------------------------------------------------------------  
# -------------------- Risk Assessment & Improvement 탭 ------------------------
# -----------------------------------------------------------------------------  
with tabs[1]:
    st.markdown(f'<div class="sub-header">{texts["tab_phase"]}</div>', unsafe_allow_html=True)

    col_api, col_dataset = st.columns([2, 1])
    with col_api:
        api_key = st.text_input(texts["api_key_label"], type="password", key="api_key_all")
    with col_dataset:
        dataset_options = ["건축", "토목", "플랜트"]
        dataset_name = st.selectbox(
            texts["dataset_label"],
            dataset_options,
            key="dataset_all"
        )

    if ss.retriever_pool_df is None or st.button(texts["load_data_btn"], type="primary"):
        if not api_key:
            st.warning(texts["api_key_warning"])
        else:
            with st.spinner(texts["data_loading"]):
                try:
                    df = load_data(dataset_name)
                    if len(df) > 10:
                        train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
                    else:
                        train_df = df.copy()

                    pool_df = train_df.copy()
                    pool_df["content"] = pool_df.apply(lambda r: " ".join(r.values.astype(str)), axis=1)

                    to_embed = pool_df["content"].tolist()
                    max_texts = min(len(to_embed), 30)
                    st.info(texts["demo_limit_info"].format(max_texts=max_texts))

                    embeds = embed_texts_with_openai(to_embed[:max_texts], api_key=api_key)
                    vecs = np.array(embeds, dtype="float32")
                    dim = vecs.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(vecs)

                    ss.index = index
                    ss.embeddings = vecs
                    ss.retriever_pool_df = pool_df.iloc[:max_texts]
                    st.success(texts["data_load_success"].format(max_texts=max_texts))
                    with st.expander("📊 로드된 데이터 미리보기"):
                        st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"데이터 로딩 중 오류: {e}")

    st.divider()
    st.markdown(f"### {texts['performing_assessment'].split('.')[0]}")

    activity = st.text_area(
        texts["activity_label"],
        placeholder={
            "Korean": "예: 임시 현장 저장소에서 포크리프트를 이용한 철골 구조재 하역작업",
            "English": "e.g.: Unloading steel structural materials using forklift at temporary site storage",
            "Chinese": "例: 在临时现场仓库使用叉车卸载钢结构材料"
        }.get(ss.language),
        height=100,
        key="user_activity"
    )
    include_similar_cases = st.checkbox(texts["include_similar_cases"], value=True)
    run_button = st.button(texts["run_assessment"], type="primary", use_container_width=True)

    if run_button:
        if not activity:
            st.warning(texts["activity_warning"])
        elif not api_key:
            st.warning(texts["api_key_warning"])
        elif ss.index is None:
            st.warning(texts["load_first_warning"])
        else:
            with st.spinner(texts["performing_assessment"]):
                try:
                    # ===== Phase 1 =====
                    prompt_to_english = (
                        "Translate the following construction work activity into English. "
                        "Only provide the translation:\n\n" + activity
                    )
                    activity_en = generate_with_gpt(prompt_to_english, api_key)
                    if not activity_en:
                        activity_en = activity

                    sim_docs = ss.retriever_pool_df.copy().reset_index(drop=True)
                    sim_docs_en = translate_similar_cases(sim_docs, api_key)

                    q_emb_list = embed_texts_with_openai([activity_en], api_key=api_key)
                    if not q_emb_list:
                        st.error("위험성 평가를 파싱할 수 없습니다.")
                        st.stop()
                    q_emb = q_emb_list[0]

                    D, I = ss.index.search(np.array([q_emb], dtype="float32"), k=min(10, len(sim_docs_en)))
                    if I is None or len(I[0]) == 0:
                        st.error("유사한 사례를 찾을 수 없습니다.")
                        st.stop()

                    sim_docs_subset = sim_docs_en.iloc[I[0]].reset_index(drop=True)

                    hazard_prompt_en = construct_prompt_phase1_hazard(sim_docs_subset, activity_en)
                    hazard_en = generate_with_gpt(hazard_prompt_en, api_key)
                    if not hazard_en:
                        st.error("위험성 평가를 파싱할 수 없습니다.")
                        st.stop()

                    risk_prompt_en = construct_prompt_phase1_risk(sim_docs_subset, activity_en, hazard_en)
                    risk_json_en = generate_with_gpt(risk_prompt_en, api_key)
                    parse_result = parse_gpt_output_phase1(risk_json_en)
                    if not parse_result:
                        st.error("위험성 평가를 파싱할 수 없습니다.")
                        st.expander("GPT 원본 응답").write(risk_json_en)
                        st.stop()

                    freq, intensity, T_val = parse_result
                    grade = determine_grade(T_val)

                    # ===== Phase 2 =====
                    prompt_phase2_en = construct_prompt_phase2(
                        sim_docs_subset, activity_en, hazard_en, freq, intensity, T_val, api_key
                    )
                    improvement_json_en = generate_with_gpt(prompt_phase2_en, api_key)
                    parsed_improvement = parse_gpt_output_phase2(improvement_json_en)
                    improvement_plan_en = parsed_improvement.get("improvement_plan", "")
                    improved_freq = parsed_improvement.get("improved_freq", 1)
                    improved_intensity = parsed_improvement.get("improved_intensity", 1)
                    improved_T = parsed_improvement.get("improved_T", improved_freq * improved_intensity)
                    rrr_value = compute_rrr(T_val, improved_T)

                    # ===== 최종 출력용 번역 =====
                    hazard_user = translate_output(hazard_en, result_language, api_key)
                    improvement_user = translate_output(improvement_plan_en, result_language, api_key)

                    # ===== 유사 사례 출력용 데이터 생성 =====
                    display_sim_records = []
                    for idx, row in sim_docs_subset.iterrows():
                        eng_act = row["activity_en"]
                        eng_haz = row["hazard_en"]
                        eng_plan = row["plan_en"]
                        orig_freq = row["빈도"]
                        orig_intensity = row["강도"]
                        orig_T = row["T"]
                        orig_grade = row["등급"]

                        if result_language == "English":
                            act_disp = eng_act
                            haz_disp = eng_haz
                            plan_disp = eng_plan
                        elif result_language == "Chinese":
                            act_disp = translate_output(eng_act, "Chinese", api_key)
                            haz_disp = translate_output(eng_haz, "Chinese", api_key)
                            plan_disp = translate_output(eng_plan, "Chinese", api_key)
                        else:  # Korean
                            act_disp = translate_output(eng_act, "Korean", api_key)
                            haz_disp = translate_output(eng_haz, "Korean", api_key)
                            plan_disp = translate_output(eng_plan, "Korean", api_key)

                        display_sim_records.append({
                            "작업활동": act_disp,
                            "유해위험요인": haz_disp,
                            "빈도": orig_freq,
                            "강도": orig_intensity,
                            "T": orig_T,
                            "등급": orig_grade,
                            "개선대책": plan_disp
                        })

                    # ===== 화면 출력 =====
                    st.markdown(f"## {texts['phase1_results']}")
                    col_r1, col_r2 = st.columns([2, 1])
                    with col_r1:
                        activity_user = (
                            translate_output(activity_en, result_language, api_key)
                            if result_language != "English"
                            else activity_en
                        )
                        st.markdown(f"**{texts['work_activity']}:** {activity_user}")
                        st.markdown(f"**{texts['predicted_hazard']}:** {hazard_user}")

                        df_display = pd.DataFrame({
                            texts["comparison_columns"][0]: ["빈도", "강도", "T 값", "위험등급"],
                            "값": [str(freq), str(intensity), str(T_val), grade]
                        })
                        st.dataframe(df_display.astype(str), use_container_width=True, hide_index=True)
                    with col_r2:
                        grade_color = get_grade_color(grade)
                        st.markdown(f"""
                        <div style="text-align:center; padding:20px; background-color:{grade_color};
                                    color:white; border-radius:10px; margin:10px 0;">
                            <h2 style="margin:0;">{texts['risk_grade_display']}</h2>
                            <h1 style="margin:10px 0; font-size:3rem;">{grade}</h1>
                            <p style="margin:0;">{texts['t_value_display']}: {T_val}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== 유사사례 표시 =====
                    if include_similar_cases and display_sim_records:
                        st.markdown(f"### {texts['similar_cases_section']}")
                        for idx, rec in enumerate(display_sim_records):
                            with st.expander(f"{texts['case_number']} {idx+1}: {rec['작업활동'][:30]}…"):
                                c1, c2 = st.columns(2)
                                with c1:
                                    st.write(f"**{texts['work_activity']} :** {rec['작업활동']}")
                                    st.write(f"**{texts['predicted_hazard']} :** {rec['유해위험요인']}")
                                    st.write(
                                        f"**{texts['risk_level_text'].format(freq=rec['빈도'], intensity=rec['강도'], T=rec['T'], grade=rec['등급'])}**"
                                    )
                                with c2:
                                    st.write(f"**{texts['improvement_plan_header']} :**")
                                    raw_plan = rec["개선대책"]
                                    formatted_plan = format_improvement_plan_for_display(raw_plan)
                                    st.markdown(formatted_plan)

                    # ===== Phase 2 출력 =====
                    st.markdown(f"## {texts['phase2_results']}")
                    c_imp, c_riskimp = st.columns([3, 2])
                    with c_imp:
                        st.markdown(f"### {texts['improvement_plan_header']}")
                        if improvement_user:
                            formatted_plan2 = format_improvement_plan_for_display(improvement_user)
                            st.markdown(formatted_plan2)
                        else:
                            st.write("개선대책 생성 실패")

                    with c_riskimp:
                        st.markdown(f"### {texts['risk_improvement_header']}")
                        comp_df_user = pd.DataFrame({
                            texts["comparison_columns"][0]: ["빈도", "강도", "T 값", "위험등급"],
                            texts["comparison_columns"][1]: [str(freq), str(intensity), str(T_val), grade],
                            texts["comparison_columns"][2]: [str(improved_freq), str(improved_intensity), str(improved_T), determine_grade(improved_T)]
                        })
                        st.dataframe(comp_df_user.astype(str), use_container_width=True, hide_index=True)
                        st.metric(
                            label=texts["risk_reduction_label"],
                            value=f"{rrr_value:.1f}%",
                            delta=f"-{T_val - improved_T} T"
                        )

                    # ===== 위험도 시각화 =====
                    st.markdown(f"### {texts['risk_visualization']}")
                    vis1, vis2 = st.columns(2)
                    with vis1:
                        st.markdown(f"**{texts['before_improvement']}**")
                        col_before = get_grade_color(grade)
                        st.markdown(f"""
                        <div style="background-color:{col_before}; color:white; padding:15px; 
                                    border-radius:10px; text-align:center; margin:10px 0;">
                            <h3 style="margin:0;">{texts['grade_label']} {grade}</h3>
                            <p style="margin:5px 0; font-size:1.2em;">{texts['t_value_display']}: {T_val}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with vis2:
                        st.markdown(f"**{texts['after_improvement']}**")
                        grade_after = determine_grade(improved_T)
                        col_after = get_grade_color(grade_after)
                        st.markdown(f"""
                        <div style="background-color:{col_after}; color:white; padding:15px; 
                                    border-radius:10px; text-align:center; margin:10px 0;">
                            <h3 style="margin:0;">{texts['grade_label']} {grade_after}</h3>
                            <p style="margin:5px 0; font-size:1.2em;">{texts['t_value_display']}: {improved_T}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ===== 세션 저장 =====
                    ss.last_assessment = {
                        "activity": activity_user,
                        "hazard": hazard_user,
                        "freq": freq,
                        "intensity": intensity,
                        "T": T_val,
                        "grade": grade,
                        "improvement_plan": improvement_user,
                        "improved_freq": improved_freq,
                        "improved_intensity": improved_intensity,
                        "improved_T": improved_T,
                        "rrr": rrr_value,
                        "similar_cases": display_sim_records
                    }

                    # ===== 엑셀 다운로드 =====
                    st.markdown(f"### {texts['download_results']}")
                    excel_bytes = create_excel_download(ss.last_assessment, display_sim_records)
                    st.download_button(
                        label=texts["excel_export"],
                        data=excel_bytes,
                        file_name="risk_assessment_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                except Exception as e:
                    st.error(f"🚨 예상치 못한 오류가 발생했습니다:\n{e}")
                    st.stop()

    # 푸터
    st.markdown('<hr style="margin-top: 3rem;">', unsafe_allow_html=True)
    footer_c1, footer_c2, footer_c3 = st.columns([1, 1, 1])
    with footer_c1:
        if os.path.exists("cau.png"):
            st.image("cau.png", width=140)
    with footer_c2:
        st.markdown(
            """
            <div style="text-align: center; padding: 20px;">
                <h4>두산에너빌리티</h4>
                <p>AI 기반 위험성 평가 시스템</p>
                <p style="font-size: 0.8rem; color: #666;">
                    © 2025 Doosan Enerbility. All rights reserved.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with footer_c3:
        if os.path.exists("doosan.png"):
            st.image("doosan.png", width=160)
