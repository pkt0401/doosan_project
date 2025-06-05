import streamlit as st
import pandas as pd
import numpy as np
import faiss
import re
import os
import io
from PIL import Image
from sklearn.model_selection import train_test_split
import openai

# -------------------------------------------------
# OpenAI 공식 엔드포인트 사용
# -------------------------------------------------

# ------------- 시스템 텍스트 -----------------
# 내부 처리는 모두 영어로 이루어지고, 결과는 컬럼 이름을 한/영 병기 형태로 출력하도록 구성합니다.
system_texts = {
    "Korean": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "시스템 개요",
        "tab_assessment": "위험성 평가 & 개선대책",
        "overview_header": "LLM 기반 위험성평가 시스템",
        "overview_text": (
            "두산에너빌리티 AI Risk Assessment는 국내 및 해외 건설현장 '수시위험성평가' "
            "및 '노동부 중대재해 사례'를 학습하여 개발된 자동 위험성평가 프로그램입니다. "
            "생성된 결과는 검증 후 사용하시기 바랍니다."
        ),
        "features_title": "시스템 특징",
        "phase1_features": (
            "- 작업활동을 영어로 변환하여 LLM에 입력\n"
            "- English 내부 처리 후, 결과를 한/영 병기 컬럼으로 출력\n"
            "- 유사 사례 검색 및 표시\n"
            "- 위험도 계산 및 개선대책 생성"
        ),
        "api_key_label": "OpenAI API 키를 입력하세요:",
        "dataset_label": "데이터셋 선택:",
        "load_data_btn": "데이터 로드 및 인덱스 구성",
        "api_key_warning": "API 키를 입력해야 합니다.",
        "data_loading": "데이터를 불러오고 인덱스를 구성하는 중...",
        "demo_limit_info": "데모용으로 {max_texts}개 항목만 임베딩합니다.",
        "data_load_success": "데이터 로드 및 인덱스 구성 완료! ({max_texts}개 항목 처리됨)",
        "activity_label": "작업활동 (한국어 또는 영어):",
        "include_similar_label": "유사 사례 포함",
        "result_language_label": "출력 언어:",
        "run_button": "위험성 평가 실행",
        "no_activity_warning": "작업활동을 입력하세요.",
        "no_index_warning": "먼저 데이터를 로드하고 인덱스를 구성하세요.",
        "parsing_error": "위험성 평가 결과를 파싱할 수 없습니다.",
        "parsing_error_improvement": "개선대책 결과를 파싱할 수 없습니다.",
        "download_excel": "📥 엑셀 다운로드"
    },
    "English": {
        "title": "Artificial Intelligence Risk Assessment",
        "tab_overview": "System Overview",
        "tab_assessment": "Assessment & Improvement",
        "overview_header": "LLM-based Risk Assessment System",
        "overview_text": (
            "Doosan Enerbility AI Risk Assessment is an automated program trained on "
            "on-demand risk-assessment reports and major-accident cases. "
            "Please review and validate all generated outputs before use."
        ),
        "features_title": "Features",
        "phase1_features": (
            "- Convert work activity to English before LLM input\n"
            "- Internally process in English, then output bilingual columns\n"
            "- Retrieve and display similar cases\n"
            "- Compute risk and generate improvement measures"
        ),
        "api_key_label": "Enter OpenAI API Key:",
        "dataset_label": "Select Dataset:",
        "load_data_btn": "Load Data & Build Index",
        "api_key_warning": "Please enter an API key.",
        "data_loading": "Loading data and building index...",
        "demo_limit_info": "Embedding only {max_texts} items for demo.",
        "data_load_success": "Data loaded & index built! ({max_texts} items processed)",
        "activity_label": "Work Activity (Korean or English):",
        "include_similar_label": "Include Similar Cases",
        "result_language_label": "Output Language:",
        "run_button": "Run Assessment",
        "no_activity_warning": "Please enter a work activity.",
        "no_index_warning": "Load data and build index first.",
        "parsing_error": "Cannot parse risk assessment output.",
        "parsing_error_improvement": "Cannot parse improvement measures output.",
        "download_excel": "📥 Download Excel"
    }
}
# -----------------------------------------------------------------------------  
# ------------------------ 페이지 설정 및 스타일 -----------------------------  
# -----------------------------------------------------------------------------  
st.set_page_config(page_title="AI Risk Assessment", page_icon="🛠️", layout="wide")
ss = st.session_state

for key, default in {
    "language": "Korean",
    "index": None,
    "embeddings": None,
    "retriever_pool_df": None,
    "last_assessment": None
}.items():
    if key not in ss:
        ss[key] = default

# 언어 선택
col0, colLang = st.columns([6, 1])
with colLang:
    lang = st.selectbox(
        "언어 선택",
        ["Korean", "English"],
        index=["Korean", "English"].index(ss.language),
        label_visibility="hidden"
    )
    ss.language = lang
texts = system_texts[ss.language]

# 헤더
st.markdown(f"<h1 style='text-align:center; color:#1E88E5;'>{texts['title']}</h1>", unsafe_allow_html=True)

# 탭 생성
tab1, tab2 = st.tabs([texts["tab_overview"], texts["tab_assessment"]])

# -----------------------------------------------------------------------------  
# ----------------------------- Overview 탭 -----------------------------------  
# -----------------------------------------------------------------------------  
with tab1:
    st.markdown(f"## {texts['overview_header']}")
    st.markdown(texts["overview_text"])
    st.markdown(f"### {texts['features_title']}")
    st.markdown(texts["phase1_features"])

# -----------------------------------------------------------------------------  
# --------------------------- Assessment 탭 -----------------------------------  
# -----------------------------------------------------------------------------  
with tab2:
    st.markdown(f"## {texts['tab_assessment']}")

    col_api, col_dataset = st.columns([2, 1])
    with col_api:
        api_key = st.text_input(texts["api_key_label"], type="password", key="api_key_all")
    with col_dataset:
        dataset_name = st.selectbox(
            texts["dataset_label"],
            ["건축", "토목", "플랜트"],
            key="dataset_select"
        )

    # 데이터 로드 및 인덱스 구성
    if ss.retriever_pool_df is None or st.button(texts["load_data_btn"], type="primary"):
        if not api_key:
            st.warning(texts["api_key_warning"])
        else:
            with st.spinner(texts["data_loading"]):
                try:
                    df = load_data(dataset_name)  # load_data는 아래에 정의
                    if len(df) > 10:
                        train_df, _ = train_test_split(df, test_size=0.1, random_state=42)
                    else:
                        train_df = df.copy()

                    pool_df = train_df.copy()
                    pool_df["content_en"] = pool_df["work_sequence_en"].tolist()

                    to_embed = pool_df["content_en"].tolist()
                    max_texts = min(len(to_embed), 30)
                    st.info(texts["demo_limit_info"].format(max_texts=max_texts))

                    embeds = embed_texts_with_openai(to_embed[:max_texts], api_key)
                    vecs = np.array(embeds, dtype="float32")
                    dim = vecs.shape[1]
                    index = faiss.IndexFlatL2(dim)
                    index.add(vecs)

                    ss.index = index
                    ss.embeddings = vecs
                    ss.retriever_pool_df = pool_df.iloc[:max_texts]
                    st.success(texts["data_load_success"].format(max_texts=max_texts))

                except Exception as e:
                    st.error(f"데이터 로드 오류: {e}")

    st.divider()
    st.markdown("### 🔍 평가 실행")

    activity_input = st.text_input(texts["activity_label"], key="user_activity")

    col_opt1, col_opt2 = st.columns([2, 1])
    with col_opt1:
        include_similar = st.checkbox(texts["include_similar_label"], value=True)
    with col_opt2:
        output_lang = st.selectbox(texts["result_language_label"], ["Korean", "English"], index=["Korean", "English"].index(ss.language))

    run_button = st.button(texts["run_button"], type="primary", use_container_width=True)

if run_button:
    if not activity_input:
        st.warning(texts["no_activity_warning"])
    elif ss.index is None:
        st.warning(texts["no_index_warning"])
    else:
        with st.spinner("처리 중..."):
            try:
                # === 1) 입력 활동을 영어로 변환 ===
                translate_to_en_prompt = f"Translate the following work activity into clear English:\n\n{activity_input.strip()}"
                activity_en = generate_with_gpt(translate_to_en_prompt, api_key, "English")

                # === 2) 임베딩 및 유사 사례 검색 ===
                q_emb_list = embed_texts_with_openai([activity_en], api_key)
                q_emb = q_emb_list[0]
                D, I = ss.index.search(np.array([q_emb], dtype="float32"), k=min(10, len(ss.retriever_pool_df)))
                sim_docs = ss.retriever_pool_df.iloc[I[0]]

                # === 3) 주요 유해위험요인 예측 (영어 내부 처리) ===
                hazard_prompt_en = construct_prompt_hazard_en(sim_docs, activity_en)
                hazard_en = generate_with_gpt(hazard_prompt_en, api_key, "English")

                # === 4) 위험도 평가 (영어 내부 처리) ===
                risk_prompt_en = construct_prompt_risk_en(sim_docs, activity_en, hazard_en)
                risk_json = generate_with_gpt(risk_prompt_en, api_key, "English")
                parse_result = parse_gpt_output_risk(risk_json)
                if not parse_result:
                    st.error(texts["parsing_error"])
                    st.stop()
                freq_en, intensity_en, T_en = parse_result
                grade_en = determine_grade(T_en)

                # === 5) 개선대책 생성 (영어 내부 처리) ===
                improvement_prompt_en = construct_prompt_improvement_en(sim_docs, activity_en, hazard_en, freq_en, intensity_en, T_en)
                improvement_json = generate_with_gpt(improvement_prompt_en, api_key, "English")
                parsed_imp = parse_gpt_output_improvement(improvement_json)
                if not parsed_imp:
                    st.error(texts["parsing_error_improvement"])
                    st.stop()
                improvement_plan_en = parsed_imp["improvement_plan"]

                # === 6) 결과를 출력 언어로 번역 ===
                if output_lang == "Korean":
                    translate_columns = {
                        "activity": ("Work Sequence", "작업활동 및 내용"),
                        "hazard": ("Hazarous Factors", "유해위험요인 및 환경측면 영향"),
                        "risk": ("Risk", "위험성"),
                        "improvement": ("Control Measures", "개선대책 및 세부관리방안")
                    }
                    col_labels = {
                        "work": "작업활동 및 내용 Work Sequence",
                        "hazard": "유해위험요인 및 환경측면 영향 Hazarous Factors",
                        "EHS": "EHS|",
                        "risk": "위험성 Risk |",
                        "control": "개선대책 및 세부관리방안 Control Measures |",
                        "in_charge": "개선담당자 In Charge",
                        "due_date": "개선일자 Correction Due Date"
                    }
                    # 번역
                    hazard_ko = generate_with_gpt(f"Translate to Korean:\n\n{hazard_en}", api_key, "Korean")
                    improvement_ko = generate_with_gpt(f"Translate to Korean preserving line breaks:\n\n{improvement_plan_en}", api_key, "Korean")
                    # 위험도 필드 자체는 숫자이므로 번역 불필요
                    activity_ko = activity_input.strip()
                else:
                    # English 출력이라면, 영어 원문 그대로 사용
                    col_labels = {
                        "work": "Work Sequence 작업활동 및 내용",
                        "hazard": "Hazarous Factors 유해위험요인 및 환경측면 영향",
                        "EHS": "EHS|",
                        "risk": "Risk | 위험성",
                        "control": "Control Measures 개선대책 및 세부관리방안 |",
                        "in_charge": "In Charge 개선담당자",
                        "due_date": "Correction Due Date 개선일자"
                    }
                    activity_ko = activity_en
                    hazard_ko = hazard_en
                    improvement_ko = improvement_plan_en

                # === 7) 화면 출력 ===
                st.markdown("## 📋 결과")
                # 주요 결과 요약
                col1, col2 = st.columns([3, 2])
                with col1:
                    st.markdown(f"**{col_labels['work']}**  \n{activity_ko}")
                    st.markdown(f"**{col_labels['hazard']}**  \n{hazard_ko}")
                with col2:
                    st.markdown(f"**{col_labels['risk']}**")
                    st.markdown(f"• 빈도 (Likelihood): {freq_en}  \n• 강도 (Severity): {intensity_en}  \n• T-value: {T_en} (Grade {grade_en})")

                st.markdown("### 🔍 유사 사례")
                if include_similar:
                    for i, row in sim_docs.iterrows():
                        work_i = row["work_sequence_en"]
                        hazard_i = row["hazard_en"]
                        freq_i = int(row["frequency"])
                        int_i = int(row["severity"])
                        T_i = freq_i * int_i
                        grade_i = determine_grade(T_i)
                        plan_i = row["control_en"]
                        # 결과 언어에 맞춰 번역 (간단하게, 한글 모드라면 한/영 병기 형태로 출력)
                        if output_lang == "Korean":
                            work_disp = row["work_sequence_ko"]
                            hazard_disp = row["hazard_ko"]
                            plan_disp = row["control_ko"]
                        else:
                            work_disp = work_i
                            hazard_disp = hazard_i
                            plan_disp = plan_i

                        with st.expander(f"유사 사례 {i + 1}"):
                            st.write(f"**{col_labels['work']}**  \n{work_disp}")
                            st.write(f"**{col_labels['hazard']}**  \n{hazard_disp}")
                            st.write(f"**{col_labels['risk']}**  \n• 빈도: {freq_i}  \n• 강도: {int_i}  \n• T: {T_i} (Grade {grade_i})")
                            st.write(f"**{col_labels['control']}**  \n{plan_disp}")

                st.markdown("## 🛠️ 개선대책")
                st.write(f"**{col_labels['control']}**  \n{improvement_ko}")
                st.write(f"**{col_labels['in_charge']}**  \n(미지정)")
                st.write(f"**{col_labels['due_date']}**  \n(미지정)")

                # === 8) 엑셀 다운로드 ===
                def create_excel():
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                        # 시트에 동일한 컬럼 순서와 이름 사용
                        columns = [
                            col_labels["work"],
                            col_labels["hazard"],
                            col_labels["EHS"],
                            col_labels["risk"],
                            col_labels["control"],
                            col_labels["in_charge"],
                            col_labels["due_date"]
                        ]
                        # 위험성 아래 row에 빈도/강도 별도 행으로 구성
                        risk_cell = (
                            f"빈도 likelihood: {freq_en}\n"
                            f"강도 severity: {intensity_en}"
                        )
                        row_data = {
                            columns[0]: activity_ko,
                            columns[1]: hazard_ko,
                            columns[2]: "",  # EHS 필드는 빈 값으로 둡니다.
                            columns[3]: risk_cell,
                            columns[4]: improvement_ko,
                            columns[5]: "",  # 담당자 미지정
                            columns[6]: ""   # 일자 미지정
                        }
                        df_out = pd.DataFrame([row_data], columns=columns)
                        df_out.to_excel(writer, sheet_name="Results", index=False)

                        # 유사사례 시트
                        if include_similar and not sim_docs.empty:
                            sim_rows = []
                            for i, row in sim_docs.iterrows():
                                freq_i = int(row["frequency"])
                                int_i = int(row["severity"])
                                T_i = freq_i * int_i
                                grade_i = determine_grade(T_i)
                                if output_lang == "Korean":
                                    work_disp = row["work_sequence_ko"]
                                    hazard_disp = row["hazard_ko"]
                                    plan_disp = row["control_ko"]
                                else:
                                    work_disp = row["work_sequence_en"]
                                    hazard_disp = row["hazard_en"]
                                    plan_disp = row["control_en"]
                                risk_cell_i = f"빈도 likelihood: {freq_i}\n강도 severity: {int_i}"
                                sim_rows.append({
                                    columns[0]: work_disp,
                                    columns[1]: hazard_disp,
                                    columns[2]: "",
                                    columns[3]: risk_cell_i,
                                    columns[4]: plan_disp,
                                    columns[5]: "",
                                    columns[6]: ""
                                })
                            df_sim = pd.DataFrame(sim_rows, columns=columns)
                            df_sim.to_excel(writer, sheet_name="Similar Cases", index=False)

                    return output.getvalue()

                excel_bytes = create_excel()
                st.download_button(
                    label=texts["download_excel"],
                    data=excel_bytes,
                    file_name="risk_assessment.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            except Exception as e:
                st.error(f"오류 발생: {e}")
                st.stop()

# -----------------------------------------------------------------------------  
# --------------------------- 유틸리티 함수들 --------------------------------  
# -----------------------------------------------------------------------------  

def determine_grade(value: int):
    """위험도 등급 계산 (영어 내부 기준)"""
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
    return 'Unknown'

@st.cache_data(show_spinner=False)
def load_data(selected_dataset_name: str):
    """
    데이터 로드함수.
    각 컬럼을 영어/한국어 양쪽으로 준비해 둡니다.
    """
    # 실제로는 xlsx를 로드하겠지만, 예시용으로 간단히 샘플 생성
    data = {
        "work_sequence_en": [
            "Unload steel beams using forklift at temporary site storage",
            "Install concrete/CMU blocks",
            "Excavation and backfilling work",
            "Exterior wall work on elevated platform",
            "Welding operations"
        ],
        "work_sequence_ko": [
            "임시 현장 저장소에서 포크리프트로 철골 구조재 하역작업",
            "콘크리트/CMU 블록 설치 작업",
            "굴착 및 되메우기 작업",
            "고소 작업대를 이용한 외벽 작업",
            "용접 작업"
        ],
        "hazard_en": [
            "Falling loads due to multiple lifting",
            "Fall due to insufficient work platform",
            "Burial from excavation wall collapse",
            "Fall due to missing safety harness",
            "Welding fumes and fire risk"
        ],
        "hazard_ko": [
            "다중 인양으로 인한 적재물 낙하",
            "불충분한 작업 발판으로 인한 추락",
            "굴착벽 붕괴로 인한 매몰",
            "안전대 미착용으로 인한 추락",
            "용접 흄 및 화재 위험"
        ],
        "frequency": [3, 3, 2, 4, 2],
        "severity": [5, 4, 5, 5, 3],
        "control_en": [
            "1) Do not lift multiple steel beams together. 2) Manage dimensions and weights.",
            "1) Install missing planks on scaffolding. 2) Equip safety harness anchorage.",
            "1) Maintain proper slope. 2) Reinforce excavation walls. 3) Inspect ground regularly.",
            "1) Enforce safety harness use. 2) Conduct pre-work safety training. 3) Install fall arrest nets.",
            "1) Provide proper ventilation. 2) Implement fire prevention measures. 3) Mandate PPE."
        ],
        "control_ko": [
            "1) 다수의 철골재를 함께 인양하지 않도록 관리\n2) 치수 및 중량 관리",
            "1) 비계에 누락된 목판 설치\n2) 안전대 부착 설비 사용",
            "1) 적절한 사면 유지\n2) 굴착 벽면 보강\n3) 정기적 지반 점검",
            "1) 안전대 착용 의무화\n2) 작업 전 안전교육 실시\n3) 추락 방지망 설치",
            "1) 적절한 환기\n2) 화재 예방 조치\n3) 보호구 착용"
        ]
    }
    df = pd.DataFrame(data)
    df["T"] = df["frequency"] * df["severity"]
    df["grade"] = df["T"].apply(determine_grade)
    return df

def embed_texts_with_openai(texts, api_key, model="text-embedding-3-large"):
    """
    OpenAI Embedding 호출 (영어 내부 처리)
    """
    if not api_key:
        return []

    openai.api_key = api_key
    embeddings = []
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        processed = [t.replace("\n", " ").strip() for t in batch]
        try:
            resp = openai.Embedding.create(model=model, input=processed)
            embeddings.extend([item["embedding"] for item in resp["data"]])
        except Exception as e:
            st.error(f"Embedding error: {e}")
            embeddings.extend([[0] * 1536] * len(batch))
    return embeddings

def generate_with_gpt(prompt, api_key, language, model="gpt-4o", max_retries=3):
    """
    OpenAI ChatCompletion 호출
    """
    if not api_key:
        return ""
    openai.api_key = api_key

    sys_prompt = "You are a construction site risk assessment expert. Provide clear, practical English responses."
    # 한국어나 중국어 출력 시, 내부에서 번역해야 하기 때문에, 여기서는 영어만 처리합니다.

    for attempt in range(max_retries):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=512,
                top_p=0.9
            )
            return resp["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt == max_retries - 1:
                st.error(f"GPT 호출 오류: {e}")
                return ""
            else:
                continue

def construct_prompt_hazard_en(sim_docs, activity_en: str):
    """
    주요 유해위험요인 예측용 영어 프롬프트
    """
    intro = "Here are examples of work activities and associated hazards:\n\n"
    example_fmt = "Case {i}:\n- Work Activity: {act}\n- Hazard: {haz}\n\n"
    for i, row in enumerate(sim_docs.itertuples(), start=1):
        intro += example_fmt.format(i=i, act=row.work_sequence_en, haz=row.hazard_en)
        if i >= 5:
            break
    query = f"Based on the above cases, predict the main hazards for the following work activity:\n\nWork Activity: {activity_en}\n\nHazard:"
    return intro + query

def construct_prompt_risk_en(sim_docs, activity_en: str, hazard_en: str):
    """
    위험도 평가용 영어 프롬프트
    """
    intro = (
        "Risk assessment criteria:\n"
        "- Frequency (1-5): 1=Very Rare, 2=Rare, 3=Occasional, 4=Frequent, 5=Very Frequent\n"
        "- Severity (1-5): 1=Minor Injury, 2=Light Injury, 3=Moderate Injury, 4=Serious Injury, 5=Fatality\n"
        "- T-value = Frequency x Severity\n\nReference cases:\n\n"
    )
    example_fmt = "Case {i}:\nInput: {inp}\nAssessment: Frequency={freq}, Severity={sev}, T-value={tval}\n\n"
    for i, row in enumerate(sim_docs.itertuples(), start=1):
        inp = f"{row.work_sequence_en} - {row.hazard_en}"
        freq = row.frequency
        sev = row.severity
        tval = freq * sev
        intro += example_fmt.format(i=i, inp=inp, freq=freq, sev=sev, tval=tval)
        if i >= 3:
            break
    query = (
        f"{intro}Please assess the following:\n\n"
        f"Work Activity: {activity_en}\nHazard: {hazard_en}\n\n"
        'Respond in JSON as: {"frequency": number, "severity": number, "T": number}'
    )
    return query

def parse_gpt_output_risk(gpt_output: str):
    """
    GPT 위험도 JSON 파싱
    """
    pattern = r'\{"frequency":\s*([1-5]),\s*"severity":\s*([1-5]),\s*"T":\s*([0-9]+)\}'
    match = re.search(pattern, gpt_output)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None

def construct_prompt_improvement_en(sim_docs, activity_en: str, hazard_en: str, freq: int, sev: int, T: int):
    """
    개선대책 생성용 영어 프롬프트
    """
    intro = "Provide at least 3 practical improvement measures (numbered) that reduce both frequency and severity.\n\n"
    example_fmt = (
        "Example {i}:\n"
        "Input Work Activity: {act}\n"
        "Input Hazard: {haz}\n"
        "Original Frequency: {freq}, Severity: {sev}, T-value: {tval}\n"
        "Output:\n{{\n"
        '  "improvement_plan": "{plan}",\n'
        "  \"improved_frequency\": {ifreq},\n"
        "  \"improved_severity\": {isev},\n"
        "  \"improved_T\": {itval},\n"
        '  "reduction_rate": {rrr}\n'
        "}}\n\n"
    )
    examples = ""
    count = 0
    for row in sim_docs.itertuples():
        orig_freq = row.frequency
        orig_sev = row.severity
        orig_t = orig_freq * orig_sev
        imp_freq = max(1, orig_freq - 1)
        imp_sev = max(1, orig_sev - 1)
        imp_t = imp_freq * imp_sev
        plan = row.control_en.replace("\n", " ")
        rrr = round(((orig_t - imp_t) / orig_t) * 100, 2) if orig_t else 0.0
        examples += example_fmt.format(
            i=count + 1,
            act=row.work_sequence_en,
            haz=row.hazard_en,
            freq=orig_freq,
            sev=orig_sev,
            tval=orig_t,
            plan=plan,
            ifreq=imp_freq,
            isev=imp_sev,
            itval=imp_t,
            rrr=rrr
        )
        count += 1
        if count >= 2:
            break

    query = (
        f"{examples}"
        f"Now assess for:\n\n"
        f"Work Activity: {activity_en}\nHazard: {hazard_en}\n"
        f"Original Frequency: {freq}, Severity: {sev}, T: {T}\n\n"
        'Output in JSON as: {"improvement_plan": "...", "improved_frequency": number, '
        '"improved_severity": number, "improved_T": number, "reduction_rate": number}'
    )
    return query

def parse_gpt_output_improvement(gpt_output: str):
    """
    개선대책 JSON 파싱
    """
    # JSON 추출
    match = re.search(r'\{.*\}', gpt_output, re.DOTALL)
    if not match:
        return None
    try:
        import json
        data = json.loads(match.group(0))
        return {
            "improvement_plan": data.get("improvement_plan", "").replace("\\n", "\n"),
            "improved_frequency": data.get("improved_frequency", 1),
            "improved_severity": data.get("improved_severity", 1),
            "improved_T": data.get("improved_T", data.get("improved_frequency", 1) * data.get("improved_severity", 1)),
            "reduction_rate": data.get("reduction_rate", 0.0)
        }
    except json.JSONDecodeError:
        return None
