import streamlit as st
import sqlite3
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from typing import TypedDict
import os

# 파인튜닝된 모델 임포트
from models.finetuned_model import FinetunedModel

# 환경 변수 설정 (실제 사용시 필요한 경우 여기에 추가)

# 페이지 설정
st.set_page_config(
    page_title="AI 펫닥터 (기본 LLM)",
    page_icon="🐕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 테마별 CSS 스타일 정의
def get_theme_css(theme_name):
    if theme_name == "🌿 자연 친화":
        return """
        <style>
        /* 전체 앱 배경 - 자연스러운 녹색 그라데이션 */
        .stApp {
            background: linear-gradient(135deg, #FFFFFF 0%, #F0EAD6 30%, #E0E5D0 70%, #D4E6C7 100%);
            min-height: 100vh;
        }
        
        /* 메인 컨테이너 */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: transparent;
        }
        
        /* 사이드바 */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(224, 229, 208, 0.8), rgba(240, 234, 214, 0.8));
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(109, 146, 126, 0.2);
        }
        
        /* 헤더 */
        .main-header {
            background: linear-gradient(135deg, #6D927E 0%, #5A7A6B 100%);
            padding: 2.5rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 30px rgba(109, 146, 126, 0.3);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.9;
            color: rgba(255,255,255,0.95);
        }
        
        /* 카드 스타일 */
        .pet-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(240,234,214,0.8) 100%);
            padding: 2rem;
            border-radius: 18px;
            border-left: 5px solid #6D927E;
            margin: 1.5rem 0;
            box-shadow: 0 6px 25px rgba(109, 146, 126, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(109, 146, 126, 0.1);
            transition: all 0.3s ease;
        }
        
        .pet-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 35px rgba(109, 146, 126, 0.25);
        }
        
        .recommendation-box {
            background: linear-gradient(135deg, #F0EAD6 0%, #E0E5D0 100%);
            padding: 1.8rem;
            border-radius: 15px;
            border: 2px solid #6D927E;
            margin: 1rem 0;
            box-shadow: 0 4px 18px rgba(109, 146, 126, 0.2);
            color: #4A4F44;
        }
        
        .health-status {
            background: linear-gradient(135deg, #FFFEF7 0%, #F5F2E8 100%);
            padding: 1.8rem;
            border-radius: 15px;
            border: 2px solid #CD5C5C;
            margin: 1rem 0;
            box-shadow: 0 4px 18px rgba(205, 92, 92, 0.2);
            color: #5C4033;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #FFF8F5 0%, #FFEBE5 100%);
            padding: 1.8rem;
            border-radius: 15px;
            border: 2px solid #CD5C5C;
            margin: 1rem 0;
            color: #8B4513;
            box-shadow: 0 4px 18px rgba(205, 92, 92, 0.2);
        }
        
        /* 버튼 */
        .stButton > button {
            background: linear-gradient(135deg, #6D927E 0%, #5A7A6B 100%);
            color: white;
            border-radius: 25px;
            border: none;
            font-weight: 600;
            padding: 0.7rem 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(109, 146, 126, 0.3);
            font-size: 1rem;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #5A7A6B 0%, #4A6B58 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(109, 146, 126, 0.4);
        }
        
        /* 입력 필드 */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border-radius: 12px;
            border: 2px solid rgba(109, 146, 126, 0.3);
            background: rgba(255,255,255,0.9);
            color: #4A4F44;
            padding: 0.7rem;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #6D927E;
            box-shadow: 0 0 0 3px rgba(109, 146, 126, 0.1);
        }
        
        /* 메트릭 */
        .stMetric {
            background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(224,229,208,0.8));
            padding: 1rem;
            border-radius: 12px;
            border-left: 4px solid #CD5C5C;
            box-shadow: 0 3px 12px rgba(109, 146, 126, 0.1);
        }
        
        /* 사이드바 요소들 */
        .css-1d391kg .stSelectbox > label,
        .css-1d391kg .stButton > button {
            color: #4A4F44;
        }
        
        /* 텍스트 색상 통일 */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #4A4F44;
        }
        
        .stApp p, .stApp span, .stApp div {
            color: #5C4033;
        }
        </style>
        """
    
    elif theme_name == "🌊 청량 블루":
        return """
        <style>
        /* 전체 앱 배경 - 시원한 블루 그라데이션 */
        .stApp {
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 30%, #90caf9 70%, #64b5f6 100%);
            min-height: 100vh;
        }
        
        /* 메인 컨테이너 */
        .main .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            background: transparent;
        }
        
        /* 사이드바 */
        .css-1d391kg {
            background: linear-gradient(180deg, rgba(227, 242, 253, 0.9), rgba(187, 222, 251, 0.8));
            backdrop-filter: blur(10px);
            border-right: 1px solid rgba(33, 150, 243, 0.2);
        }
        
        /* 헤더 */
        .main-header {
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 50%, #0d47a1 100%);
            padding: 2.5rem;
            border-radius: 20px;
            text-align: center;
            color: white;
            margin-bottom: 2rem;
            box-shadow: 0 8px 30px rgba(33, 150, 243, 0.4);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .main-header h1 {
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
            margin-bottom: 0.5rem;
            font-weight: 700;
        }
        
        .main-header p {
            font-size: 1.2rem;
            opacity: 0.95;
            color: rgba(255,255,255,0.95);
        }
        
        /* 카드 스타일 */
        .pet-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.95) 0%, rgba(227,242,253,0.8) 100%);
            padding: 2rem;
            border-radius: 18px;
            border-left: 5px solid #2196f3;
            margin: 1.5rem 0;
            box-shadow: 0 6px 25px rgba(33, 150, 243, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(33, 150, 243, 0.1);
            transition: all 0.3s ease;
        }
        
        .pet-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 35px rgba(33, 150, 243, 0.3);
        }
        
        .recommendation-box {
            background: linear-gradient(135deg, #e1f5fe 0%, #b3e5fc 100%);
            padding: 1.8rem;
            border-radius: 15px;
            border: 2px solid #03a9f4;
            margin: 1rem 0;
            box-shadow: 0 4px 18px rgba(3, 169, 244, 0.2);
            color: #01579b;
        }
        
        .health-status {
            background: linear-gradient(135deg, #e0f2f1 0%, #b2dfdb 100%);
            padding: 1.8rem;
            border-radius: 15px;
            border: 2px solid #26c6da;
            margin: 1rem 0;
            box-shadow: 0 4px 18px rgba(38, 198, 218, 0.2);
            color: #006064;
        }
        
        .warning-box {
            background: linear-gradient(135deg, #fff8e1 0%, #ffecb3 100%);
            padding: 1.8rem;
            border-radius: 15px;
            border: 2px solid #ffa726;
            margin: 1rem 0;
            color: #e65100;
            box-shadow: 0 4px 18px rgba(255, 167, 38, 0.2);
        }
        
        /* 버튼 */
        .stButton > button {
            background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
            color: white;
            border-radius: 25px;
            border: none;
            font-weight: 600;
            padding: 0.7rem 2rem;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(33, 150, 243, 0.3);
            font-size: 1rem;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, #1976d2 0%, #0d47a1 100%);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(33, 150, 243, 0.4);
        }
        
        /* 입력 필드 */
        .stTextInput > div > div > input,
        .stTextArea > div > div > textarea,
        .stSelectbox > div > div > select {
            border-radius: 12px;
            border: 2px solid rgba(33, 150, 243, 0.3);
            background: rgba(255,255,255,0.9);
            color: #01579b;
            padding: 0.7rem;
        }
        
        .stTextInput > div > div > input:focus,
        .stTextArea > div > div > textarea:focus,
        .stSelectbox > div > div > select:focus {
            border-color: #2196f3;
            box-shadow: 0 0 0 3px rgba(33, 150, 243, 0.1);
        }
        
        /* 메트릭 */
        .stMetric {
            background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(227,242,253,0.8));
            padding: 1rem;
            border-radius: 12px;
            border-left: 4px solid #26c6da;
            box-shadow: 0 3px 12px rgba(33, 150, 243, 0.15);
        }
        
        /* 사이드바 요소들 */
        .css-1d391kg .stSelectbox > label,
        .css-1d391kg .stButton > button {
            color: #01579b;
        }
        
        /* 텍스트 색상 통일 */
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
            color: #01579b;
        }
        
        .stApp p, .stApp span, .stApp div {
            color: #0277bd;
        }
        </style>
        """
    
    return ""



# LangGraph 상태 정의
class GraphState(TypedDict):
    pet_info: dict
    symptoms: str
    health_analysis: str
    supplement_recommendations: List[dict]
    consultation_id: str

class BasicLLMPetDoctor:
    def __init__(self, model_path=None):
        # 파인튜닝된 모델 초기화
        self.model_path = model_path
        self.finetuned_model = FinetunedModel(model_path=model_path)
        
        self.setup_database()
        self.setup_prompts()
        self.setup_rag_system()
        self.setup_langgraph()
        
    def setup_database(self):
        """데이터베이스 설정"""
        conn = sqlite3.connect('pet_consultations.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # 상담 기록 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS consultations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pet_name TEXT,
                pet_type TEXT,
                pet_age INTEGER,
                pet_weight REAL,
                symptoms TEXT,
                health_analysis TEXT,
                recommendations TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 영양제 데이터베이스
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS supplements (
                id INTEGER PRIMARY KEY,
                name TEXT,
                brand TEXT,
                category TEXT,
                description TEXT,
                ingredients TEXT,
                recommended_for TEXT,
                dosage TEXT,
                price REAL,
                rating REAL,
                side_effects TEXT,
                contraindications TEXT
            )
        ''')
        
        # 향상된 영양제 샘플 데이터
        sample_supplements = [
            (1, "관절 케어 플러스", "펫라이프", "관절건강", "글루코사민과 콘드로이틴이 풍부한 관절 건강 영양제", 
             "글루코사민, 콘드로이틴, MSM, 콜라겐", "관절염, 관절 통증, 노령견, 대형견", "체중 10kg당 1정", 35000, 4.5,
             "드물게 위장 장애", "신장 질환, 당뇨병 주의"),
            
            (2, "소화 건강 프로바이오틱", "펫케어", "소화기건강", "10억 CFU 유산균과 소화효소가 함유된 소화 개선 영양제",
             "락토바실러스, 비피도박테리움, 프레바이오틱", "소화불량, 설사, 변비, 장염", "1일 1회 1포", 28000, 4.3,
             "초기 가스 증가 가능", "면역억제제 복용시 주의"),
            
            (3, "멀티 비타민 & 미네랄", "펫비타", "종합영양", "반려동물 전용 종합 비타민 미네랄 복합제",
             "비타민 A,B,C,D,E, 아연, 철분, 엽산", "영양 보충, 면역력 강화, 성장기", "체중 5kg당 0.5정", 22000, 4.1,
             "과량 섭취시 비타민 과다증", "간 질환시 철분 섭취 주의"),
            
            (4, "오메가3 피쉬오일", "마린펫", "피부모질", "순수 알래스카 연어에서 추출한 고농도 오메가3",
             "EPA 300mg, DHA 200mg, 비타민E", "피부염, 털빠짐, 알레르기, 심장건강", "체중 5kg당 0.5ml", 31000, 4.6,
             "드물게 생선 알레르기", "혈액응고장애 약물과 병용 주의"),
            
            (5, "간 건강 실리마린", "펫리버", "간기능", "밀크씨슬에서 추출한 고농도 실리마린",
             "실리마린 80%, 타우린, 비타민B", "간기능 저하, 해독, 간염 회복", "체중 10kg당 1정", 26000, 4.4,
             "드물게 알레르기 반응", "담관 폐쇄시 금기"),
            
            (6, "면역력 강화 베타글루칸", "이뮨펫", "면역강화", "효모에서 추출한 베타글루칸과 면역 복합체",
             "베타글루칸, 아연, 셀레늄, 비타민C", "면역력 저하, 반복 감염, 회복기", "1일 1회 1캡슐", 33000, 4.2,
             "없음", "자가면역질환시 주의"),
            
            (7, "심장 건강 코엔자임Q10", "카디오펫", "심장건강", "심장 근육 에너지 생산을 돕는 코엔자임Q10",
             "코엔자임Q10, L-카르니틴, 타우린", "심장병, 호흡곤란, 기침, 노령견", "체중 10kg당 1정", 38000, 4.3,
             "드물게 위장 장애", "혈압약 복용시 상담 필요"),
            
            (8, "요로 건강 크랜베리", "유로펫", "비뇨기건강", "크랜베리 추출물과 D-만노스가 함유된 요로 건강제",
             "크랜베리 추출물, D-만노스, 비타민C", "방광염, 요로감염, 혈뇨", "1일 2회 1정", 29000, 4.0,
             "드물게 설사", "신장결석 병력시 주의")
        ]
        
        cursor.executemany('''
            INSERT OR REPLACE INTO supplements 
            (id, name, brand, category, description, ingredients, recommended_for, 
             dosage, price, rating, side_effects, contraindications) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', sample_supplements)
        
        conn.commit()
        conn.close()
    
    def setup_prompts(self):
        """전문가 수준의 프롬프트 템플릿 설정"""
        
        self.symptom_analysis_prompt = """
당신은 경험이 풍부한 수의사입니다. 반려동물의 증상을 분석하여 가능한 건강 문제를 평가해주세요.

**반려동물 정보:**
- 이름: {pet_name}
- 종류: {pet_type}
- 나이: {pet_age}세
- 체중: {pet_weight}kg

**증상:**
{symptoms}

**관련 의학 지식:**
{medical_context}

다음 형식으로 분석해주세요:

**🔍 주요 증상 분석:**
- 관찰된 증상들의 의학적 의미
- 증상의 심각도 평가

**🏥 가능한 진단:**
1. 가장 가능성 높은 질환 (확률 포함)
2. 고려해야 할 다른 질환들

**⚠️ 주의사항:**
- 응급상황 여부
- 수의사 진료 필요성
- 관찰해야 할 추가 증상

**📋 추천 조치:**
- 즉시 취할 수 있는 응급처치
- 일상 관리 방법
- 영양 보충 필요성

정확한 진단을 위해서는 반드시 전문 수의사의 진료를 받으시기 바랍니다.
"""

        self.supplement_recommendation_prompt = """
당신은 반려동물 영양학 전문가입니다. 분석된 건강 상태를 바탕으로 적절한 영양제를 추천해주세요.

**건강 분석 결과:**
{health_analysis}

**반려동물 정보:**
- 종류: {pet_type}
- 나이: {pet_age}세  
- 체중: {pet_weight}kg

**사용 가능한 영양제:**
{available_supplements}

다음 기준으로 영양제를 추천해주세요:

**추천 기준:**
1. 증상과의 연관성
2. 반려동물의 나이/체중 적합성
3. 안전성 및 부작용
4. 다른 영양제와의 상호작용
5. 비용 대비 효과

**추천 형식:**
각 영양제별로:
- 추천 이유 (의학적 근거)
- 예상 효과
- 복용법 및 주의사항
- 다른 영양제와 병용 가능성
- 효과를 보기까지 예상 기간

**중요:** 
- 최대 3개까지만 추천
- 반려동물의 현재 상태에 가장 적합한 것부터 우선순위 부여
- 부작용이나 금기사항이 있다면 반드시 언급
"""

        self.emergency_assessment_prompt = """
다음 증상들이 응급상황에 해당하는지 평가해주세요:

증상: {symptoms}
반려동물: {pet_type}, {pet_age}세

응급도를 1-5단계로 평가하고 이유를 설명해주세요:
1 = 일상 관찰, 2 = 며칠 내 병원, 3 = 1-2일 내 병원, 4 = 당일 병원, 5 = 즉시 응급실

평가 결과와 근거를 제시해주세요.
"""

    def setup_rag_system(self):
        """RAG 시스템 설정 - 더 풍부한 지식베이스"""
        
        # 확장된 수의학 지식베이스
        knowledge_base = [
            # 관절 질환
            "개의 관절염은 연골의 퇴행성 변화로 발생하며, 주요 증상으로는 절뚝거림, 계단 오르내리기 거부, 활동량 감소가 있습니다. 대형견과 노령견에서 흔하며, 글루코사민과 콘드로이틴 보충이 도움됩니다. 체중 관리와 적절한 운동이 중요합니다.",
            
            "슬개골 탈구는 소형견에서 흔한 질환으로, 무릎뼈가 정상 위치에서 벗어나는 상태입니다. 간헐적 절뚝거림, 다리를 들고 걷기, 점프 후 절뚝거림 등의 증상을 보입니다. 정도에 따라 내과적 치료나 수술이 필요합니다.",
            
            # 소화기 질환  
            "급성 위장염은 구토, 설사, 식욕부진을 주요 증상으로 합니다. 식이 변화, 스트레스, 세균 감염 등이 원인이 될 수 있습니다. 금식 후 점진적 식이 재개와 프로바이오틱 보충이 도움됩니다.",
            
            "고양이의 털볼은 그루밍 과정에서 삼킨 털이 위장관에 축적되어 발생합니다. 건조한 기침, 구토, 변비가 주요 증상이며, 털볼 전용 사료와 브러싱이 예방에 도움됩니다.",
            
            # 피부 질환
            "아토피 피부염은 환경 알레르겐에 대한 과민반응으로 발생하는 만성 피부질환입니다. 가려움, 발진, 털빠짐, 2차 세균감염이 흔합니다. 오메가3 지방산 보충과 알레르겐 회피가 중요합니다.",
            
            "음식 알레르기는 특정 단백질에 대한 면역반응으로 발생합니다. 가려움, 소화불량, 귀 염증이 주요 증상이며, 제한 식이 요법을 통한 진단이 필요합니다.",
            
            # 간 질환
            "간 기능 저하는 식욕부진, 구토, 황달, 복수 등의 증상을 보입니다. 독성 물질 노출, 감염, 종양 등이 원인이 될 수 있습니다. 실리마린과 같은 간 보호제가 도움이 됩니다.",
            
            # 심장 질환
            "심장병은 기침, 호흡곤란, 운동 불내성, 복수 등의 증상을 보입니다. 선천성 심질환과 후천성 심질환으로 구분되며, 코엔자임Q10과 타우린 보충이 도움됩니다.",
            
            # 비뇨기 질환
            "방광염은 빈뇨, 혈뇨, 소변시 통증을 주요 증상으로 합니다. 세균 감염, 스트레스, 결석 등이 원인이며, 크랜베리 추출물과 충분한 수분 섭취가 도움됩니다.",
            
            # 노령견 관리
            "노령견은 관절염, 심장병, 간기능 저하, 인지기능 저하 등 다양한 문제를 보일 수 있습니다. 정기적인 건강검진과 적절한 영양 보충이 중요합니다.",
            
            # 응급상황
            "다음 증상들은 응급상황입니다: 의식 잃음, 경련, 심한 호흡곤란, 지속적 구토/설사, 복부 팽만, 체온 40도 이상, 창백한 잇몸. 즉시 동물병원 응급실로 가야 합니다."
        ]
        
        # 문서 생성 및 분할
        documents = [Document(page_content=text) for text in knowledge_base]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(documents)
        
        # 임베딩 및 벡터스토어 생성
        try:
            # OpenAI 임베딩 사용 (API 키 필요)
            embeddings = OpenAIEmbeddings()
            self.vectorstore = FAISS.from_documents(splits, embeddings)
        except:
            try:
                # 무료 HuggingFace 임베딩 사용 (대안)
                embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.vectorstore = FAISS.from_documents(splits, embeddings)
            except:
                # 임베딩 없이 키워드 매칭으로 대체
                self.vectorstore = None
                self.knowledge_base_text = " ".join(knowledge_base)

    def setup_langgraph(self):
        """LangGraph 워크플로우 설정"""
        workflow = StateGraph(GraphState)
        
        # 노드 추가
        workflow.add_node("emergency_check", self.emergency_check)
        workflow.add_node("analyze_symptoms", self.analyze_symptoms)
        workflow.add_node("recommend_supplements", self.recommend_supplements)
        workflow.add_node("save_consultation", self.save_consultation)
        
        # 엣지 설정
        workflow.set_entry_point("emergency_check")
        workflow.add_edge("emergency_check", "analyze_symptoms")
        workflow.add_edge("analyze_symptoms", "recommend_supplements") 
        workflow.add_edge("recommend_supplements", "save_consultation")
        workflow.add_edge("save_consultation", END)
        
        self.app = workflow.compile()

    def emergency_check(self, state: GraphState) -> GraphState:
        """응급상황 체크"""
        symptoms = state["symptoms"].lower()
        pet_info = state["pet_info"]
        
        emergency_keywords = [
            "의식을 잃", "경련", "호흡곤란", "숨을 못", "피를 토", "복부팽만", 
            "고열", "41도", "창백", "잇몸이 하얗", "지속적 구토", "심한 설사"
        ]
        
        emergency_level = 0
        emergency_reasons = []
        
        for keyword in emergency_keywords:
            if keyword in symptoms:
                emergency_level = max(emergency_level, 4)  # 높은 응급도
                emergency_reasons.append(f"'{keyword}' 증상 발견")
        
        # 나이 고려
        if pet_info['age'] > 10 and any(word in symptoms for word in ["숨가쁨", "기침", "식욕없음"]):
            emergency_level = max(emergency_level, 3)
            emergency_reasons.append("고령 + 심각한 증상")
        
        if emergency_level >= 4:
            state["health_analysis"] = f"""
🚨 **응급상황 의심** 🚨

**응급도: {emergency_level}/5**

**응급 의심 근거:**
{chr(10).join(f"• {reason}" for reason in emergency_reasons)}

**즉시 조치:**
1. 가까운 24시간 동물병원 응급실로 즉시 이동
2. 이동 중 반려동물을 따뜻하게 유지
3. 구토물이 기도로 들어가지 않도록 주의
4. 병원에 미리 전화하여 상황 설명

**⚠️ 중요: 영양제 추천보다 응급 처치가 우선입니다!**
"""
        
        return state

    def analyze_symptoms(self, state: GraphState) -> GraphState:
        """LLM을 사용한 증상 분석"""
        
        # 응급상황인 경우 추가 분석 생략
        if "응급상황" in state.get("health_analysis", ""):
            return state
        
        pet_info = state["pet_info"]
        symptoms = state["symptoms"]
        
        # RAG를 통한 관련 정보 검색
        if self.vectorstore:
            relevant_docs = self.vectorstore.similarity_search(symptoms, k=3)
            medical_context = "\n".join([doc.page_content for doc in relevant_docs])
        else:
            # 키워드 매칭 대체
            medical_context = self.get_relevant_knowledge(symptoms)
        
        # 파인튜닝된 모델 사용
        try:
            prompt = self.symptom_analysis_prompt.format(
                pet_name=pet_info['name'],
                pet_type=pet_info['type'],
                pet_age=pet_info['age'],
                pet_weight=pet_info['weight'],
                symptoms=symptoms,
                medical_context=medical_context
            )
            
            # 파인튜닝된 모델로 응답 생성
            analysis = self.finetuned_model.generate_response(prompt, context=medical_context)
            
        except Exception as e:
            # OpenAI API 사용 불가시 규칙 기반 분석
            print(f"OpenAI API 오류: {e}")
            analysis = self.rule_based_analysis(pet_info, symptoms, medical_context)
        
        state["health_analysis"] = analysis
        return state

    def get_relevant_knowledge(self, symptoms):
        """키워드 매칭을 통한 관련 지식 추출"""
        symptoms_lower = symptoms.lower()
        relevant_knowledge = []
        
        knowledge_map = {
            "절뚝": "개의 관절염은 연골의 퇴행성 변화로 발생하며, 주요 증상으로는 절뚝거림, 계단 오르내리기 거부, 활동량 감소가 있습니다.",
            "관절": "관절 문제는 글루코사민과 콘드로이틴 보충이 도움되며, 체중 관리와 적절한 운동이 중요합니다.",
            "구토": "급성 위장염은 구토, 설사, 식욕부진을 주요 증상으로 합니다. 금식 후 점진적 식이 재개가 필요합니다.",
            "설사": "설사는 식이 변화, 스트레스, 세균 감염 등이 원인이 될 수 있습니다. 프로바이오틱 보충이 도움됩니다.",
            "가려움": "아토피 피부염은 가려움, 발진, 털빠짐을 주요 증상으로 하며, 오메가3 지방산 보충이 효과적입니다.",
            "털빠짐": "털빠짐은 영양 불균형이나 알레르기가 원인일 수 있으며, 오메가3 보충이 도움됩니다.",
        }
        
        for keyword, knowledge in knowledge_map.items():
            if keyword in symptoms_lower:
                relevant_knowledge.append(knowledge)
        
        return "\n".join(relevant_knowledge) if relevant_knowledge else "일반적인 수의학 지식을 바탕으로 분석합니다."

    def rule_based_analysis(self, pet_info, symptoms, context):
        """규칙 기반 분석 (LLM 백업)"""
        analysis = f"**{pet_info['name']}({pet_info['type']}, {pet_info['age']}세)의 건강 분석**\n\n"
        
        symptoms_lower = symptoms.lower()
        
        if any(keyword in symptoms_lower for keyword in ["절뚝", "다리", "관절", "계단"]):
            analysis += """🔍 **관절 관련 문제 의심**
• 관절염 또는 관절 손상 가능성
• 노령견의 경우 퇴행성 관절염 가능성 높음
• 소형견의 경우 슬개골 탈구 의심

**추천 조치:**
• 계단 사용 제한, 미끄럽지 않은 바닥재 사용
• 관절 영양제 (글루코사민, 콘드로이틴) 고려
• 체중 관리 중요
• 수의사 진료 권장

"""
            
        if any(keyword in symptoms_lower for keyword in ["구토", "토", "설사", "소화", "식욕"]):
            analysis += """🔍 **소화기 문제 의심**
• 급성 위장염 또는 식이 불내성 가능성
• 스트레스나 식이 변화가 원인일 수 있음
• 탈수 위험 주의

**추천 조치:**
• 12-24시간 금식 후 점진적 식이 재개
• 소량씩 자주 급식
• 프로바이오틱 고려
• 증상 지속시 수의사 진료

"""
            
        if any(keyword in symptoms_lower for keyword in ["가려움", "긁", "털빠짐", "발진", "피부"]):
            analysis += """🔍 **피부 관련 문제 의심**
• 알레르기 피부염 또는 아토피 가능성
• 음식 알레르기나 환경 알레르기 고려
• 2차 세균 감염 주의

**추천 조치:**
• 알레르기 유발 요소 제거
• 오메가3 지방산 보충
• 항알레르기 샴푸 사용
• 지속시 알레르기 검사 권장

"""
        
        if not any(keyword in symptoms_lower for keyword in ["절뚝", "구토", "가려움", "설사"]):
            analysis += """🔍 **일반적인 건강 관리**
• 구체적인 질병 징후는 발견되지 않음
• 예방적 건강 관리 중요
• 정기적인 건강검진 권장

**추천 조치:**
• 균형잡힌 영양 공급
• 적절한 운동과 스트레스 관리
• 정기적인 건강검진

"""
        
        analysis += "\n⚠️ **중요**: 이 분석은 참고용이며, 정확한 진단과 치료를 위해서는 전문 수의사와 상담하시기 바랍니다."
        
        return analysis

    def recommend_supplements(self, state: GraphState) -> GraphState:
        """영양제 추천"""
        # 응급상황인 경우 영양제 추천 생략
        if "응급상황" in state.get("health_analysis", ""):
            state["supplement_recommendations"] = []
            return state
        
        symptoms = state["symptoms"].lower()
        pet_info = state["pet_info"]
        health_analysis = state["health_analysis"]
        
        # 데이터베이스에서 영양제 정보 조회
        conn = sqlite3.connect('pet_consultations.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # 증상별 카테고리 매핑
        category_mapping = {
            "관절": "관절건강",
            "소화": "소화기건강", 
            "피부": "피부모질",
            "면역": "면역강화",
            "심장": "심장건강",
            "간": "간기능",
            "방광": "비뇨기건강"
        }
        
        relevant_categories = []
        for keyword, category in category_mapping.items():
            if keyword in symptoms or keyword in health_analysis.lower():
                relevant_categories.append(category)
        
        # 기본 종합영양제 추가
        if not relevant_categories:
            relevant_categories.append("종합영양")
        
        # 영양제 조회
        recommendations = []
        for category in relevant_categories:
            cursor.execute(
                "SELECT * FROM supplements WHERE category = ? ORDER BY rating DESC LIMIT 2",
                (category,)
            )
            recommendations.extend(cursor.fetchall())
        
        conn.close()
        
        # 파인튜닝된 모델을 통한 영양제 추천 (선택적)
        try:
            available_supplements = self.format_supplements_for_llm(recommendations)
            
            prompt = self.supplement_recommendation_prompt.format(
                health_analysis=health_analysis,
                pet_type=pet_info['type'],
                pet_age=pet_info['age'],
                pet_weight=pet_info['weight'],
                available_supplements=available_supplements
            )
            
            # 파인튜닝된 모델로 응답 생성
            system_prompt = "당신은 반려동물 영양학 전문가입니다. 안전하고 효과적인 영양제를 추천해주세요."
            llm_recommendation = self.finetuned_model.generate_response(prompt, context=system_prompt)
            
        except Exception as e:
            print(f"LLM 추천 오류: {e}")
            llm_recommendation = None
        
        # 영양제 정보를 딕셔너리 형태로 변환
        supplement_list = []
        for rec in recommendations[:3]:  # 최대 3개까지
            supplement_info = {
                'id': rec[0],
                'name': rec[1],
                'brand': rec[2],
                'category': rec[3],
                'description': rec[4],
                'ingredients': rec[5],
                'recommended_for': rec[6],
                'dosage': rec[7],
                'price': rec[8],
                'rating': rec[9],
                'side_effects': rec[10] if len(rec) > 10 else "알려진 부작용 없음",
                'contraindications': rec[11] if len(rec) > 11 else "특별한 금기사항 없음",
                'llm_analysis': llm_recommendation if llm_recommendation else "기본 추천"
            }
            supplement_list.append(supplement_info)
        
        state["supplement_recommendations"] = supplement_list
        return state

    def format_supplements_for_llm(self, supplements):
        """LLM에 전달할 영양제 정보 포맷"""
        formatted = []
        for supp in supplements:
            formatted.append(f"""
제품명: {supp[1]}
브랜드: {supp[2]}
카테고리: {supp[3]}
설명: {supp[4]}
주요 성분: {supp[5]}
추천 대상: {supp[6]}
복용법: {supp[7]}
가격: {supp[8]:,}원
평점: {supp[9]}/5.0
부작용: {supp[10] if len(supp) > 10 else '없음'}
금기사항: {supp[11] if len(supp) > 11 else '없음'}
""")
        return "\n".join(formatted)

    def save_consultation(self, state: GraphState) -> GraphState:
        """상담 내용 저장"""
        pet_info = state["pet_info"]
        
        conn = sqlite3.connect('pet_consultations.db', check_same_thread=False)
        cursor = conn.cursor()
        
        recommendations_json = json.dumps(state["supplement_recommendations"], ensure_ascii=False)
        
        cursor.execute('''
            INSERT INTO consultations 
            (pet_name, pet_type, pet_age, pet_weight, symptoms, health_analysis, recommendations)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            pet_info['name'],
            pet_info['type'], 
            pet_info['age'],
            pet_info['weight'],
            state['symptoms'],
            state['health_analysis'],
            recommendations_json
        ))
        
        consultation_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        state["consultation_id"] = str(consultation_id)
        return state

    def get_consultation_history(self, limit=10):
        """상담 이력 조회"""
        conn = sqlite3.connect('pet_consultations.db', check_same_thread=False)
        
        query = '''
            SELECT pet_name, pet_type, symptoms, timestamp, health_analysis
            FROM consultations 
            ORDER BY timestamp DESC 
            LIMIT ?
        '''
        
        df = pd.read_sql_query(query, conn, params=(limit,))
        conn.close()
        
        return df
    
    def reset_database(self):
        """데이터베이스 초기화"""
        conn = sqlite3.connect('pet_consultations.db', check_same_thread=False)
        cursor = conn.cursor()
        
        # 모든 테이블 삭제
        cursor.execute('DROP TABLE IF EXISTS consultations')
        cursor.execute('DROP TABLE IF EXISTS supplements')
        
        conn.commit()
        conn.close()
        
        # 데이터베이스 재설정
        self.setup_database()

# 시스템 초기화
@st.cache_resource
def init_system(model_path="models/finetuned_model"):
    return BasicLLMPetDoctor(model_path=model_path)

# 모델 설정 체크
def check_model_setup():
    """파인튜닝된 모델 설정 확인"""
    # 모델 파일 존재 여부 등 필요한 체크 로직 추가
    return True

# 메인 앱 시작
# 사이드바에서 입력한 모델 경로 사용
if 'model_path' in locals():
    system = init_system(model_path=model_path)
else:
    system = init_system()

# 테마별 헤더 텍스트
theme_headers = {
    "🌿 자연 친화": {
        "icon": "🐕🌿",
        "title": "AI 펫닥터 - 자연과 함께",
        "subtitle": "자연 친화적인 방식으로 반려동물의 건강을 케어합니다"
    },
    "🌊 청량 블루": {
        "icon": "🐕💙",
        "title": "AI 펫닥터 - 프레시 블루", 
        "subtitle": "깔끔하고 시원한 디자인으로 건강 상담"
    }
}

# 테마 설정
if 'theme_choice' not in st.session_state:
    st.session_state.theme_choice = "🌿 자연 친화"

# CSS 적용
st.markdown(get_theme_css(st.session_state.theme_choice), unsafe_allow_html=True)

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 테마 선택
    theme_choice = st.selectbox("🎨 테마 선택", list(theme_headers.keys()), 
                               index=list(theme_headers.keys()).index(st.session_state.theme_choice))
    
    # 테마가 변경되면 세션 상태 업데이트
    if theme_choice != st.session_state.theme_choice:
        st.session_state.theme_choice = theme_choice
        st.rerun()
    
    # 현재 테마의 헤더 표시
    current_header = theme_headers.get(st.session_state.theme_choice, theme_headers["🌿 자연 친화"])
    st.markdown(f'<div class="main-header"><h1>{current_header["icon"]} {current_header["title"]}</h1><p>{current_header["subtitle"]}</p></div>', unsafe_allow_html=True)
    
    # 모델 경로 입력 (선택사항)
    model_path = st.text_input("파인튜닝된 모델 경로 (선택사항)", value="models/finetuned_model", help="파인튜닝된 모델의 경로를 입력하세요")
    
    # 데이터베이스 리셋 버튼
    if st.button("🔄 데이터베이스 초기화", help="오류 발생시 사용"):
        system.reset_database()
        st.experimental_rerun()
    
    st.markdown("---")
    
    st.header("📋 메뉴")
    menu = st.radio("", ["🩺 AI 상담", "📊 상담 이력", "💊 영양제 목록", "ℹ️ 사용 가이드"])
    
    st.markdown("---")
    
    # 모델 상태 표시
    if check_model_setup():
        st.success("✅ 파인튜닝된 모델 로드됨")
    else:
        st.info("ℹ️ 기본 분석 모드")
    
    st.markdown("---")
    st.markdown("### 💡 주요 기능")
    st.markdown("""
    - 🔍 **증상 분석**: AI가 증상을 의학적으로 분석
    - 🚨 **응급상황 감지**: 위험한 증상 자동 감지
    - 💊 **맞춤 영양제**: 증상별 영양제 추천
    - 📱 **간편한 인터페이스**: 누구나 쉽게 사용
    """)

# 메인 컨텐츠
if menu == "🩺 AI 상담":
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="pet-card">', unsafe_allow_html=True)
        st.subheader("🐾 반려동물 정보")
        
        pet_name = st.text_input("이름", placeholder="예: 멍멍이")
        pet_type = st.selectbox("종류", ["개", "고양이", "기타"])
        
        col_age, col_weight = st.columns(2)
        with col_age:
            pet_age = st.number_input("나이 (세)", min_value=0, max_value=30, value=3)
        with col_weight:
            pet_weight = st.number_input("몸무게 (kg)", min_value=0.1, max_value=100.0, value=5.0, step=0.1)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="pet-card">', unsafe_allow_html=True)
        st.subheader("🩺 증상 설명")
        symptoms = st.text_area(
            "어떤 증상을 보이나요?", 
            placeholder="""예시:
- 며칠 전부터 절뚝거리고 있어요
- 계단 오르내리기를 힘들어해요
- 평소보다 활동량이 줄었어요
- 가끔 다리를 들고 걸어요""",
            height=150
        )
        
        # 추가 정보
        with st.expander("📝 추가 정보 (선택사항)"):
            symptom_duration = st.selectbox(
                "증상 지속 기간",
                ["1일 미만", "1-3일", "1주일", "2주 이상", "1달 이상"]
            )
            
            current_medication = st.text_input("현재 복용 중인 약물", placeholder="없음")
            
            previous_illness = st.text_area("기존 병력", placeholder="특이사항 없음", height=80)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        analyze_button = st.button("🔍 AI 건강 분석 시작", use_container_width=True, type="primary")
    
    with col2:
        if analyze_button and pet_name and symptoms:
            with st.spinner("🤖 AI가 반려동물의 상태를 분석하고 있습니다..."):
                try:
                    # LangGraph 실행
                    initial_state = {
                        "pet_info": {
                            "name": pet_name,
                            "type": pet_type,
                            "age": pet_age,
                            "weight": pet_weight
                        },
                        "symptoms": symptoms,
                        "health_analysis": "",
                        "supplement_recommendations": [],
                        "consultation_id": ""
                    }
                    
                    result = system.app.invoke(initial_state)
                    
                    # 건강 분석 결과 표시
                    st.markdown('<div class="health-status">', unsafe_allow_html=True)
                    st.markdown("### 📊 AI 건강 분석 결과")
                    st.markdown(result["health_analysis"])
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 응급상황이 아닌 경우에만 영양제 추천 표시
                    if result["supplement_recommendations"]:
                        st.markdown("### 💊 맞춤 영양제 추천")
                        
                        for i, supplement in enumerate(result["supplement_recommendations"], 1):
                            st.markdown('<div class="recommendation-box">', unsafe_allow_html=True)
                            
                            col_info, col_action = st.columns([3, 1])
                            
                            with col_info:
                                st.markdown(f"**{i}. {supplement['name']}** ({supplement['brand']})")
                                st.write(f"📂 **카테고리**: {supplement['category']}")
                                st.write(f"📝 **설명**: {supplement['description']}")
                                st.write(f"🧪 **주요 성분**: {supplement['ingredients']}")
                                
                                # 상세 정보 접기/펼치기
                                with st.expander("자세한 정보 보기"):
                                    st.write(f"**추천 대상**: {supplement['recommended_for']}")
                                    st.write(f"**복용법**: {supplement['dosage']}")
                                    st.write(f"**부작용**: {supplement['side_effects']}")
                                    st.write(f"**주의사항**: {supplement['contraindications']}")
                            
                            with col_action:
                                st.metric("⭐ 평점", f"{supplement['rating']}/5.0")
                                st.metric("💰 가격", f"₩{supplement['price']:,}")
                                if st.button(f"구매 정보", key=f"buy_{supplement['id']}"):
                                    st.info("실제 구매는 신뢰할 수 있는 온라인몰이나 동물병원을 이용해주세요.")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                    
                    # 상담 완료 메시지
                    st.success(f"✅ 상담이 완료되었습니다! (상담 ID: {result['consultation_id']})")
                    
                    # 추가 조치 안내
                    st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                    st.markdown("""
                    ### ⚠️ 중요 안내사항
                    
                    - 이 분석은 **참고용**이며 전문 수의사 진료를 대체할 수 없습니다
                    - 증상이 지속되거나 악화되면 **반드시 동물병원**에 방문하세요
                    - 영양제 복용 전 현재 복용 중인 약물과의 **상호작용을 확인**하세요
                    - 응급 상황으로 판단되는 경우 **즉시 응급 동물병원**에 연락하세요
                    
                    **24시간 응급 동물병원**: 지역 응급 동물병원 검색을 권장합니다.
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"분석 중 오류가 발생했습니다: {str(e)}")
                    st.info("시스템 관리자에게 문의하거나 잠시 후 다시 시도해주세요.")
        
        elif analyze_button:
            st.warning("반려동물 이름과 증상을 모두 입력해주세요.")

elif menu == "📊 상담 이력":
    st.subheader("📋 최근 상담 이력")
    
    try:
        history_df = system.get_consultation_history(20)
        
        if not history_df.empty:
            st.info(f"총 {len(history_df)}건의 상담 기록이 있습니다.")
            
            for idx, row in history_df.iterrows():
                with st.expander(f"🐾 {row['pet_name']} ({row['pet_type']}) - {row['timestamp']}"):
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("**📝 증상:**")
                        st.write(row['symptoms'])
                    
                    with col2:
                        st.write("**🔍 분석 결과:**")
                        st.write(row['health_analysis'][:200] + "..." if len(row['health_analysis']) > 200 else row['health_analysis'])
        else:
            st.info("아직 상담 이력이 없습니다. 첫 번째 AI 상담을 시작해보세요!")
            
    except Exception as e:
        st.error(f"상담 이력을 불러오는 중 오류가 발생했습니다: {str(e)}")

elif menu == "💊 영양제 목록":
    st.subheader("💊 영양제 카탈로그")
    
    try:
        conn = sqlite3.connect('pet_consultations.db', check_same_thread=False)
        supplements_df = pd.read_sql_query("SELECT * FROM supplements ORDER BY category, rating DESC", conn)
        conn.close()
        
        # 필터링 옵션
        col1, col2, col3 = st.columns(3)
        
        with col1:
            categories = ["전체"] + list(supplements_df['category'].unique())
            selected_category = st.selectbox("카테고리", categories)
        
        with col2:
            min_price, max_price = st.slider(
                "가격 범위 (원)", 
                min_value=int(supplements_df['price'].min()),
                max_value=int(supplements_df['price'].max()),
                value=(int(supplements_df['price'].min()), int(supplements_df['price'].max()))
            )
        
        with col3:
            min_rating = st.selectbox("최소 평점", [0.0, 3.0, 4.0, 4.5], index=0)
        
        # 필터 적용
        filtered_df = supplements_df.copy()
        
        if selected_category != "전체":
            filtered_df = filtered_df[filtered_df['category'] == selected_category]
        
        filtered_df = filtered_df[
            (filtered_df['price'] >= min_price) & 
            (filtered_df['price'] <= max_price) &
            (filtered_df['rating'] >= min_rating)
        ]
        
        st.info(f"{len(filtered_df)}개의 영양제가 검색되었습니다.")
        
        # 영양제 표시
        for idx, supplement in filtered_df.iterrows():
            with st.container():
                col1, col2, col3 = st.columns([2, 2, 1])
                
                with col1:
                    st.markdown(f"### {supplement['name']}")
                    st.write(f"**브랜드:** {supplement['brand']}")
                    st.write(f"**카테고리:** {supplement['category']}")
                    st.write(supplement['description'])
                
                with col2:
                    st.write(f"**주요 성분:** {supplement['ingredients']}")
                    st.write(f"**추천 대상:** {supplement['recommended_for']}")
                    st.write(f"**복용법:** {supplement['dosage']}")
                    
                    with st.expander("부작용 및 주의사항"):
                        st.write(f"**부작용:** {supplement.get('side_effects', '알려진 부작용 없음')}")
                        st.write(f"**금기사항:** {supplement.get('contraindications', '특별한 금기사항 없음')}")
                
                with col3:
                    st.metric("⭐ 평점", f"{supplement['rating']}/5.0")
                    st.metric("💰 가격", f"₩{supplement['price']:,}")
                    st.button("상세 정보", key=f"detail_{supplement['id']}")
                
                st.markdown("---")
                
    except Exception as e:
        st.error(f"영양제 목록을 불러오는 중 오류가 발생했습니다: {str(e)}")

elif menu == "ℹ️ 사용 가이드":
    st.subheader("📖 AI 펫닥터 사용 가이드")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🚀 시작하기", "💡 효과적인 사용법", "⚠️ 주의사항", "🔧 기술 정보"])
    
    with tab1:
        st.markdown("""
        ### 🚀 AI 펫닥터 시작하기
        
        #### 1단계: 반려동물 정보 입력
        - 이름, 종류(개/고양이), 나이, 체중을 정확히 입력하세요
        - 나이와 체중은 영양제 용량 계산에 중요합니다
        
        #### 2단계: 증상 상세 기술
        - 언제부터 증상이 시작되었는지
        - 어떤 상황에서 증상이 나타나는지  
        - 증상의 정도와 빈도
        - 평소와 다른 행동 변화
        
        #### 3단계: AI 분석 결과 확인
        - 건강 상태 분석 결과 꼼꼼히 읽기
        - 응급상황 여부 확인
        - 추천 영양제 정보 검토
        
        #### 4단계: 전문가 상담
        - AI 분석은 참고용입니다
        - 심각한 증상은 반드시 수의사 진료
        - 영양제 복용 전 전문가와 상의
        """)
    
    with tab2:
        st.markdown("""
        ### 💡 효과적인 사용법
        
        #### 📝 증상 기술 팁
        
        **좋은 예시:**
        > "3일 전부터 왼쪽 뒷다리를 절뚝거리기 시작했어요. 
        > 평소 좋아하던 계단 오르기를 거부하고, 
        > 산책 시간도 평소 30분에서 10분으로 줄었어요.
        > 만지면 아픈 듯 소리를 내기도 합니다."
        
        **피해야 할 예시:**
        > "다리가 아픈 것 같아요"
        
        #### 🎯 카테고리별 주요 키워드
        
        **관절 문제**: 절뚝거림, 계단, 점프, 활동량 감소, 다리 들기
        **소화기 문제**: 구토, 설사, 식욕부진, 복부팽만, 변비  
        **피부 문제**: 가려움, 긁기, 털빠짐, 발진, 붉어짐
        **호흡기 문제**: 기침, 숨가쁨, 호흡곤란, 콧물
        **행동 변화**: 무기력, 숨기, 공격성, 불안
        """)
    
    with tab3:
        st.markdown("""
        ### ⚠️ 중요 주의사항
        
        #### 🚨 즉시 응급실로 가야 하는 증상
        - 의식을 잃거나 경련을 일으킴
        - 심한 호흡곤란 (헐떡임이 멈추지 않음)
        - 피를 토하거나 혈변을 봄
        - 복부가 심하게 팽창함 (위염전 의심)
        - 체온이 41도 이상 또는 35도 이하
        - 잇몸이 창백하거나 푸른빛을 띔
        
        #### 💊 영양제 복용 주의사항
        - 현재 복용 중인 약물과의 상호작용 확인
        - 알레르기 반응 주의 깊게 관찰
        - 권장량을 초과하여 복용하지 않기
        - 증상 악화시 즉시 중단하고 수의사 상담
        
        #### 🔒 개인정보 보호
        - 상담 기록은 로컬에 저장됩니다
        - 개인 식별 정보는 수집하지 않습니다
        - 필요시 브라우저 쿠키를 삭제하여 기록 제거 가능
        """)
    
    with tab4:
        st.markdown("""
        ### 🔧 기술 정보
        
        #### 🤖 AI 모델 정보
        - **기본 모드**: 규칙 기반 분석 + 수의학 지식베이스
        - **고급 모드**: OpenAI GPT-3.5/4.0 + RAG (API 키 필요)
        - **데이터베이스**: SQLite (로컬 저장)
        - **벡터 검색**: FAISS (의학 지식 검색)
        
        #### 📊 데이터 소스
        - 수의학 교과서 및 논문
        - 대한수의사회 가이드라인  
        - 국제 수의학 저널
        - 영양제 제조사 공식 자료
        
        #### 🔄 업데이트 정보
        - 영양제 데이터: 주 1회 자동 업데이트
        - 의학 지식: 월 1회 전문가 검토
        - 시스템 개선: 사용자 피드백 반영
        
        #### 🛠️ 시스템 요구사항
        - 인터넷 연결 (API 사용시)
        - 모던 웹 브라우저 (Chrome, Firefox, Safari)
        - JavaScript 활성화 필수
        """)

# 푸터
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p><strong>⚠️ 의료 면책 조항</strong></p>
    <p>이 AI 펫닥터는 정보 제공 목적으로만 사용되며, 전문 수의사의 진료나 조언을 대체하지 않습니다.</p>
    <p>반려동물의 건강에 대한 모든 결정은 자격을 갖춘 수의사와 상의하시기 바랍니다.</p>
    <hr style='margin: 1rem 0; border: none; border-top: 1px solid #eee;'>
    <p>💝 Made with ❤️ for our furry friends | 🏥 Always consult your veterinarian</p>
</div>
""", unsafe_allow_html=True)