# -- coding: utf-8 --
import pandas as pd
import numpy as np
import streamlit as st
import joblib
import plotly.graph_objects as go 
import plotly.express as px

# ==========================================================
# 1. LOAD MODEL AND FIXED PARAMETERS
# ==========================================================

@st.cache_resource
def load_model():
    """Loads the pre-trained model (aris_model.pkl)."""
    try:
        model = joblib.load('aris_model.pkl')
        return model
    except FileNotFoundError:
        return None
    except Exception as e:
        return None

# ==========================================================
# 2. ARIS Index Calculation (Enhanced Sensitivity)
# ==========================================================

def calculate_aris_data(model, vibration, temp, corrosion_score, change_rate, flow_rate, lube_health, stress_in, rul_in):
    """Calculates ARIS Index and dynamic feature importance (CCP) with high sensitivity."""
    
    if model is None:
        return 0, 0, 0
        
    new_data = pd.DataFrame([[vibration, temp, corrosion_score, change_rate]], 
                            columns=['Vibration_X', 'Bearing_Temp', 'Historical_Corrosion_Score', 'Vibration_Change_Rate'])
    
    try:
        failure_prob = model.predict_proba(new_data)[0][1]
    except Exception:
        return 0, 0, 0
        
    risk_index = round(failure_prob * 120) 
    
    temp_normalized = (temp - 30) / 55 
    vib_normalized = (vibration - 1) / 24 
    
    extra_risk_points = 0
    if temp_normalized > 0.3:
        extra_risk_points += (temp_normalized - 0.3) * 150 
    if vib_normalized > 0.3:
        extra_risk_points += (vib_normalized - 0.3) * 150 
        
    
    # -----------------------------------------------------------
    # إضافة تأثير العوامل الجديدة (Flow Rate و Lube Health و Stress و RUL)
    # -----------------------------------------------------------
    
    # 1. تأثير معدل التدفق (Flow Rate)
    if flow_rate < 0.7:
        extra_risk_points += (0.7 - flow_rate) * 50 
    elif flow_rate > 1.1:
        extra_risk_points += (flow_rate - 1.1) * 75
        
    # 2. تأثير جودة التزييت (Lube Health)
    if lube_health < 0.4:
        extra_risk_points += (0.4 - lube_health) * 100 
    
    # 3. تأثير إجهاد الشد (Stress - القيمة العالية تزيد الخطر)
    if stress_in > 0.6:
        extra_risk_points += (stress_in - 0.6) * 80
        
    # 4. تأثير الزمن المتبقي للعمل (RUL - القيمة المنخفضة تزيد الخطر)
    if rul_in < 0.3:
        extra_risk_points += (0.3 - rul_in) * 120
    
    # -----------------------------------------------------------
        
    risk_index = risk_index + extra_risk_points
    risk_index = int(min(100, risk_index)) 
    
    corrosion_normalized = (corrosion_score - 0.1) / 0.9  
    rate_normalized = change_rate / 1.0
    
    corrosion_influence = 0.45 * corrosion_normalized
    rate_influence = 0.40 * rate_normalized
    temp_vib_influence = 0.15 * (temp_normalized + vib_normalized) / 2
    
    total_dynamic_influence = corrosion_influence + rate_influence + temp_vib_influence
    
    if total_dynamic_influence > 0.01:
        corrosion_contribution = round((corrosion_influence / total_dynamic_influence) * 100)
        rate_contribution = round((rate_influence / total_dynamic_influence) * 100)
    else:
        corrosion_contribution = 45 
        rate_contribution = 40
        
    total_contribution = corrosion_contribution + rate_contribution
    if total_contribution > 100:
         corrosion_contribution = round(corrosion_contribution * 100 / total_contribution)
         rate_contribution = round(rate_contribution * 100 / total_contribution)
    
    return risk_index, corrosion_contribution, rate_contribution

# ==========================================================
# 3. MOCK HISTORICAL DATA FUNCTION
# ==========================================================

def get_historical_data(risk_index):
    """Generates mock historical ARIS data based on the current index."""
    
    days = pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D')
    
    base_value = risk_index - 15  
    
    history = np.linspace(base_value, risk_index, 30)
    noise = np.random.normal(0, 5, 30) 
    
    historical_risks = np.clip(history + noise, 0, 100).round(0)
    
    historical_risks[-1] = risk_index
    
    df = pd.DataFrame({
        'التاريخ': days,
        'مؤشر ARIS التاريخي': historical_risks
    })
    
    return df

# ==========================================================
# 4. Risk Explanation and Recommendations (Arabic)
# ==========================================================

def explain_risk(risk_index, corr_contrib, rate_contrib, vibration_in, asset_id, flow_rate, lube_health, stress_in, rul_in):
    """Generates the Arabic risk explanation, strong recommendations, and XAI."""
    
    if corr_contrib > (100 - corr_contrib) or (corr_contrib > 50 and vibration_in < 10):
        dominant_factor = "التآكل التاريخي وسلامة المعدن"
        action_focus = "فحص بالموجات فوق الصوتية (UT) أو فحص ILI/OSI مفصل."
    else:
        dominant_factor = "الاهتزاز ودرجة الحرارة"
        action_focus = "إجراء موازنة دقيقة وتوسيط للعمود أو استبدال رولمان بلي (Bearing)."
        
    
    st.subheader(f"التقييم الحالي لـ: {asset_id}")
    st.header("تقرير مؤشر الخطر النهائي")
    
    delta_text = ('آمن' if risk_index < 20 else 
                  ('منخفض' if risk_index < 35 else 
                   ('متوسط' if risk_index < 50 else 
                    ('مرتفع' if risk_index < 80 else 'فشل وشيك'))))

    # عرض المؤشر الدائري (Gauge Chart)
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = risk_index,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "مؤشر ARIS الحالي", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#000033"},
            'steps': [
                {'range': [0, 35], 'color': "lightgreen"},
                {'range': [35, 50], 'color': "yellow"},
                {'range': [50, 80], 'color': "orange"},
                {'range': [80, 100], 'color': "red"}
            ],
            'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 80}}))

    fig_gauge.update_layout(font = {'color': "black", 'family': "Arial"}, autosize=False, width=400, height=250)
    
    st.markdown("---")
    st.subheader("تاريخ الخطر (آخر 30 يوماً)")
    
    historical_df = get_historical_data(risk_index)
    
    fig_line = px.line(
        historical_df, 
        x='التاريخ', 
        y='مؤشر ARIS التاريخي', 
        title='معدل تدهور مؤشر ARIS',
        labels={'مؤشر ARIS التاريخي': 'نسبة الخطر (%)', 'التاريخ': 'التاريخ'},
        markers=True
    )
    
    fig_line.add_hline(y=35, line_dash="dash", line_color="green", annotation_text="منطقة منخفضة/آمنة", annotation_position="top right")
    fig_line.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="حد التدخل المتوسط", annotation_position="top left")
    fig_line.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="حد الفشل الوشيك", annotation_position="top right")

    fig_line.update_traces(line_color='#0077b6', line_width=3) 
    fig_line.update_yaxes(range=[0, 100]) 
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.plotly_chart(fig_line, use_container_width=True)
    
    if risk_index < 20:
        st.balloons()
    
    if risk_index < 20:
        st.info(f"💡 التوصية (الحل): *لا يوجد خطر تشغيلي*. لا يتطلب تدخل فني.")
        
    elif 20 <= risk_index < 35:
        st.success(f"✅ مؤشر ARIS: {risk_index}% (المستوى: منخفض - بدأ التدهور).")
        st.info(f"💡 التوصية (الحل): *إصدار أمر عمل (WO) خلال 30 يوماً*. يجب على فريق الصيانة التجهيز لـ {action_focus} لتجنب ارتفاع الخطر المفاجئ.")
        
    elif 35 <= risk_index < 50:
        st.warning(f"⚠ مؤشر ARIS: {risk_index}% (المستوى: متوسط - يتطلب تدخل).")
        st.info(f"💡 التوصية (الحل): *إصدار أمر عمل عاجل خلال 7 أيام. الخطر ناتج عن **{dominant_factor}*. يجب إجراء {action_focus} فحصاً عميقاً وتحضير قطع الغيار مسبقاً.")
            
    elif 50 <= risk_index < 80:
        st.error(f"🚨 مؤشر ARIS: {risk_index}% (المستوى: مرتفع - يقترب من الحد النهائي).")
        st.error(f"🚨 التوصية (الحل): *إيقاف مخطط له خلال 48 ساعة. الخطر وشيك بسبب **{dominant_factor}*. يجب تنفيذ {action_focus} على الفور مع الإغلاق المؤقت لوحدة التشغيل. [حل مباشر يجنب خسائر الملايين].")
        
    else: # 80% and above (CRITICAL LIMIT)
        st.error(f"❌ مؤشر ARIS: {risk_index}% (المستوى: فشل وشيك - تجاوز الحد النهائي).")
        st.error("❌ التوصية (الحل): *إيقاف فوري وعاجل للمضخة*. يجب إزالة المكونات المتضررة (Overhaul) واستبدالها بالكامل.")
        
        
    # تم تعديل العنوان والتنسيق هنا
    st.subheader("تفسير القرار (XAI) - السبب الرئيسي والأسباب الفرعية:")
    
    vib_temp_contrib = 100 - corr_contrib - rate_contrib
    if vib_temp_contrib < 0: vib_temp_contrib = 0 
    
    st.markdown("""
    ---
    *السبب الرئيسي لارتفاع الخطر (الذي يحتاج إلى تدخل):*
    """)
    
    if corr_contrib >= vib_temp_contrib and corr_contrib >= rate_contrib:
        st.success(f"🥇 التآكل التاريخي وسلامة المعدن:** نسبة تأثير {corr_contrib}%. يُنصح بفحص UT عاجل لسلامة الهيكل.")
    elif vib_temp_contrib >= corr_contrib and vib_temp_contrib >= rate_contrib:
        st.success(f"🥇 الإجهاد الميكانيكي اللحظي (اهتزاز/حرارة):** نسبة تأثير {vib_temp_contrib}%. يُنصح بإعادة موازنة وتوسيط العمود فوراً.")
    else:
        st.success(f"🥇 معدل التدهور في الأداء:** نسبة تأثير {rate_contrib}%. يشير إلى تدهور سريع يتطلب مراجعة سجلات التشغيل.")

    st.markdown("""
    ---
    *الأسباب الفرعية المساهمة (تنبيهات المدخلات الإضافية):*
    """)
    
    # إضافة تفسير لدور المدخلات الإضافية
    messages = []
    if flow_rate < 0.7:
        messages.append(f"🔸 انخفاض معدل التدفق ({flow_rate}): أداء المضخة متدنٍ، يزيد الضغط على الأجزاء الداخلية.")
    if flow_rate > 1.1:
        messages.append(f"🔸 ارتفاع معدل التدفق ({flow_rate}): يزيد من خطر التآكل بالتعرية (Erosion).")
    if lube_health < 0.4:
        messages.append(f"🔴 جودة التزييت منخفضة ({lube_health}): يسرّع تآكل المحامل بشكل كبير. *تحتاج إلى تغيير الزيت فوراً.*")
    if stress_in > 0.6:
        messages.append(f"🔴 إجهاد الشد مرتفع ({stress_in}): يزيد من خطر التآكل الإجهادي (SCC) الهيكلي. *تحتاج إلى فحص غير إتلافي (NDT).*")
    if rul_in < 0.3:
        messages.append(f"🔸 الزمن المتبقي للعمل منخفض ({rul_in}): المعدة تقترب من نهاية العمر الافتراضي. *تحضير خطة استبدال.*")
        
    if messages:
        for msg in messages:
            st.markdown(f"- {msg}")
    else:
        st.info("- لا توجد ملاحظات حرجة من العوامل الميكانيكية والتشغيلية الإضافية.")
        
    st.markdown("---")
    st.info("نظام ARIS يقدم تقريراً مباشراً وواضحاً لفرق الصيانة والفحص لاتخاذ الإجراء الصحيح فوراً.")


# ==========================================================
# 5. Streamlit Main Interface
# ==========================================================

st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.title("💡 نظام ARIS: رؤية استباقية لمستقبل أصولك الصناعية")
st.caption("👈 *نظام مؤشر المخاطر القائم على الذكاء الاصطناعي (ARIS Index)*: حلول مبتكرة لمواجهة التآكل وتدهور المعدات، مقدمة برؤية استشرافية لتعزيز الكفاءة التشغيلية وتقليل التكاليف.")

# Load Model
model = load_model()

# ==========================================================
# 6. التعامل مع حالة فشل تحميل النموذج
# ==========================================================
if model is None:
    st.header("تطبيق ARIS Index غير متاح حالياً")
    st.warning("⚠ لا يمكن عرض مؤشر الخطر لأن *ملف النموذج (aris_model.pkl)* فشل في التحميل. يرجى التأكد من وجود الملف في مجلد المشروع الرئيسي.")
    st.stop()
    
# ==========================================================
# 7. واجهة المستخدم
# ==========================================================

# ----------------------------------------------------
# 7.1. SIDEBAR (CCP & Materials Info)
# ----------------------------------------------------
st.sidebar.header("نقاط التحكم الحرجة (CCP)")

asset_id = st.sidebar.selectbox(
    "اختر موقع المضخة:",
    options=[
        "1. مضخة الرياض الرئيسية - A (بيئة جافة)", 
        "2. مضخة الدمام الساحلية - B (بيئة بحرية)",
        "3. مضخة الجبيل الصناعية - C (بيئة كيميائية/أكثر حمضية)",
        "4. مضخة راس تنورة - D (بيئة بحرية/ملحية)"
    ],
    index=0,
    help="اختر المضخة لتحديد سياقها البيئي الذي يؤثر على التآكل التاريخي."
)

vibration_in = st.sidebar.slider("1. الاهتزاز الحالي (Vibration_X):", min_value=1.0, max_value=25.0, value=7.0, step=0.1, help="حد الخطر يبدأ عند 12.5 مم/ث.")
temp_in = st.sidebar.slider("2. حرارة العمود (Bearing_Temp):", min_value=30.0, max_value=85.0, value=55.0, step=0.1, help="حد الخطر يبدأ عند 75 درجة مئوية.")

corrosion_default = 0.2 
if "الدمام الساحلية" in asset_id:
    corrosion_default = 0.55 
elif "الجبيل الصناعية" in asset_id:
    corrosion_default = 0.60 
elif "راس تنورة" in asset_id:
    corrosion_default = 0.70 
    
corrosion_in = st.sidebar.slider("3. خطر الفحص التاريخي (ILI/OSI):", min_value=0.1, max_value=1.0, value=corrosion_default, step=0.01, help="يعكس جودة المعدن المتبقية (0.1 ممتاز، 1.0 فشل).")
change_rate_in = st.sidebar.slider("4. معدل التغير في الاهتزاز:", min_value=0.0, max_value=1.0, value=0.15, step=0.01, help="يعكس السرعة التي يتدهور بها الأداء (1.0 يعني تدهور سريع).")


# === عوامل التآكل الميكانيكي الإضافية ===
st.sidebar.markdown("---")
st.sidebar.subheader("عوامل التدهور الميكانيكي والتشغيلي")

flow_rate_in = st.sidebar.slider("5. معدل التدفق التشغيلي (نسبة):", min_value=0.5, max_value=1.5, value=1.0, step=0.05, help="معدل التدفق الحالي (1.0 = التدفق الأمثل).")

lube_health_in = st.sidebar.slider("6. صحة جودة التزييت (Lube Health):", min_value=0.0, max_value=1.0, value=0.8, step=0.1, help="نسبة جودة الزيت (1.0 ممتاز، 0.0 تالف).")

stress_in = st.sidebar.slider("7. إجهاد السطح/الشد (Tensile Stress):", min_value=0.0, max_value=1.0, value=0.4, step=0.1, help="مستوى الإجهاد الهيكلي (1.0 = إجهاد مرتفع جداً).")

rul_in = st.sidebar.slider("8. الزمن المتبقي للعمل (RUL):", min_value=0.0, max_value=1.0, value=0.7, step=0.1, help="الزمن المتوقع المتبقي لعمر المعدة (1.0 = جديد، 0.0 = انتهى العمر الافتراضي).")


# === المواد والطلاء في القائمة الجانبية (Sidebar) ===
st.sidebar.markdown("---")
st.sidebar.header("قاعدة بيانات المواد المرجعية")
st.sidebar.caption("لربط المدخلات بالخطر الأساسي")

material_options = [
    "الفولاذ الكربوني (CS)", "الفولاذ المقاوم للصدأ 316L", 
    "فولاذ دوبلكس (Duplex 2205)", "سبائك النيكل (Inconel 625)",
    "التيتانيوم", "البرونز", "LCS", "304 SS", "Super Duplex", "Hastelloy C276"
]
coating_options = [
    "إيبوكسي (Epoxy)", "بولي يوريثين (PU)", 
    "إيبوكسي مرتبط بالانصهار (FBE)", "طلاء السيراميك",
    "3LPE", "Zinc Primer", "Glass Flake", "Polyurea", "Phenolic", "Rubber Lining"
]

st.sidebar.selectbox("نوع المعدن المستخدم:", options=material_options, index=1, help="نوع المعدن يؤثر على مدى مقاومة التآكل (Corrosion Score).")
st.sidebar.selectbox("نوع الطلاء:", options=coating_options, index=2, help="يقلل الطلاء الفعال من خطر التآكل الداخلي.")


# ----------------------------------------------------
# 7.2. MAIN PAGE CONTENT (Results)
# ----------------------------------------------------

risk_result, corr_contrib, rate_contrib = calculate_aris_data(
    model, 
    vibration_in, 
    temp_in, 
    corrosion_in, 
    change_rate_in,
    flow_rate_in, 
    lube_health_in,
    stress_in, 
    rul_in 
)

explain_risk(
    risk_result, 
    corr_contrib, 
    rate_contrib, 
    vibration_in, 
    asset_id,
    flow_rate_in, 
    lube_health_in,
    stress_in, 
    rul_in 
)

st.markdown("---")
# تم تعديل نص نقطة العرض هنا
st.info("""
*💡 نقطة العرض (Pitch Point):* هذه العملية تظهر سرعة اتخاذ القرار للمشغلين والمهندسين. 
فقط أدخل قراءات المستشعرات، وخلال *أقل من ثانية* يمنحك نظام ARIS: 
1. *الخطر الحالي* (المؤشر الدائري).
2. *اتجاه التدهور* (الرسم البياني).
3. *التوصية بالإجراء الفوري* (مثل إيقاف خلال 48 ساعة).
4. *تفسير للسبب الجذري* (لتوجيه الفريق الفني الصحيح: فحص أم صيانة).
""")