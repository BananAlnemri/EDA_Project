import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import calendar
from sklearn.linear_model import LinearRegression


# 1. Page Configuration - هنا نضع علم السعودية 🇸🇦 مكان التاج
st.set_page_config(
    page_title="Saudi Tourism Intelligence", 
    page_icon="🇸🇦", 
    layout="wide"
)


# ألوان موسم الرياض للرسومات
riyadh_season_colors = ['#FF4B4B', '#FFD700', '#1E90FF', '#32CD32', '#FF69B4', '#8A2BE2', '#00CED1']

# CSS لتوحيد لون الخلفية في الموقع والسايد بار والرسومات
st.markdown("""
    <style>
    .stApp, [data-testid="stSidebar"] {
        background-color: #0E1117 !important; /* أسود كربوني فخم */
    }
    div[data-testid="metric-container"] {
        background-color: #161B22; /* رمادي غامق جداً للبطاقات */
        border: 1px solid #30363D;
        border-radius: 10px;
    }
    h1, h2, h3 { color: #FFFFFF !important; }
    </style>
    """, unsafe_allow_html=True)
    

@st.cache_data
def load_and_clean():
    df = pd.read_csv('tourism_with_temps.csv')
    df['date'] = pd.to_datetime(df[['year', 'month']].assign(day=1)) # إنشاء عمود تاريخ من السنة والشهر
    df['temp_diff'] = df['destination_temp'] - df['origin_temp'] # حساب فرق درجات الحرارة
    df['spend_per_trip'] = df['spendSAR'] / df['trips']    # حساب متوسط الإنفاق لكل رحلة
    return df

df = load_and_clean()
from sklearn.linear_model import LinearRegression

@st.cache_resource
def train_advanced_temp_model(df):
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import LabelEncoder
    
    # تنظيف وتجهيز البيانات
    t_df = df.dropna(subset=['destination_temp', 'destinationProvinceNameEn']).copy()
    le_region = LabelEncoder()
    t_df['region_encoded'] = le_region.fit_transform(t_df['destinationProvinceNameEn'])
    
    # الموديل سيتعلم من (السنة، الشهر، والمنطقة)
    X = t_df[['year', 'month', 'region_encoded']].values
    y = t_df['destination_temp'].values
    
    model = LinearRegression().fit(X, y)
    return model, le_region

temp_model, region_encoder = train_advanced_temp_model(df)
# --- Sidebar (Ministry Logo & Filters) ---
with st.sidebar:
    try:
        st.image("MOT.png", use_container_width=True)
    except:
        st.error("⚠️ لم يتم العثور على ملف MOT.png في المجلد")
    
    st.markdown("---")
    st.title("Dashboard Filters 🔍") #To allow the user to choose the year they want to see the data for
    selected_years = st.multiselect("Select Year(s)", options=sorted(df['year'].unique()), default=df['year'].unique())
    st.markdown("---")
    st.subheader("🔮 Advanced Weather Predictor")
    
    # المدخلات التي طلبتِها
    p_year_t = st.number_input("Select Year", 2024, 2030, 2025, key="temp_year")
    p_month_t = st.slider("Select Month", 1, 12, 6, key="temp_month")
    p_region_t = st.selectbox("Select Destination", options=region_encoder.classes_, key="temp_region")

    if st.button("Predict Future Weather"):
        # تحويل المنطقة لرقم يفهمه الموديل
        region_idx = region_encoder.transform([p_region_t])[0]
        
        # التنبؤ
        predicted_temp = temp_model.predict([[p_year_t, p_month_t, region_idx]])[0]
        
        st.metric(f"Expected Temp in {p_region_t}", f"{predicted_temp:.1f}°C")
        st.caption(f"Forecast for {calendar.month_name[p_month_t]} {p_year_t}")
filtered_df = df[df['year'].isin(selected_years)]

# --- MAIN LAYOUT ---
st.title("Saudi Arabia Tourism Strategic Analysis 2018-2023")
st.markdown("---")

# --- الإحصائيات العامة (KPIs) في الأعلى ---
m1, m2, m3, m4 = st.columns(4) # Using the Univariate Analysis to show the total trips, total revenue, average spend per trip, and average temperature gap
m1.metric("Total Trips", f"{filtered_df['trips'].sum():,.0f}")
m2.metric("Total Revenue", f"SAR {filtered_df['spendSAR'].sum():,.0f}")
m3.metric("Avg Spend/Trip", f"{filtered_df['spend_per_trip'].mean():,.0f} SAR")
m4.metric("Avg Temp Gap", f"{filtered_df['temp_diff'].mean():.1f}°C")

st.markdown("---")

# --- SECTION 1: PURPOSE OF VISIT ---
st.header("1. Purpose of Visit & Spending Behavior")
col1, col2 = st.columns(2)
# Bivariate Analysis: تحليل ثنائي المتغيرات بين غرض الزيارة والإنفاق، بالإضافة إلى توزيع الرحلات حسب غرض الزيارة.
with col1:
    purpose_data = filtered_df.groupby('visitPurposeEn')['trips'].sum().reset_index()
    fig_donut = px.pie(purpose_data, values='trips', names='visitPurposeEn', 
                       title="Distribution of Trip Purposes", hole=0.5,
                       color_discrete_sequence=riyadh_season_colors)
    fig_donut.update_traces(textinfo='percent+label')
    # توحيد خلفية الرسمة مع خلفية الموقع
    fig_donut.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_donut, use_container_width=True)

with col2:
    purpose_spend = filtered_df.groupby('visitPurposeEn')['spendSAR'].mean().sort_values(ascending=False).reset_index()
    fig_purpose_bar = px.bar(purpose_spend, x='visitPurposeEn', y='spendSAR', 
                             title="Average Spending by Purpose (SAR)", color='visitPurposeEn',
                             color_discrete_sequence=riyadh_season_colors)
    fig_purpose_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_purpose_bar, use_container_width=True)

# --- SECTION 2: DESTINATION RANKINGS ---
# Outlier Analysis
#  لرؤية الوجهات التي قد تكون خارج الترتيب الطبيعي من حيث عدد الرحلات أو الإنفاق، 
# مما يساعد في فهم ما إذا كانت هناك وجهات معينة تجذب بشكل غير متناسب أو إذا كان هناك تأثير موسمي قوي على وجهات معينة.
st.header("2. Destination Rankings (Comparative Views)")
t1, t2, t3 = st.tabs(["All Regions", "Excluding Makkah", "Excluding Makkah & Riyadh"])
with t1:
    st.subheader("Market Share: All Regions")
    top_all = filtered_df.groupby('destinationProvinceNameEn')['trips'].sum().sort_values(ascending=False).reset_index()
    fig_all = px.bar(top_all, x='destinationProvinceNameEn', y='trips', color='destinationProvinceNameEn',
                     color_discrete_sequence=riyadh_season_colors)
    fig_all.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_all, use_container_width=True)

with t2:
    no_makk = filtered_df[filtered_df['destinationProvinceNameEn'] != 'Makkah']
    top_no_m = no_makk.groupby('destinationProvinceNameEn')['trips'].sum().sort_values(ascending=False).reset_index()
    fig_no_m = px.bar(top_no_m, x='destinationProvinceNameEn', y='trips', color='destinationProvinceNameEn',
                      color_discrete_sequence=riyadh_season_colors)
    fig_no_m.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_no_m, use_container_width=True)

with t3:
    spec = filtered_df[~filtered_df['destinationProvinceNameEn'].isin(['Makkah', 'Riyadh'])]
    top_spec = spec.groupby('destinationProvinceNameEn')['trips'].sum().sort_values(ascending=False).reset_index()
    fig_spec = px.bar(top_spec, x='destinationProvinceNameEn', y='trips', color='destinationProvinceNameEn',
                       color_discrete_sequence=riyadh_season_colors)
    fig_spec.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_spec, use_container_width=True)

# --- SECTION 3: CLIMATE DYNAMICS & REGIONAL SPENDING ---
# Correlation Analysis to explore the relationship between temperature differences and spending patterns, as well as to identify which destinations are most affected by climate dynamics in terms of traveler behavior and expenditure.
st.header("3. Climate Dynamics & Regional Spending")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Origin vs. Destination Temperatures")
    temp_trend = filtered_df.groupby('month')[['origin_temp', 'destination_temp']].mean().reset_index()
    temp_trend['month_name'] = temp_trend['month'].apply(lambda x: calendar.month_name[x])
    fig_temp_line = px.line(temp_trend, x='month_name', y=['origin_temp', 'destination_temp'], 
                            title="Average Temperature Trends", markers=True,
                            color_discrete_sequence=['#FF4B4B', '#1E90FF'])
    fig_temp_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_temp_line, use_container_width=True)

with col_b:
    st.subheader("Average Spending per Destination")
    dest_spend = filtered_df.groupby('destinationProvinceNameEn')['spend_per_trip'].mean().sort_values(ascending=False).reset_index()
    fig_dest_spend = px.bar(dest_spend, x='spend_per_trip', y='destinationProvinceNameEn', 
                            orientation='h', title="Avg Spend per Trip", color='destinationProvinceNameEn',
                            color_discrete_sequence=riyadh_season_colors)
    fig_dest_spend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_dest_spend, use_container_width=True)

# --- SECTION 3.1: HISTORICAL SPENDING TRENDS ---
# Trend Analysis to examine how spending patterns have evolved over the years for different destinations,
#  which can provide insights into the growth or decline of certain regions and help identify emerging trends in traveler behavior and preferences.
st.header("3.1 Destination Spending Trends (2018-2023)")
selected_dest = st.selectbox("Select Destination to Analyze Trend", options=sorted(df['destinationProvinceNameEn'].unique()))

# تجهيز البيانات: حساب متوسط الصرف لكل شهر وسنة للوجهة المختارة
trend_df = df[df['destinationProvinceNameEn'] == selected_dest].groupby(['year', 'month'])['spend_per_trip'].mean().reset_index()
trend_df['month_name'] = trend_df['month'].apply(lambda x: calendar.month_name[x])

fig_trend = px.line(trend_df, x='month_name', y='spend_per_trip', color='year',
                    title=f"Spending Trend in {selected_dest} by Month Across Years",
                    markers=True, category_orders={"month_name": list(calendar.month_name)[1:]},
                    color_discrete_sequence=px.colors.qualitative.Vivid)

fig_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white", hovermode="x unified")
st.plotly_chart(fig_trend, use_container_width=True)


# --- SECTION 4: HEAT ESCAPE & MONTHLY TRENDS ---
# Variance Analysis to investigate the "heat escape" phenomenon, where travelers may prefer cooler destinations during hotter months, 
# and to analyze monthly spending patterns to identify peak periods and potential seasonal effects on tourism behavior.
st.header("4. Heat Escape & Monthly Patterns")
col3, col4 = st.columns(2)

with col3:
    st.subheader("The 'Heat Escape' Effect")
    fig_temp = px.scatter(filtered_df.sample(min(1000, len(filtered_df))), x='temp_diff', y='trips', 
                          color='destinationProvinceNameEn', title="Trips vs. Temperature Gap",
                          color_discrete_sequence=riyadh_season_colors)
    fig_temp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_temp, use_container_width=True)
with col4:
    st.subheader("Spending Ranked by Month")
    month_data = filtered_df.groupby('month')['spendSAR'].mean().sort_values(ascending=False).reset_index()
    month_data['month_name'] = month_data['month'].apply(lambda x: calendar.month_name[x])
    fig_month = px.bar(month_data, x='month_name', y='spendSAR', title="Highest Spending Months",
                       color='month_name', color_discrete_sequence=riyadh_season_colors)
    fig_month.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_month, use_container_width=True)

# --- SECTION 5: SEASONS ANALYSIS ---
st.header("5. Seasonal Impact: Riyadh & Jeddah")
col5, col6 = st.columns(2)

with col5:
    st.subheader("Riyadh: In-Season vs Off-Season")
    riyadh = filtered_df[filtered_df['destinationProvinceNameEn'] == 'Riyadh'].copy()
    riyadh['Season_Status'] = riyadh['month'].isin([10,11,12,1,2,3]).map({True: 'Riyadh Season', False: 'Off-Season'})
    fig_riyadh = px.box(riyadh, x='Season_Status', y='spendSAR', color='Season_Status',
                        color_discrete_sequence=['#FF4B4B', '#8A2BE2'])
    fig_riyadh.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_riyadh, use_container_width=True)

with col6:
    st.subheader("Jeddah Season vs Riyadh Season (Avg Spending)")
    j_season = filtered_df[(filtered_df['destinationProvinceNameEn'] == 'Makkah') & (filtered_df['month'].isin([6,7]))]
    r_season = filtered_df[(filtered_df['destinationProvinceNameEn'] == 'Riyadh') & (filtered_df['month'].isin([10,11,12,1,2,3]))]
    
    comp_df = pd.DataFrame({
        'Season': ['Jeddah Season', 'Riyadh Season'],
        'Avg Spending': [j_season['spendSAR'].mean(), r_season['spendSAR'].mean()]
    })
    fig_comp = px.bar(comp_df, x='Season', y='Avg Spending', color='Season',
                      color_discrete_sequence=['#00CED1', '#FF4B4B'])
    fig_comp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
    st.plotly_chart(fig_comp, use_container_width=True)

# --- SECTION 6: STRATEGIC EXECUTIVE INSIGHTS ---
st.divider()
st.header("💡 Strategic Executive Insights")
ins1, ins2 = st.columns(2)
with ins1:
    st.markdown(f"📌 Market Drivers: Religious tourism in Makkah is the main driver. Excluding it, Riyadh leads in business and leisure.")
with ins2:
    st.markdown(f"🌡 Climate Escape: Travelers seek cooler destinations in summer (e.g., Aseer). July is a peak spending month.")

st.write("Analysis Developed by: Banan Alnemri")