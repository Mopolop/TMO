# ============================================================
# Streamlit веб-приложение
# НИРС: Предсказание региона мира по характеристикам страны
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

# ─── Настройка страницы ───────────────────────────────────────
st.set_page_config(
    page_title="GeoRegion Predictor",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Предсказание региона мира по характеристикам страны")
st.markdown("**НИРС по дисциплине «Технологии машинного обучения»**")
st.divider()

# ─── Загрузка и подготовка данных ────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv('world-data-2023.csv', encoding='latin-1')
    df.loc[df['Abbreviation'] == 'ST', 'Country'] = 'Sao Tome and Principe'

    region_map = {
        'Afghanistan':'Asia','Albania':'Europe','Algeria':'Africa','Andorra':'Europe',
        'Angola':'Africa','Antigua and Barbuda':'North & Central America',
        'Argentina':'South America','Armenia':'Europe','Australia':'Oceania',
        'Austria':'Europe','Azerbaijan':'Europe','The Bahamas':'North & Central America',
        'Bahrain':'Asia','Bangladesh':'Asia','Barbados':'North & Central America',
        'Belarus':'Europe','Belgium':'Europe','Belize':'North & Central America',
        'Benin':'Africa','Bhutan':'Asia','Bolivia':'South America',
        'Bosnia and Herzegovina':'Europe','Botswana':'Africa','Brazil':'South America',
        'Brunei':'Asia','Bulgaria':'Europe','Burkina Faso':'Africa','Burundi':'Africa',
        'Ivory Coast':'Africa','Cape Verde':'Africa','Cambodia':'Asia','Cameroon':'Africa',
        'Canada':'North & Central America','Central African Republic':'Africa','Chad':'Africa',
        'Chile':'South America','China':'Asia','Colombia':'South America','Comoros':'Africa',
        'Republic of the Congo':'Africa','Costa Rica':'North & Central America',
        'Croatia':'Europe','Cuba':'North & Central America','Cyprus':'Europe',
        'Czech Republic':'Europe','Democratic Republic of the Congo':'Africa',
        'Denmark':'Europe','Djibouti':'Africa','Dominica':'North & Central America',
        'Dominican Republic':'North & Central America','Ecuador':'South America',
        'Egypt':'Africa','El Salvador':'North & Central America','Equatorial Guinea':'Africa',
        'Eritrea':'Africa','Estonia':'Europe','Eswatini':'Africa','Ethiopia':'Africa',
        'Fiji':'Oceania','Finland':'Europe','France':'Europe','Gabon':'Africa',
        'The Gambia':'Africa','Georgia':'Europe','Germany':'Europe','Ghana':'Africa',
        'Greece':'Europe','Grenada':'North & Central America','Guatemala':'North & Central America',
        'Guinea':'Africa','Guinea-Bissau':'Africa','Guyana':'South America',
        'Haiti':'North & Central America','Vatican City':'Europe','Honduras':'North & Central America',
        'Hungary':'Europe','Iceland':'Europe','India':'Asia','Indonesia':'Asia',
        'Iran':'Asia','Iraq':'Asia','Republic of Ireland':'Europe','Israel':'Asia',
        'Italy':'Europe','Jamaica':'North & Central America','Japan':'Asia','Jordan':'Asia',
        'Kazakhstan':'Europe','Kenya':'Africa','Kiribati':'Oceania','Kuwait':'Asia',
        'Kyrgyzstan':'Asia','Laos':'Asia','Latvia':'Europe','Lebanon':'Asia',
        'Lesotho':'Africa','Liberia':'Africa','Libya':'Africa','Liechtenstein':'Europe',
        'Lithuania':'Europe','Luxembourg':'Europe','Madagascar':'Africa','Malawi':'Africa',
        'Malaysia':'Asia','Maldives':'Asia','Mali':'Africa','Malta':'Europe',
        'Marshall Islands':'Oceania','Mauritania':'Africa','Mauritius':'Africa',
        'Mexico':'North & Central America','Federated States of Micronesia':'Oceania',
        'Moldova':'Europe','Monaco':'Europe','Mongolia':'Asia','Montenegro':'Europe',
        'Morocco':'Africa','Mozambique':'Africa','Myanmar':'Asia','Namibia':'Africa',
        'Nauru':'Oceania','Nepal':'Asia','Netherlands':'Europe','New Zealand':'Oceania',
        'Nicaragua':'North & Central America','Niger':'Africa','Nigeria':'Africa',
        'North Korea':'Asia','North Macedonia':'Europe','Norway':'Europe','Oman':'Asia',
        'Pakistan':'Asia','Palau':'Oceania','Palestinian National Authority':'Asia',
        'Panama':'North & Central America','Papua New Guinea':'Oceania',
        'Paraguay':'South America','Peru':'South America','Philippines':'Asia',
        'Poland':'Europe','Portugal':'Europe','Qatar':'Asia','Romania':'Europe',
        'Russia':'Europe','Rwanda':'Africa','Saint Kitts and Nevis':'North & Central America',
        'Saint Lucia':'North & Central America',
        'Saint Vincent and the Grenadines':'North & Central America',
        'Samoa':'Oceania','San Marino':'Europe','Sao Tome and Principe':'Africa',
        'Saudi Arabia':'Asia','Senegal':'Africa','Serbia':'Europe','Seychelles':'Africa',
        'Sierra Leone':'Africa','Singapore':'Asia','Slovakia':'Europe','Slovenia':'Europe',
        'Solomon Islands':'Oceania','Somalia':'Africa','South Africa':'Africa',
        'South Korea':'Asia','South Sudan':'Africa','Spain':'Europe','Sri Lanka':'Asia',
        'Sudan':'Africa','Suriname':'South America','Sweden':'Europe','Switzerland':'Europe',
        'Syria':'Asia','Tajikistan':'Asia','Tanzania':'Africa','Thailand':'Asia',
        'East Timor':'Asia','Togo':'Africa','Tonga':'Oceania',
        'Trinidad and Tobago':'North & Central America','Tunisia':'Africa','Turkey':'Europe',
        'Turkmenistan':'Asia','Tuvalu':'Oceania','Uganda':'Africa','Ukraine':'Europe',
        'United Arab Emirates':'Asia','United Kingdom':'Europe','United States':'North & Central America',
        'Uruguay':'South America','Uzbekistan':'Asia','Vanuatu':'Oceania',
        'Venezuela':'South America','Vietnam':'Asia','Yemen':'Asia','Zambia':'Africa',
        'Zimbabwe':'Africa',
    }
    df['Region'] = df['Country'].map(region_map)

    def clean_numeric(series):
        return (series.astype(str)
                .str.replace(r'[\$,%\s]', '', regex=True)
                .str.replace(',', '', regex=False)
                .replace('nan', np.nan)
                .pipe(pd.to_numeric, errors='coerce'))

    cols_to_clean = [
        'Density\n(P/Km2)', 'Agricultural Land( %)', 'Land Area(Km2)',
        'Armed Forces size', 'Co2-Emissions', 'CPI', 'CPI Change (%)',
        'Forested Area (%)', 'Gasoline Price', 'GDP',
        'Gross primary education enrollment (%)',
        'Gross tertiary education enrollment (%)',
        'Minimum wage', 'Out of pocket health expenditure',
        'Population', 'Population: Labor force participation (%)',
        'Tax revenue (%)', 'Total tax rate', 'Unemployment rate', 'Urban_population'
    ]
    for col in cols_to_clean:
        df[col] = clean_numeric(df[col])

    drop_cols = ['Abbreviation', 'Calling Code', 'Capital/Major City',
                 'Currency-Code', 'Largest city', 'Official language', 'Minimum wage']
    df.drop(columns=drop_cols, inplace=True)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ['Latitude', 'Longitude']]
    df[numeric_cols] = df.groupby('Region')[numeric_cols].transform(
        lambda x: x.fillna(x.median()))
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    return df

df = load_data()

feature_cols = [
    'Density\n(P/Km2)', 'Agricultural Land( %)', 'Land Area(Km2)',
    'Armed Forces size', 'Birth Rate', 'Co2-Emissions', 'CPI',
    'CPI Change (%)', 'Fertility Rate', 'Forested Area (%)',
    'Gasoline Price', 'GDP', 'Gross primary education enrollment (%)',
    'Gross tertiary education enrollment (%)', 'Infant mortality',
    'Life expectancy', 'Maternal mortality ratio',
    'Out of pocket health expenditure', 'Physicians per thousand',
    'Population', 'Population: Labor force participation (%)',
    'Tax revenue (%)', 'Total tax rate', 'Unemployment rate', 'Urban_population'
]

X = df[feature_cols]
le = LabelEncoder()
y = le.fit_transform(df['Region'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y)

region_colors = {
    'Africa': '#55A868', 'Asia': '#DD8452', 'Europe': '#4C72B0',
    'North & Central America': '#C44E52', 'Oceania': '#937860', 'South America': '#8172B2'
}

# ─── Боковая панель ───────────────────────────────────────────
st.sidebar.header("⚙️ Настройка модели")
model_choice = st.sidebar.selectbox(
    "Выберите модель",
    ["Random Forest", "Gradient Boosting"]
)

st.sidebar.subheader("Гиперпараметры")

if model_choice == "Random Forest":
    n_estimators = st.sidebar.slider("Количество деревьев (n_estimators)", 10, 300, 100, 10)
    max_depth = st.sidebar.select_slider("Максимальная глубина (max_depth)",
                                          options=[3, 5, 7, 10, 15, 20, 'None'], value=10)
    min_samples_split = st.sidebar.slider("Мин. выборка для разбиения (min_samples_split)", 2, 20, 2)
    max_depth_val = None if max_depth == 'None' else int(max_depth)
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth_val,
        min_samples_split=min_samples_split,
        random_state=42
    )
else:
    n_estimators = st.sidebar.slider("Количество деревьев (n_estimators)", 10, 300, 100, 10)
    learning_rate = st.sidebar.select_slider("Скорость обучения (learning_rate)",
                                              options=[0.01, 0.05, 0.1, 0.2, 0.3], value=0.1)
    max_depth = st.sidebar.slider("Максимальная глубина (max_depth)", 2, 10, 3)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=42
    )

# ─── Обучение модели ─────────────────────────────────────────
with st.spinner("Обучение модели..."):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='macro')
    prec = f1_score(y_test, y_pred, average='macro')

# ─── Метрики ────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
col1.metric("🎯 Accuracy", f"{acc:.4f}")
col2.metric("📊 F1-score (macro)", f"{f1:.4f}")
col3.metric("📌 Precision (macro)", f"{prec:.4f}")

st.divider()

# ─── Графики ────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Матрица ошибок",
    "🌳 Важность признаков",
    "🗺️ Распределение регионов",
    "🔍 Предсказание страны"
])

with tab1:
    st.subheader(f"Матрица ошибок — {model_choice}")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_,
                linewidths=0.5, linecolor='white', ax=ax)
    ax.set_xlabel('Предсказанный регион', fontsize=11)
    ax.set_ylabel('Истинный регион', fontsize=11)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.caption("Диагональные элементы — верно классифицированные страны.")

with tab2:
    st.subheader(f"Важность признаков — {model_choice}")
    if hasattr(model, 'feature_importances_'):
        feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(9, 8))
        colors = ['#4C72B0' if v > feat_imp.median() else '#AEC6CF' for v in feat_imp.values]
        feat_imp.plot(kind='barh', ax=ax, color=colors)
        ax.set_title('Важность признаков', fontweight='bold')
        ax.axvline(feat_imp.median(), color='red', linestyle='--', alpha=0.7, label='Медиана')
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        top3 = feat_imp.nlargest(3)
        st.info(f"**Топ-3 признака:** {', '.join(top3.index.tolist())}")

with tab3:
    st.subheader("Распределение стран по регионам")
    col_a, col_b = st.columns(2)
    with col_a:
        counts = df['Region'].value_counts()
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(counts.index, counts.values,
                      color=[region_colors[r] for r in counts.index], edgecolor='white')
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(val), ha='center', fontweight='bold')
        plt.xticks(rotation=25, ha='right')
        ax.set_ylabel('Количество стран')
        ax.set_title('Стран по регионам')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col_b:
        fig, ax = plt.subplots(figsize=(7, 5))
        for region, color in region_colors.items():
            subset = df[df['Region'] == region]
            ax.scatter(subset['Infant mortality'], subset['Life expectancy'],
                       c=color, label=region, s=60, alpha=0.8, edgecolors='white')
        ax.set_xlabel('Младенческая смертность')
        ax.set_ylabel('Продолжительность жизни')
        ax.set_title('Жизнь vs Смертность')
        ax.legend(fontsize=7, loc='lower left')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab4:
    st.subheader("🔍 Предсказать регион для страны")
    st.markdown("Выберите страну из датасета и модель предскажет её регион:")
    selected_country = st.selectbox("Страна:", df['Country'].sort_values().tolist())
    country_data = df[df['Country'] == selected_country][feature_cols]
    country_scaled = scaler.transform(country_data)
    pred_region_idx = model.predict(country_scaled)[0]
    pred_region = le.inverse_transform([pred_region_idx])[0]
    true_region = df[df['Country'] == selected_country]['Region'].values[0]

    col_pred, col_true = st.columns(2)
    col_pred.metric("🤖 Предсказанный регион", pred_region)
    col_true.metric("✅ Истинный регион", true_region)
    if pred_region == true_region:
        st.success("✅ Модель предсказала верно!")
    else:
        st.error(f"❌ Ошибка: предсказано '{pred_region}', истинное значение '{true_region}'")

    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(country_scaled)[0]
        proba_df = pd.DataFrame({
            'Регион': le.classes_,
            'Вероятность': proba
        }).sort_values('Вероятность', ascending=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = [region_colors.get(r, '#999999') for r in proba_df['Регион']]
        ax.barh(proba_df['Регион'], proba_df['Вероятность'], color=colors, edgecolor='white')
        ax.set_xlabel('Вероятность')
        ax.set_title(f'Вероятности регионов для: {selected_country}')
        ax.set_xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

st.divider()
st.caption("НИРС | Технологии машинного обучения | Global Country Information Dataset 2023")
