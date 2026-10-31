import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore') 

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Rajasthan House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ─────────────────────────────────────────────
#  RAJASTHAN DATA
# ─────────────────────────────────────────────
RAJASTHAN_DATA = {
    "Jaipur": {
        "localities": ["Vaishali Nagar", "Mansarovar", "Malviya Nagar", "C-Scheme",
                       "Jagatpura", "Chitrakoot", "Tonk Road", "Ajmer Road",
                       "Sitapura", "Pratap Nagar"],
        "base_price": 4500   # ₹ per sq ft
    },
    "Jodhpur": {
        "localities": ["Ratanada", "Paota", "Shastri Nagar", "Sardarpura",
                       "Pal Road", "Bhagat Ki Kothi", "Chopasni Housing Board",
                       "Residency Road", "Mandore Road", "New Pali Road"],
        "base_price": 3200
    },
    "Udaipur": {
        "localities": ["Hiran Magri", "Bhuwana", "Fatehpura", "Madhuban",
                       "Pratap Nagar", "Savina", "Sector 4", "Sector 11",
                       "Sukhadia Circle", "Sisarma Road"],
        "base_price": 3600
    },
    "Kota": {
        "localities": ["Talwandi", "Vigyan Nagar", "Mahaveer Nagar",
                       "Dadwara Colony", "Rajiv Gandhi Nagar", "Nayapura",
                       "Gumanpura", "Jawahar Nagar", "Subhash Nagar", "Rangbari"],
        "base_price": 2800
    },
    "Ajmer": {
        "localities": ["Vaishali Nagar", "Ramganj", "Kishangarh Road",
                       "Madar Gate", "Civil Lines", "Pushkar Road",
                       "Gandhi Nagar", "Kekri Road", "Adarsh Nagar", "Pal Road"],
        "base_price": 2500
    },
    "Bikaner": {
        "localities": ["Sadul Ganj", "Shiv Bari Road", "Ambedkar Circle",
                       "Rani Bazar", "Ganga Shahar", "Jassusar Gate",
                       "Pawan Puri", "Karni Nagar", "Murli Manohar Nagar", "Lalgarh"],
        "base_price": 2200
    },
    "Sikar": {
        "localities": ["Fatehpur Road", "Jaipur Road", "Laxmangarh Road",
                       "Station Road", "Ajitgarh", "Piprali Road",
                       "Jhunjhunu Road", "Losal Road", "Shyam Nagar", "Bhairu Ji Ka Rasta"],
        "base_price": 2000
    },
    "Alwar": {
        "localities": ["Bhiwadi", "Neemrana", "Rajgarh Road", "Tijara Road",
                       "Behror", "Palawala", "Thanagazi", "Kotkasim",
                       "Tapukara", "Kishangarh Bas"],
        "base_price": 2400
    },
}

LOCALITY_MULTIPLIER = {
    "premium": 1.4, "good": 1.2, "average": 1.0, "developing": 0.85
}

# ─────────────────────────────────────────────
#  SYNTHETIC DATASET GENERATOR
# ─────────────────────────────────────────────
@st.cache_data
def generate_dataset():
    np.random.seed(42)
    records = []

    cities = list(RAJASTHAN_DATA.keys())
    bhk_options = [1, 2, 3, 4, 5]

    for _ in range(5000):
        city = np.random.choice(cities)
        info = RAJASTHAN_DATA[city]
        locality_idx = np.random.randint(0, len(info["localities"]))
        locality = info["localities"][locality_idx]
        bhk = np.random.choice(bhk_options, p=[0.10, 0.35, 0.35, 0.15, 0.05])

        # Area ranges by BHK (sq ft)
        area_ranges = {1: (400, 700), 2: (700, 1100), 3: (1100, 1700), 4: (1700, 2500), 5: (2500, 4000)}
        area = np.random.randint(*area_ranges[bhk])

        house_age = np.random.randint(0, 30)
        base = info["base_price"]

        # Locality tier (first third = premium, etc.)
        if locality_idx < 3:
            tier = 1.35
        elif locality_idx < 6:
            tier = 1.15
        elif locality_idx < 8:
            tier = 1.0
        else:
            tier = 0.85

        # Price formula
        age_factor = max(0.6, 1 - house_age * 0.012)
        bhk_bonus = 1 + (bhk - 1) * 0.05
        price = base * tier * age_factor * bhk_bonus * area
        noise = np.random.normal(1.0, 0.06)
        price = int(price * noise)

        records.append({
            "City": city,
            "Locality": locality,
            "BHK": bhk,
            "Area_SqFt": area,
            "House_Age": house_age,
            "Price": price
        })

    return pd.DataFrame(records)

# ─────────────────────────────────────────────
#  TRAIN MODEL
# ─────────────────────────────────────────────
@st.cache_resource
def train_model():
    df = generate_dataset()

    le_city = LabelEncoder()
    le_locality = LabelEncoder()

    df["City_enc"] = le_city.fit_transform(df["City"])
    df["Locality_enc"] = le_locality.fit_transform(df["Locality"])

    features = ["City_enc", "Locality_enc", "BHK", "Area_SqFt", "House_Age"]
    X = df[features]
    y = df["Price"]

    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X, y)

    return model, le_city, le_locality, df

# ─────────────────────────────────────────────
#  LOAD
# ─────────────────────────────────────────────
model, le_city, le_locality, df = train_model()

# ─────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────
st.markdown("""
    <h1 style='text-align:center; color:#E85D04;'>🏠 Rajasthan House Price Predictor</h1>
    <p style='text-align:center; color:gray;'>Get an instant estimated price for any property in Rajasthan</p>
    <hr>
""", unsafe_allow_html=True)

st.subheader("📋 Enter Property Details")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox(
        "🌆 Select City",
        options=list(RAJASTHAN_DATA.keys()),
        help="Choose the city where the property is located"
    )

    locality = st.selectbox(
        "📍 Select Locality / Area",
        options=RAJASTHAN_DATA[city]["localities"],
        help="Choose the specific locality in the city"
    )

with col2:
    bhk = st.selectbox(
        "🛏️ Select BHK Type",
        options=[1, 2, 3, 4, 5],
        index=1,
        format_func=lambda x: f"{x} BHK",
        help="Number of bedrooms (BHK)"
    )

    area = st.number_input(
        "📐 Area (Square Feet)",
        min_value=200,
        max_value=10000,
        value=1000,
        step=50,
        help="Total built-up area of the property in square feet"
    )

house_age = st.slider(
    "🏗️ House Age (Years)",
    min_value=0,
    max_value=50,
    value=5,
    help="How old is the property? (0 = brand new)"
)

st.markdown("")

# ─────────────────────────────────────────────
#  PREDICT
# ─────────────────────────────────────────────
if st.button("🔍 Predict House Price", use_container_width=True, type="primary"):

    # Encode inputs
    try:
        city_enc = le_city.transform([city])[0]
    except Exception:
        city_enc = 0

    try:
        locality_enc = le_locality.transform([locality])[0]
    except Exception:
        locality_enc = 0

    input_data = np.array([[city_enc, locality_enc, bhk, area, house_age]])
    predicted_price = model.predict(input_data)[0]

    # Price range (±8%)
    low = predicted_price * 0.92
    high = predicted_price * 1.08

    def fmt(p):
        if p >= 1e7:
            return f"₹{p/1e7:.2f} Cr"
        elif p >= 1e5:
            return f"₹{p/1e5:.2f} Lakh"
        else:
            return f"₹{p:,.0f}"

    price_per_sqft = int(predicted_price / area)

    st.markdown("---")
    st.markdown("### 🏷️ Prediction Result")

    col_r1, col_r2, col_r3 = st.columns(3)

    with col_r1:
        st.metric("💰 Estimated Price", fmt(predicted_price))
    with col_r2:
        st.metric("📉 Low Estimate", fmt(low))
    with col_r3:
        st.metric("📈 High Estimate", fmt(high))

    st.markdown("")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.info(f"📐 **Price per Sq Ft:** ₹{price_per_sqft:,}")
    with col_d2:
        condition = "New / Excellent" if house_age <= 3 else ("Good" if house_age <= 10 else ("Fair" if house_age <= 20 else "Old"))
        st.info(f"🏗️ **Property Condition:** {condition}")

    # Summary card
    st.success(f"""
    **Property Summary**
    - 📍 {locality}, {city}, Rajasthan
    - 🛏️ {bhk} BHK | 📐 {area:,} sq ft | 🏗️ {house_age} year(s) old
    - 💰 Estimated Price Range: **{fmt(low)} – {fmt(high)}**
    """)

    # Market comparison
    st.markdown("---")
    st.markdown("### 📊 Market Comparison in " + city)

    city_df = df[df["City"] == city]
    same_bhk = city_df[city_df["BHK"] == bhk]

    if len(same_bhk) > 0:
        avg_price = same_bhk["Price"].mean()
        min_price = same_bhk["Price"].min()
        max_price = same_bhk["Price"].max()

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Average Price", fmt(avg_price))
        with c2:
            st.metric("Lowest Listing", fmt(min_price))
        with c3:
            st.metric("Highest Listing", fmt(max_price))

        diff_pct = ((predicted_price - avg_price) / avg_price) * 100
        if abs(diff_pct) < 5:
            st.info(f"✅ Your property is priced **close to the market average** for {bhk} BHK in {city}.")
        elif diff_pct > 0:
            st.warning(f"⬆️ Your property is **{diff_pct:.1f}% above** the average {bhk} BHK price in {city}.")
        else:
            st.success(f"⬇️ Your property is **{abs(diff_pct):.1f}% below** the average {bhk} BHK price in {city} — a good deal!")

# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray; font-size:12px;'>"
    "⚠️ Prices are estimates based on Rajasthan market trends. "
    "Actual prices may vary based on floor, amenities, builder, and legal status."
    "</p>",
    unsafe_allow_html=True
)
