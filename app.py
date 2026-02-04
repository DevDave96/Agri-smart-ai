import streamlit as st
import pandas as pd 
import numpy as np
import tensorflow as tf 
import joblib
import plotly.graph_objects as go 

#---Page Config---
st.set_page_config(
    page_title="AgriSmart AI",
    page_icon="🌿",
    layout="wide"
)


# --- CUSTOM CSS FOR STYLING ---
st.markdown("""
    <style>
    /* Main Background - Soft Sage */
    .stApp {
        background-color: #f8faf8;
    }
    
    /* Global Font Settings */
    html, body, [class*="css"] {
        font-family: 'Inter', 'Segoe UI', sans-serif;
        color: #2c3e50;
    }

    /* Modern Headers */
    h1 {
        color: #1b5e20 !important;
        font-weight: 800 !important;
        letter-spacing: -1px;
    }
    h2, h3 {
        color: #2e7d32 !important;
        font-weight: 600 !important;
    }

    /* Style the Sidebar/Input Containers */
    div[data-testid="stVerticalBlock"] > div:has(div.stSlider) {
        background-color: #ffffff;
        padding: 25px;
        border-radius: 20px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 1px solid #e0e6e0;
    }

    /* Beautiful Slider & Input Labels */
    .stSlider label, .stNumberInput label {
        font-weight: 600 !important;
        color: #1b5e20 !important;
        font-size: 1.1rem !important;
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 15px 20px !important;
        font-size: 1.2rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.3);
        margin-top: 20px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 15px rgba(46, 125, 50, 0.4);
    }

    /* Result Card Styling */
    .result-card {
        background: white;
        padding: 40px;
        border-radius: 25px;
        text-align: center;
        border-bottom: 8px solid #2e7d32;
        box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    }
    </style>
    """, unsafe_allow_html=True)


#---ICON DICTIONARY (To make it pretty)---
crop_icons = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🌱", "kidneybeans": "🫘",
    "pigeonpeas": "🌿", "mothbeans": "🌿", "mungbean": "🌿", "blackgram": "🌿",
    "lentil": "🌿", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🥭", "coconut": "🥥", "cotton": "☁️",
    "jute": "🧵", "coffee": "☕"
}

# --- ADD THIS DATA NOW ---
crop_advice = {
    "rice": "Requires constant water. Maintain 2-4 inches of water depth in the field. Best in heavy clay soils.",
    "maize": "Needs high nitrogen during the vegetative stage. Ensure good drainage to avoid root rot.",
    "chickpea": "Drought-tolerant. Avoid excess water during flowering. Needs well-aerated soil.",
    "kidneybeans": "Needs moderate moisture. Very sensitive to frost and high winds.",
    "pigeonpeas": "Slow-growing initially; great for soil health as it fixes nitrogen.",
    "pomegranate": "Requires hot, dry summers for fruit ripening. Prune regularly for better yield.",
    "banana": "Heavy feeder of Potassium. Needs protection from strong winds and plenty of water.",
    "mango": "Deep-rooted tree. Avoid heavy irrigation during the flowering period.",
    "grapes": "Requires a trellis system. Pruning is key to controlling fruit quality.",
    "watermelon": "Needs sandy soil and lots of space. Water heavily until fruit reaches full size.",
    "apple": "Requires 'chilling hours' in winter to produce fruit.",
    "orange": "Needs well-drained soil. Sensitive to cold; maintain consistent soil moisture.",
    "papaya": "Very fast-growing. Avoid waterlogging at all costs as the stems rot easily.",
    "coconut": "Thrives in coastal saline soils. Needs high humidity and year-round warmth.",
    "cotton": "Requires a long frost-free period and plenty of sunshine.",
    "jute": "Needs a hot and wet climate. Best grown in alluvial soil.",
    "coffee": "Grows best under a canopy of shade trees. Requires acidic soil."
}

#---LOAD ASSETS---
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('crop_ann_model.keras')
    scaler = joblib.load('crop_scaler.pkl')
    le = joblib.load('crop_label_encoder.pkl')
    return model, scaler, le

model, scaler, le = load_assets()

# --- THE REST OF YOUR UI CODE GOES BELOW ---
st.title("🌱 AgriSmart: Neural Crop Intelligence")
st.markdown("##### *Predicting the future of your harvest with Deep Learning*")
st.write("---")

# --- MAIN UI LAYOUT ---
# We use columns to make it look like a dashboard
col_input, col_display = st.columns([1,1], gap="large")

with col_input:
    st.subheader("📍 Field Environment Data")
    st.write("Adjust the parameters based on your soil test and local weather.")

   # ---Using a container to group the inputs nicely

    with st.container():
        n = st.slider("Nitrogen (N)", 0, 150, 50, help="Amount of Nitrogen in soil")
        p = st.slider("Phosphorus (P)", 0, 150, 50, help="Amount of Phosphorus in soil")
        k = st.slider("Potassium (K)", 0, 250, 50, help="Amount of Potassium in soil")

        # ---Two columns inside the input area for more compact look
        c1, c2 = st.columns(2)
        with c1:
             temp = st.number_input("Temperature(°C)", value=25.0, step=0.1)
             ph = st.number_input("Soil pH level", 0.0, 14.0, 6.5, step=0.1)
        with c2:
             hum = st.number_input("Humidity (%)", 0.0, 100.0, 60.0, step=0.1)
             rain = st.number_input("Rainfall (mm)", value=100.0, step=1.0)

        st.write("---")
        predict_btn = st.button("Generate AI Recommendation")

with col_display:
     st.subheader("📊 Recommendation Result")

     if predict_btn:
          # ---Prepare data for the model
          input_data = np.array([[n,p,k,temp,hum,ph,rain]])
          input_scaled = scaler.transform(input_data)


          #---Prediction
          all_probs = model.predict(input_scaled)[0]
          top_3_indices = all_probs.argsort()[-3:][::-1]
          top_3_crops = le.inverse_transform(top_3_indices)
          top_3_confidences = all_probs[top_3_indices] * 100
          crop = top_3_crops[0]
          confidence = top_3_confidences[0]


          # ---Get the icon for the crop
          icon = crop_icons.get(crop.lower(), "🌱")


          # ---Display the Result
          st.markdown(f"""
            <div class="result-card">
                <p style="text-transform: uppercase; color: #666; letter-spacing: 2px; font-weight: bold; margin-bottom: 0;">Optimal Crop Found</p>
                <h1 style="font-size: 100px; margin: 10px 0;">{icon}</h1>
                <h1 style="color: #1b5e20; font-size: 50px; margin: 0;">{crop.upper()}</h1>
                <p style="font-size: 20px; color: #555; margin-top: 15px;">
                    Our Neural Network is <b>{confidence:.1f}%</b> confident in this recommendation.
                </p>
            </div>
        """, unsafe_allow_html=True)
          

          # ---The Gauge Chart (Visual Confidence)
          fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = confidence,
            title = {'text': "AI Confidence %", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#2e7d32"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#eeeeee",
                'steps': [
                    {'range': [0, 50], 'color': '#ffcccb'},
                    {'range': [50, 80], 'color': '#fff9c4'},
                    {'range': [80, 100], 'color': '#c8e6c9'}
                ],
            }
        ))
          fig.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=0))
          st.plotly_chart(fig, use_container_width=True)

          st.write("---")
          st.subheader("🥈 Alternative Crop Suitability")

          fig_alt = go.Figure(go.Bar(
              x=top_3_confidences,
              # This makes the names look nice (Capitalized)
              y=[c.title() for c in top_3_crops], 
              orientation='h',
              marker_color=['#1b5e20', '#66bb6a', '#a5d6a7'],
              text=[f"{val:.1f}%" for val in top_3_confidences],
              textposition='auto',
          ))

          fig_alt.update_layout(
              height=250,
              margin=dict(l=10, r=10, t=10, b=10),
              xaxis=dict(title="Confidence %", range=[0, 110]),
              yaxis=dict(autorange="reversed") # Put the best one at the top
          )

          st.plotly_chart(fig_alt, use_container_width=True)
          
          # A little bit of extra flavor text
          st.info(f"🧐 **Scientist's Note:** Your soil is also **{top_3_confidences[1]:.1f}%** suitable for **{top_3_crops[1].title()}**. Small adjustments to your nutrients could make this a viable second option!")

          st.markdown("### 💡 Expert Cultivation Advice")

          advice = crop_advice.get(crop.lower(), "Ensure regular soil testing and proper irrigation for best results.")

          st.success(f"**Pro Tip for {crop.title()}:** {advice}")

          if ph < 5.5:
              st.warning("⚠️ **Soil Note:** Your pH is quite acidic. Consider adding lime to improve nutrient uptake.")
          elif ph > 7.5:
              st.warning("⚠️ **Soil Note:** Your soil is alkaline. Consider adding organic mulch to balance it.")

          st.write("---")
          report_text = f"""
          AGRISMART AI - CROP RECOMMENDATION REPORT
          Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
          -----------------------------------------
          FIELD DATA PROVIDED:
          - Nitrogen (N): {n}
          - Phosphorus (P): {p}
          - Potassium (K): {k}
          - Temperature: {temp}C
          - Humidity: {hum}%
          - Soil pH: {ph}
          - Rainfall: {rain}mm

          AI RECOMMENDATION:
          - Recommended Crop: {crop.upper()}
          - AI Confidence: {confidence:.1f}%
          - Second Best: {top_3_crops[1].title()} ({top_3_confidences[1]:.1f}%)

          EXPERT ADVICE:
          {advice}
          -----------------------------------------
          Generated by AgriSmart Intelligence
          """

          st.download_button(
              label="📥 Download Harvest Report",
              data=report_text,
              file_name=f"AgriSmart_{crop.lower()}_report.txt",
              mime="text/plain",
              use_container_width=True
          )

          if confidence > 80:
               st.balloons()
        
     else:
        # What shows before the user clicks predict
        st.info("👈 Enter your field data and click the green button to see the AI's suggestion.")
        st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?auto=format&fit=crop&w=600&q=80", use_column_width=True)


 # --- FOOTER ---
st.markdown("---")

st.markdown("<center>AgriSmart Intelligence © 2026 | Built with Artificial Neural Networks</center>", unsafe_allow_html=True)
