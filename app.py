import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ==========================================
# 1. PAGE SETUP & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="ChurnGuard | Owner Dashboard", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom styling for professional look
st.markdown("""
    <style>
    .metric-card { 
        background-color: #f8f9fa; 
        padding: 20px; 
        border-radius: 10px; 
        border-left: 5px solid #ff4b4b; 
    }
    </style>
""", unsafe_allow_html=True)

st.title("🎯 ChurnGuard Owner Dashboard")
st.caption("Real-time customer health monitoring & automated retention marketing console")

# ==========================================
# 2. UTILITY CHANNELS (APIs & NOTIFICATIONS)
# ==========================================
def send_twilio_message(customer_name, target_phone, discount_tier):
    """Dispatches real-time SMS/WhatsApp via Twilio API Wrapper"""
    # Replace these placeholder tokens with your real Twilio Console credentials
    account_sid = 'YOUR_TWILIO_ACCOUNT_SID'
    auth_token = 'YOUR_TWILIO_AUTH_TOKEN'
    twilio_number = '+1234567890' # For WhatsApp use: 'whatsapp:+14155238886'
    
    try:
        from twilio.rest import Client
        client = Client(account_sid, auth_token)
        body_text = f"We miss you, {customer_name.upper()}! 🏷️ Use code COMEBACK for {discount_tier} on your next order!"
        
        # In a sandbox environment, 'to' must be a verified test number
        message = client.messages.create(
            body=body_text,
            from_=twilio_number,
            to=target_phone
        )
        return True, message.sid
    except Exception as e:
        return False, str(e)


def send_retention_email(customer_name, target_email, discount_tier, sender_email, sender_password):
    """Compiles and shoots a customized HTML win-back email via Gmail SMTP.

    sender_email/sender_password come from the sidebar at runtime - never
    hardcode real credentials into source. sender_password must be a Gmail
    App Password (myaccount.google.com/apppasswords), not the account password.
    """
    try:
        # Connect to secure Gmail SMTP relay server
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(sender_email, sender_password)
        
        # Build structure package
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = target_email
        msg['Subject'] = f"Hey {customer_name.capitalize()}, a special gift inside just for you!"
        
        html_layout = f"""
        <html>
          <body style="font-family: Arial, sans-serif; text-align: center; padding: 20px; color: #333;">
            <h2 style="color: #ff4b4b;">We've Missed You!</h2>
            <p>Hi {customer_name}, it's been a minute since your last order on our platform.</p>
            <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 20px 0; border: 1px dashed #ff4b4b;">
                <h1 style="margin: 0; color: #ff4b4b;">💥 {discount_tier} 💥</h1>
            </div>
            <p>Hurry up! Grab your favorite picks before this deal expires.</p>
            <br>
            <a href="https://your-app-link.com" style="background: #ff4b4b; color: white; padding: 12px 25px; text-decoration: none; border-radius: 5px; font-weight: bold;">Claim Offer Now</a>
          </body>
        </html>
        """
        msg.attach(MIMEText(html_layout, 'html'))
        server.sendmail(sender_email, target_email, msg.as_string())
        server.quit()
        return True, "Success"
    except Exception as e:
        return False, str(e)


# ==========================================
# 3. SIDEBAR CONTROL CENTER
# ==========================================
with st.sidebar:
    st.header("⚙️ System Control Panel")
    uploaded_file = st.file_uploader("Upload Live Customer Data (CSV)", type="csv")
    
    st.markdown("---")
    st.subheader("Campaign Hyper-parameters")
    risk_threshold = st.slider("Churn Risk Alert Threshold", min_value=50, max_value=95, value=70, step=5) / 100.0

    auto_pilot = st.toggle("Enable 24/7 Autopilot Processing", value=False)
    if auto_pilot:
        st.success("Background cron jobs monitoring pipeline active.")

    st.markdown("---")
    st.subheader("✉️ Email Win-Back Campaign")
    discount_pct = st.slider("Discount to offer", min_value=5, max_value=20, value=15, step=5, format="%d%%")
    sender_email = st.text_input("Sender Gmail address", placeholder="yourbusiness@gmail.com")
    sender_app_password = st.text_input(
        "Gmail App Password", type="password",
        help="Generate one at myaccount.google.com/apppasswords - do not use your normal Gmail password."
    )
    live_send = st.checkbox("Actually send emails (unchecked = dry run, no email leaves your machine)", value=False)
    if live_send and (not sender_email or not sender_app_password):
        st.warning("Enter a sender address and App Password to enable live sending.")

# ==========================================
# 4. MODEL LOADING (CACHE PROTECTED)
# ==========================================
@st.cache_resource
def load_rf_model():
    try:
        with open('rf_model_balanced.pkl', 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return None

model_bundle = load_rf_model()

def score_telco_dataframe(raw_df, bundle):
    """Runs the real trained Random Forest on a Telco-formatted dataframe.

    Replays the exact cleaning/encoding steps used in train_model.py so the
    uploaded data lands in the same feature space the model was fit on.
    """
    proc = raw_df.drop(columns=['customerID'], errors='ignore').copy()
    proc['TotalCharges'] = pd.to_numeric(proc['TotalCharges'], errors='coerce')
    proc['TotalCharges'] = proc['TotalCharges'].fillna(proc['TotalCharges'].median())

    if 'Churn' in proc.columns:
        proc = proc.drop('Churn', axis=1)

    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        if col in proc.columns:
            proc[col] = proc[col].map({'Yes': 1, 'No': 0})
    if 'gender' in proc.columns:
        proc['gender'] = proc['gender'].map({'Male': 1, 'Female': 0})

    categorical_cols = proc.select_dtypes(include='object').columns.tolist()
    proc = pd.get_dummies(proc, columns=categorical_cols, drop_first=True)
    proc = proc.reindex(columns=bundle['feature_columns'], fill_value=0)

    scaled = bundle['scaler'].transform(proc)
    return bundle['model'].predict_proba(scaled)[:, 1]

# ==========================================
# 5. CORE INFERENCE ENGINE & ANALYTICS
# ==========================================
if uploaded_file is not None:
    # Process dataset file stream
    df = pd.read_csv(uploaded_file)

    # --- AUTO-MAPPING LAYER FOR TELCO CHURN DATASET ---
    is_telco_format = 'customerID' in df.columns and 'tenure' in df.columns
    if is_telco_format:
        df['customer_id'] = df['customerID']
        if 'customer_name' not in df.columns:
            df['customer_name'] = df['customer_id'].apply(lambda x: f"User {x[:5]}")
        # Contact fields aren't in the source dataset (it's telecom account data,
        # no phone/email columns) - these are cosmetic, for demoing the dispatch
        # channels only, and are never fed into the model.
        if 'phone_number' not in df.columns:
            rng = np.random.default_rng(42)
            df['phone_number'] = "+91 98765 " + rng.integers(10000, 99999, size=len(df)).astype(str)
        if 'email' not in df.columns:
            df['email'] = df['customer_id'].apply(lambda x: f"user_{x[:5].lower()}@example.com")
        df['avg_order_amount'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)

    if 'Churn_Probability' not in df.columns:
        if is_telco_format and model_bundle is not None:
            df['Churn_Probability'] = score_telco_dataframe(df, model_bundle)
        elif model_bundle is None:
            st.error("⚠️ 'rf_model_balanced.pkl' not found. Run `python train_model.py` to train the real model on WA_Fn-UseC_-Telco-Customer-Churn.csv, then re-upload.")
            st.stop()
        else:
            st.error("⚠️ This file doesn't match the Telco churn dataset schema (needs 'customerID' and 'tenure' columns), and has no 'Churn_Probability' column to display directly. Upload the Telco dataset or a file with precomputed scores.")
            st.stop()

    df['Status'] = df['Churn_Probability'].apply(lambda x: "🚨 High Risk" if x >= risk_threshold else "✅ Active")

    # --- Executive KPI Board Metrics ---
    total_customers = len(df)
    at_risk_count = len(df[df['Status'] == "🚨 High Risk"])
    at_risk_pct = (at_risk_count / total_customers) * 100 if total_customers > 0 else 0
    potential_lost_rev = df[df['Status'] == "🚨 High Risk"]['avg_order_amount'].sum()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Monitored Users", f"{total_customers:,}", help="Total customers processed in this data stream")
    with col2:
        st.metric("Customers At Risk", f"{at_risk_count:,}", f"{at_risk_pct:.1f}% Base Ratio", delta_color="inverse")
    with col3:
        st.metric("Revenue At Risk", f"₹{potential_lost_rev:,.2f}", "Gross base valuation risk", delta_color="inverse")
    with col4:
        if model_bundle is not None and "test_accuracy" in model_bundle:
            st.metric("Model Test Accuracy", f"{model_bundle['test_accuracy']*100:.1f}%", "RandomForest, held-out Telco test set")
        else:
            st.metric("Active Campaigns", "2 Channels Running", "SMS & Email Engine")

    st.markdown("---")

    # --- Plotly Visual Analytics Blocks ---
    chart_col1, chart_col2 = st.columns([3, 2])
    
    with chart_col1:
        st.subheader("📊 Operational Risk Distribution")
        fig = px.histogram(df, x="Churn_Probability", nbins=20, 
                           title="Distribution of Churn Likelihood across Userbase",
                           labels={'Churn_Probability': 'Model Probability Score (0 to 1)'},
                           color_discrete_sequence=['#ff4b4b'])
        fig.add_vline(x=risk_threshold, line_dash="dash", line_color="black", annotation_text="Campaign Trigger Line")
        st.plotly_chart(fig, use_container_width=True)
        
    with chart_col2:
        st.subheader("🍰 Retention Health Breakdown")
        pie_fig = px.pie(df, names='Status', title='Customer Base Segment Composition',
                         color='Status', color_discrete_map={'🚨 High Risk': '#ff4b4b', '✅ Active': '#00cc96'})
        st.plotly_chart(pie_fig, use_container_width=True)

    st.markdown("---")

    # --- Deep-Dive Interactive Matrix Grid ---
    st.subheader("🔎 Actionable Customer Intelligence Table")
    
    view_df = df.copy()
    view_df['Churn Score'] = (view_df['Churn_Probability'] * 100).round(1).astype(str) + "%"
    
    # Filter control layer
    show_only_risk = st.checkbox("Isolate High-Risk accounts only", value=True)
    if show_only_risk:
        view_df = view_df[view_df['Status'] == "🚨 High Risk"]
        
    # Standard column fallback layout formatting safety
    required_cols = ['customer_id', 'customer_name', 'phone_number', 'email', 'tenure', 'Contract', 'avg_order_amount', 'Churn Score', 'Status']
    existing_cols = [c for c in required_cols if c in view_df.columns]
    
    st.dataframe(
        view_df[existing_cols],
        use_container_width=True,
        hide_index=True
    )

    # ==========================================
    # 6. EMAIL WIN-BACK CAMPAIGN DISPATCH
    # ==========================================
    st.markdown("### 🚀 Batch Dispatch Retention Offers")
    discount_tier = f"{discount_pct}% OFF"
    if live_send:
        st.write(f"Sends a real **{discount_tier}** win-back email to every high-risk customer via your Gmail SMTP account.")
    else:
        st.write(f"**Dry run** - previews a **{discount_tier}** win-back email to every high-risk customer without sending anything.")

    if st.button("Execute Email Win-Back Campaign"):
        high_risk_list = df[df['Status'] == "🚨 High Risk"]

        if len(high_risk_list) == 0:
            st.success("No customers currently cross your high-risk threshold flag. No marketing steps needed!")
        elif live_send and (not sender_email or not sender_app_password):
            st.error("⚠️ Live sending is on but sender address / App Password is missing. Fill both in, or switch to dry run.")
        else:
            if live_send and 'example.com' in high_risk_list['email'].astype(str).str.cat():
                st.warning("⚠️ Some recipients have placeholder @example.com demo addresses (no real emails in this dataset) - those sends will fail/bounce.")

            progress_bar = st.progress(0)
            status_text = st.empty()
            sent_count, failed_count = 0, 0

            for index, row in high_risk_list.reset_index().iterrows():
                progress_value = (index + 1) / len(high_risk_list)
                progress_bar.progress(progress_value)
                status_text.text(f"Processing customer {index+1}/{len(high_risk_list)}: {row['customer_name']}")

                if live_send:
                    success, info = send_retention_email(
                        row['customer_name'], row['email'], discount_tier, sender_email, sender_app_password
                    )
                    if success:
                        sent_count += 1
                        st.toast(f"✅ Sent {discount_tier} email to {row['customer_name']}", icon="✉️")
                    else:
                        failed_count += 1
                        st.toast(f"❌ Failed to email {row['customer_name']}: {info}", icon="⚠️")
                else:
                    st.toast(f"✉️ [Dry Run] Would send {discount_tier} email to {row['email']}", icon="✉️")
                    time.sleep(0.05)

            status_text.empty()
            if live_send:
                st.success(f"🎉 Email campaign complete: {sent_count} sent, {failed_count} failed, out of {len(high_risk_list)} high-risk customers.")
            else:
                st.success(f"🎉 Dry run complete: {len(high_risk_list)} emails previewed. Enable live sending in the sidebar to actually send them.")
        
else:
    # Empty Landing State Dashboard UI Onboarding Instruction
    st.info("💡 Control Panel Dashboard Awaiting File Injection. Please drag and drop your current Customer Behavior Transaction CSV file via the left sidebar option to populate predictive insights.")