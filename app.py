import streamlit as st
import pandas as pd
import time
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv
from twilio.rest import Client
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pharmacy Refill Assistant",
    page_icon="💊",
    layout="wide"
)

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #fff5f5 0%, #ffe3e3 100%);
    }

    h1, h2, h3, h4 {
        color: #7f1d1d !important;
        font-family: Arial, sans-serif;
    }

    p, span, div, label {
        color: #5f2120;
        font-family: Arial, sans-serif;
    }

    .hero-box {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        padding: 1.3rem 1.5rem;
        border-radius: 22px;
        color: white !important;
        box-shadow: 0 10px 24px rgba(220, 38, 38, 0.18);
        margin-bottom: 1rem;
    }

    .hero-box h1, .hero-box p, .hero-box strong {
        color: white !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 700;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #dc2626, #b91c1c);
        color: white;
    }

    div[data-baseweb="select"] > div {
        background-color: #ffffff !important;
        border: 1px solid #fecaca !important;
        border-radius: 12px;
    }

    [data-testid="stDataFrame"] {
        background-color: #fff5f5 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- ENV ----------------
load_dotenv()

# ---------------- DATA ----------------
CSV_FILE = "prescription_refill_demo_dataset (15).csv"

def load_data():
    return pd.read_csv(CSV_FILE)

def save_data(df):
    df.to_csv(CSV_FILE, index=False)

if "df" not in st.session_state:
    st.session_state.df = load_data()

if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = None

if "last_sms_result" not in st.session_state:
    st.session_state.last_sms_result = None

# ---------------- HELPERS ----------------
def normalize_label(value):
    return str(value).strip().lower().replace(" ", "_")

def get_patient_phone(row):
    if "phone_number" in row.index and pd.notna(row["phone_number"]):
        return str(row["phone_number"])
    return os.getenv("DEMO_PATIENT_PHONE", "+15555555555")

def predict_exhaustion_date(row):
    adjustment = int(row["prior_early_refill_count"]) * -1
    days_remaining = (
        int(row["expected_days_supply"])
        - int(row["days_since_last_refill"])
        + adjustment
    )
    days_remaining = max(0, days_remaining)
    return (datetime.now() + timedelta(days=days_remaining)).strftime("%B %d, %Y")

def triage_decision(row):
    if row["medication_group"] == "ADHD / Controlled":
        return "Needs Review"
    elif int(row["days_since_last_refill"]) > int(row["expected_days_supply"]):
        return "Needs Assistance"
    elif int(row["prior_early_refill_count"]) > 2:
        return "Needs Review"
    return "Approved"

def get_confidence_score(row):
    score = 95 - (int(row["prior_early_refill_count"]) * 7)

    if row["medication_group"] == "ADHD / Controlled":
        score -= 10

    if int(row["days_since_last_refill"]) > int(row["expected_days_supply"]):
        score -= 8

    return max(60, score)

def generate_pending_sms():
    return (
        "Pharmacy Update: We received your refill request. "
        "It is currently under pharmacist review. "
        "We’ll text you again once a final decision has been made."
    )

def generate_patient_sms(row, decision):
    if decision in ["Approved", "Needs Review", "Needs Assistance"]:
        return generate_pending_sms()
    return generate_pending_sms()

def generate_ready_sms(patient_id, medication_group):
    return (
        f"Pharmacy Update: Your {medication_group} prescription is ready for pickup. "
        f"Please bring ID if needed. Reply STOP to opt out."
    )

def generate_human_approved_sms(patient_id, medication_group):
    return (
        f"Pharmacy Update: A pharmacist has approved your refill request for {medication_group}. "
        f"We are now processing it for patient {patient_id}. "
        "We’ll send another message when it is ready for pickup. Reply STOP to opt out."
    )

def generate_human_rejected_sms(patient_id, medication_group):
    return (
        f"Pharmacy Update: A pharmacist reviewed your refill request for {medication_group} "
        f"and was unable to approve it for patient {patient_id} at this time. "
        "Please contact the pharmacy for next steps. Reply STOP to opt out."
    )

def generate_ai_explanation(row):
    reasons = []
    risk_score = 0

    if row["medication_group"] == "ADHD / Controlled":
        reasons.append("Controlled medication detected → elevated regulatory review risk")
        risk_score += 30

    if int(row["prior_early_refill_count"]) > 2:
        reasons.append("Repeated early refill behavior detected → adherence anomaly pattern")
        risk_score += 25

    if int(row["days_since_last_refill"]) > int(row["expected_days_supply"]):
        reasons.append("Medication gap identified → potential non-adherence risk")
        risk_score += 20

    if not reasons:
        reasons.append("No major risk signals detected → refill behavior appears stable")

    return reasons, risk_score

def send_sms_message(patient_phone, sms_body):
    account_sid = os.getenv("TWILIO_ACCOUNT_SID")
    auth_token = os.getenv("TWILIO_AUTH_TOKEN")
    twilio_number = os.getenv("TWILIO_PHONE_NUMBER")

    if not account_sid or not auth_token or not twilio_number:
        return {
            "success": False,
            "message": "Missing Twilio environment variables.",
            "sid": None
        }

    try:
        client = Client(account_sid, auth_token)
        message = client.messages.create(
            body=sms_body,
            from_=twilio_number,
            to=patient_phone
        )

        return {
            "success": True,
            "message": sms_body,
            "sid": message.sid
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e),
            "sid": None
        }

def send_ready_sms(patient_phone, patient_id, medication_group):
    sms_body = generate_ready_sms(patient_id, medication_group)
    return send_sms_message(patient_phone, sms_body)

def get_human_status(patient_id):
    row = st.session_state.df[st.session_state.df["patient_id"] == patient_id].iloc[0]
    human_label = normalize_label(row["human_label"])

    if human_label == "approved":
        return "Approved"
    elif human_label == "rejected":
        return "Rejected"
    else:
        return "Pending Review"

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero-box">
    <h1 style="margin-bottom: 0.35rem;">💊 Pharmacy Refill Assistant</h1>
    <p style="margin-bottom: 0.65rem;">
        AI-powered prescription triage, patient messaging, and pharmacist review in one streamlined workflow.
    </p>
    <p style="margin:0;">
        <strong>Features:</strong> Refill analysis • SMS updates • Explainable AI • Triage dashboard
    </p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📱 Patient App",
    "👨‍⚕️ Pharmacist View",
    "📊 Analytics"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Patient Refill Experience")
    st.caption("Simple single-column patient view.")

    with st.container(border=True):
        p_id = st.selectbox("Select Patient Profile", st.session_state.df["patient_id"])
        patient_row = st.session_state.df[st.session_state.df["patient_id"] == p_id].iloc[0]
        patient_phone = get_patient_phone(patient_row)

        if st.button("Check Refill Status", use_container_width=True):
            with st.status("Reviewing refill request...", expanded=True) as status:
                st.write("Verifying prescription history...")
                time.sleep(0.6)
                st.write("Reviewing refill timing and usage behavior...")
                time.sleep(0.6)
                st.write("Running decision support checks...")
                time.sleep(0.6)
                st.write("Preparing patient update...")
                time.sleep(0.6)

                exhaustion_date = predict_exhaustion_date(patient_row)
                decision = triage_decision(patient_row)
                confidence = get_confidence_score(patient_row)
                initial_sms = generate_patient_sms(patient_row, decision)
                explanation, risk_score = generate_ai_explanation(patient_row)

                sms_result = send_sms_message(patient_phone, initial_sms)

                st.session_state.last_analysis = {
                    "patient_id": patient_row["patient_id"],
                    "patient_phone": patient_phone,
                    "medication_group": patient_row["medication_group"],
                    "decision": decision,
                    "human_status": "Pending Review",
                    "confidence": confidence,
                    "risk_score": risk_score,
                    "predicted_end": exhaustion_date,
                    "initial_sms": initial_sms,
                    "explanation": explanation,
                    "request_reason": patient_row["request_reason"],
                    "days_since_last_refill": patient_row["days_since_last_refill"],
                    "expected_days_supply": patient_row["expected_days_supply"],
                    "prior_early_refill_count": patient_row["prior_early_refill_count"],
                }
                st.session_state.last_sms_result = sms_result

                status.update(label="Refill status updated", state="complete")

    data = st.session_state.last_analysis

    if data is None:
        st.info("Run a refill status check to preview the patient results.")
    else:
        latest_human_status = get_human_status(data["patient_id"])

        st.markdown("### Refill Status")
        with st.container(border=True):
            st.write(f"**Medication:** {data['medication_group']}")
            st.write(f"**Patient ID:** {data['patient_id']}")
            st.write(f"**Predicted supply end date:** {data['predicted_end']}")

            if latest_human_status == "Pending Review":
                st.warning("Refill under pharmacist review")
                st.write("Your refill request was submitted successfully. A pharmacist will make the final decision.")
            elif latest_human_status == "Approved":
                st.success("Refill approved by pharmacist")
                st.write("A pharmacist approved your refill request and you have been notified.")
            elif latest_human_status == "Rejected":
                st.error("Refill not approved")
                st.write("A pharmacist reviewed your request and it was not approved at this time.")

        st.markdown("### Decision Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("AI Recommendation", data["decision"])
        c2.metric("Confidence", f'{data["confidence"]}%')
        c3.metric("Risk Score", data["risk_score"])

        st.markdown("### Patient SMS Preview")
        with st.container(border=True):
            if latest_human_status == "Approved":
                st.info(generate_human_approved_sms(data["patient_id"], data["medication_group"]))
            elif latest_human_status == "Rejected":
                st.info(generate_human_rejected_sms(data["patient_id"], data["medication_group"]))
            else:
                st.info(data["initial_sms"])

            st.caption(f"Recipient: {data['patient_phone']}")

        sms_result = st.session_state.last_sms_result
        if sms_result and sms_result["success"]:
            st.success("Patient notification sent.")
            st.caption(f"Message SID: {sms_result['sid']}")
        elif sms_result and not sms_result["success"]:
            st.warning("SMS could not be sent live. Preview shown above instead.")
            st.caption(f"SMS error: {sms_result['message']}")

        with st.expander("See clinical reasoning"):
            st.write("**Decision factors:**")
            for reason in data["explanation"]:
                st.write(f"- {reason}")

            st.write("**Data used:**")
            st.write(f"- Days since last refill: {data['days_since_last_refill']}")
            st.write(f"- Expected days supply: {data['expected_days_supply']}")
            st.write(f"- Prior early refill count: {data['prior_early_refill_count']}")
            st.write(f"- Request reason: {data['request_reason']}")

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Pharmacist Triage Queue")

    review_needed = st.session_state.df[
        ~st.session_state.df["human_label"].astype(str).str.lower().isin(["approved", "rejected"])
    ]

    if review_needed.empty:
        st.success("No cases currently pending review.")
    else:
        for i, row in review_needed.iterrows():
            with st.expander(f"Case {row['patient_id']} — {row['medication_group']}"):
                col1, col2 = st.columns([2, 1])

                ai_decision = triage_decision(row)
                patient_phone = get_patient_phone(row)
                normalized_human_label = normalize_label(row["human_label"])
                normalized_ai_label = normalize_label(ai_decision)

                with col1:
                    st.write(f"**AI Recommendation:** {ai_decision}")
                    st.write(f"**Request Reason:** {row['request_reason']}")
                    st.write(f"**Prior Early Refills:** {row['prior_early_refill_count']}")
                    st.write(f"**Days Since Last Refill:** {row['days_since_last_refill']}")
                    st.write(f"**Expected Days Supply:** {row['expected_days_supply']}")
                    st.write(f"**Patient Phone:** {patient_phone}")

                    if normalized_ai_label != normalized_human_label and normalized_human_label not in ["pending_review", "needs_review"]:
                        st.error("Override detected: human decision differs from AI recommendation.")

                with col2:
                    if st.button("Approve + Notify", key=f"approve_{row['patient_id']}", use_container_width=True):
                        st.session_state.df.at[i, "human_label"] = "approved"
                        save_data(st.session_state.df)

                        sms_body = generate_human_approved_sms(
                            patient_id=row["patient_id"],
                            medication_group=row["medication_group"]
                        )
                        sms_result = send_sms_message(patient_phone, sms_body)
                        st.session_state.last_sms_result = sms_result

                        if sms_result["success"]:
                            st.success("Refill approved and patient notified.")
                        else:
                            st.warning(f"Refill approved, but SMS failed: {sms_result['message']}")

                        st.rerun()

                    if st.button("Reject + Notify", key=f"reject_{row['patient_id']}", use_container_width=True):
                        st.session_state.df.at[i, "human_label"] = "rejected"
                        save_data(st.session_state.df)

                        sms_body = generate_human_rejected_sms(
                            patient_id=row["patient_id"],
                            medication_group=row["medication_group"]
                        )
                        sms_result = send_sms_message(patient_phone, sms_body)
                        st.session_state.last_sms_result = sms_result

                        if sms_result["success"]:
                            st.error("Refill rejected and patient notified.")
                        else:
                            st.warning(f"Refill rejected, but SMS failed: {sms_result['message']}")

                        st.rerun()

                    if st.button("Escalate Review", key=f"escalate_{row['patient_id']}", use_container_width=True):
                        st.session_state.df.at[i, "human_label"] = "escalate"
                        save_data(st.session_state.df)
                        st.info("Case escalated for additional review.")
                        st.rerun()

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Pharmacy Analytics Dashboard")

    review_needed = st.session_state.df[
        ~st.session_state.df["human_label"].astype(str).str.lower().isin(["approved", "rejected"])
    ]

    active_rx = len(st.session_state.df[st.session_state.df["active_rx"] == 1])
    pending_triage = len(review_needed)
    high_risk_escalations = len(
        st.session_state.df[
            st.session_state.df["human_label"].astype(str).str.lower() == "escalate"
        ]
    )

    overrides = sum(
        normalize_label(triage_decision(row)) != normalize_label(row["human_label"])
        for _, row in st.session_state.df.iterrows()
    )

    override_rate = (overrides / len(st.session_state.df)) * 100 if len(st.session_state.df) > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active Prescriptions", active_rx)
    c2.metric("Pending Triage", pending_triage)
    c3.metric("High-Risk Escalations", high_risk_escalations)
    c4.metric("AI Override Rate", f"{override_rate:.1f}%")

    st.caption("Override rate shows the percentage of cases where the human decision differs from the AI recommendation.")

    st.markdown("### Medication Group Distribution")
    counts = st.session_state.df["medication_group"].value_counts()

    fig, ax = plt.subplots()
    counts.plot(kind="bar", ax=ax)
    ax.set_facecolor("#fff5f5")
    fig.patch.set_facecolor("#fff5f5")

    for bar in ax.patches:
        bar.set_color("#ef4444")

    ax.tick_params(axis="x", colors="#7f1d1d", rotation=90)
    ax.tick_params(axis="y", colors="#7f1d1d")
    for spine in ax.spines.values():
        spine.set_color("#fecaca")

    st.pyplot(fig)

    st.markdown("### Refill Behavior Snapshot")
    behavior_df = st.session_state.df[
        ["days_since_last_refill", "expected_days_supply", "prior_early_refill_count"]
    ]
    st.dataframe(behavior_df, use_container_width=True)