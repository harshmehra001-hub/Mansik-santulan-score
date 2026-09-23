import streamlit as st
import pandas as pd
import joblib


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="MindTrack AI",
    page_icon="🧠",
    layout="wide"
)


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load("Mental_Health_Model.pkl")


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title("🧠 MindTrack AI")

st.write(
    "Student Social Media & Mental Health Impact Analyzer"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("🧠 MindTrack AI")

    st.write(
        "An ML-based project for analyzing student "
        "lifestyle and social-media factors."
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write("Mental_Health_Model.pkl")

    st.divider()

    st.caption(
        "Python • Pandas • Machine Learning • Streamlit"
    )


# =========================================================
# MAIN COLUMNS
# =========================================================

left, right = st.columns(
    [1.2, 0.8],
    gap="large"
)


# =========================================================
# LEFT SIDE
# =========================================================

with left:

    st.header("👤 Student Information")

    st.caption(
        "Enter the student's information below."
    )


    # -----------------------------------------------------
    # ACADEMIC & PERSONAL
    # -----------------------------------------------------

    with st.expander(
        "🎓 Academic & Personal Information",
        expanded=True
    ):

        col1, col2 = st.columns(2)

        with col1:

            age = st.number_input(
                "Age",
                min_value=10,
                max_value=80,
                value=21
            )

            gender = st.selectbox(
                "Gender",
                [
                    "Male",
                    "Female"
                ]
            )

            country = st.selectbox(
                "Country",
                [
                    "India",
                    "USA",
                    "Canada",
                    "Australia",
                    "UK",
                    "Germany",
                    "Bangladesh",
                    "Brazil",
                    "Japan",
                    "South Korea",
                    "France",
                    "Spain",
                    "Italy",
                    "Other"
                ]
            )

        with col2:

            academic_level = st.selectbox(
                "Academic Level",
                [
                    "High School",
                    "Undergraduate",
                    "Graduate"
                ]
            )

            platform = st.selectbox(
                "Most Used Platform",
                [
                    "Instagram",
                    "Facebook",
                    "YouTube",
                    "TikTok",
                    "Twitter",
                    "LinkedIn",
                    "Snapchat",
                    "WhatsApp",
                    "Other"
                ]
            )

            purpose = st.selectbox(
                "Purpose of Social Media",
                [
                    "Entertainment",
                    "Education",
                    "Networking",
                    "News"
                ]
            )


    # -----------------------------------------------------
    # DIGITAL HABITS
    # -----------------------------------------------------

    with st.expander(
        "📱 Digital Habits",
        expanded=True
    ):

        col1, col2 = st.columns(2)

        with col1:

            usage_hours = st.number_input(
                "Daily Social Media Usage (hours)",
                min_value=0.0,
                max_value=24.0,
                value=5.0,
                step=0.5
            )

            daily_unlocks = st.number_input(
                "Daily Phone Unlocks",
                min_value=0,
                max_value=1000,
                value=120
            )

        with col2:

            study_hours = st.number_input(
                "Study Hours / Day",
                min_value=0.0,
                max_value=24.0,
                value=4.0,
                step=0.5
            )

            physical_activity = st.number_input(
                "Physical Activity Hours / Day",
                min_value=0.0,
                max_value=24.0,
                value=1.0,
                step=0.5
            )


    # -----------------------------------------------------
    # HEALTH
    # -----------------------------------------------------

    with st.expander(
        "😴 Health & Stress",
        expanded=True
    ):

        col1, col2 = st.columns(2)

        with col1:

            sleep_hours = st.number_input(
                "Sleep Hours / Night",
                min_value=0.0,
                max_value=24.0,
                value=7.0,
                step=0.5
            )

        with col2:

            stress_level = st.selectbox(
                "Stress Level",
                [
                    "Low",
                    "Medium",
                    "High",
                    "Very High"
                ]
            )


    st.write("")

    predict_button = st.button(
        "🔮 Predict Mental Health Score",
        type="primary",
        use_container_width=True
    )


# =========================================================
# RIGHT SIDE
# =========================================================

with right:

    st.header("📊 Prediction Result")

    st.caption(
        "Your ML prediction will appear here."
    )


    # =====================================================
    # BEFORE PREDICTION
    # =====================================================

    if not predict_button:

        st.info(
            "👈 Fill in the student information and "
            "click **Predict Mental Health Score**."
        )

        st.subheader("🔍 Factors Analyzed")

        st.write("📱 Social Media Usage")

        st.write("📞 Daily Phone Unlocks")

        st.write("📚 Study Hours")

        st.write("🏃 Physical Activity")

        st.write("😴 Sleep Duration")

        st.write("😰 Stress Level")

        st.write("🎓 Academic Information")


    # =====================================================
    # AFTER PREDICTION
    # =====================================================

    else:

        # -------------------------------------------------
        # CREATE INPUT DATA
        # -------------------------------------------------

        input_data = pd.DataFrame({

            "Age": [age],

            "Gender": [gender],

            "Country": [country],

            "Academic_Level": [academic_level],

            "Most_Used_Platform": [platform],

            "Purpose_Of_Use": [purpose],

            "Avg_Daily_Usage_Hours": [
                usage_hours
            ],

            "Daily_Unlocks": [
                daily_unlocks
            ],

            "Study_Hours": [
                study_hours
            ],

            "Physical_Activity_Hours": [
                physical_activity
            ],

            "Sleep_Hours_Per_Night": [
                sleep_hours
            ],

            "Stress_Level": [
                stress_level
            ]

        })


        # -------------------------------------------------
        # PREDICTION
        # -------------------------------------------------

        try:

            prediction = model.predict(
                input_data
            )

            score = float(
                prediction[0]
            )


            # Score between 0 and 10
            score = max(
                0,
                min(
                    10,
                    score
                )
            )


            # -------------------------------------------------
            # CATEGORY
            # -------------------------------------------------

            if score >= 7:

                level = "😊 Good"

                message = (
                    "The model estimates a relatively "
                    "positive score."
                )

            elif score >= 5:

                level = "😐 Moderate"

                message = (
                    "The model estimates a moderate "
                    "score."
                )

            else:

                level = "⚠️ Needs Attention"

                message = (
                    "The model estimates that this "
                    "profile may need attention."
                )


            # -------------------------------------------------
            # SCORE
            # -------------------------------------------------

            st.subheader(
                "🧠 Mental Health Score"
            )

            st.metric(
                "Predicted Score",
                f"{score:.2f} / 10"
            )


            # -------------------------------------------------
            # PROGRESS
            # -------------------------------------------------

            st.progress(
                score / 10
            )


            # -------------------------------------------------
            # STATUS
            # -------------------------------------------------

            st.success(
                f"Prediction Level: {level}"
            )

            st.info(
                message
            )


            # -------------------------------------------------
            # STUDENT SNAPSHOT
            # -------------------------------------------------

            st.subheader(
                "📌 Student Snapshot"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "📱 Social Media",
                    f"{usage_hours:.1f} hrs"
                )

            with col2:

                st.metric(
                    "😴 Sleep",
                    f"{sleep_hours:.1f} hrs"
                )


            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "📚 Study",
                    f"{study_hours:.1f} hrs"
                )

            with col2:

                st.metric(
                    "🏃 Activity",
                    f"{physical_activity:.1f} hrs"
                )


            # -------------------------------------------------
            # INPUT DETAILS
            # -------------------------------------------------

            with st.expander(
                "🔎 View Input Details"
            ):

                st.dataframe(
                    input_data,
                    use_container_width=True,
                    hide_index=True
                )


            # -------------------------------------------------
            # DISCLAIMER
            # -------------------------------------------------

            st.warning(
                "⚠️ This is an ML-based educational/project "
                "prediction. It is not a medical diagnosis."
            )


        # -------------------------------------------------
        # ERROR
        # -------------------------------------------------

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.write(
                "There is a mismatch between the input "
                "columns and the columns used while "
                "training the model."
            )

            st.code(
                str(e)
            )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "🧠 MindTrack AI • Python • Streamlit • Machine Learning"
)