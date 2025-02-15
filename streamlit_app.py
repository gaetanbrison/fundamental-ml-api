import streamlit_shadcn_ui as ui
from streamlit_option_menu import option_menu
import streamlit as st
import pandas as pd

import base64
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.datasets import load_iris, load_wine, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn import metrics as mt
import plotly.express as px
import streamlit as st
from PIL import Image
import altair as alt
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px


import plotly.express as px
import streamlit.components.v1 as components
import streamlit as st
from pycaret.datasets import get_data
from pycaret.regression import *
import time
import matplotlib.pyplot as plt

import streamlit as st
from pycaret.regression import *
from pycaret.datasets import get_data
import pandas as pd
import time


import time
import sys
import io


### The st.title() function sets the title of the Streamlit application


import streamlit as st

# Title with logo
import streamlit as st
import base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert local image to base64
image_path = "icon.png"  # Update with your image path
image_base64 = get_base64_image(image_path)

# Display title with a logo on the right with reduced spacing
st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
        <h1 style="margin: 0; flex-grow: 1;">Welcome to Fundamental</h1>
        <img src="data:image/png;base64,{image_base64}" width="80" style="margin-left: 10px;">
    </div>
    """,
    unsafe_allow_html=True
)

st.write(" ")
st.write(" ")
### menu bar

selected = option_menu(
  menu_title = None,
  options = ["💽 01 Data","📊 02 Viz","⚡️ 03 Pred"],
  default_index = 0,
  orientation = "horizontal",

)




import streamlit as st
import secrets

# Initialize session state if not present
if "api_keys" not in st.session_state:
    st.session_state.api_keys = {}

st.sidebar.header("AWS API Key Generator")

# User inputs their email
email = st.sidebar.text_input("Enter your email:")

if email:
    # Check if the email already has an API key
    if email not in st.session_state.api_keys:
        st.session_state.api_keys[email] = "AKIA" + secrets.token_hex(16).upper()

    # Display the API key
    st.sidebar.success(f"Your AWS API Key:\n{st.session_state.api_keys[email]}")



#load data
#@st.cache_resource(experimental_allow_widgets=True)
def get_dataset(select_dataset):
    if "Wine Quality 🍷" in select_dataset:
        df = pd.read_csv("wine_quality_red.csv")
    elif "Titanic 🛳️" in select_dataset: 
        df = sns.load_dataset('titanic')
        df = df.drop(['deck','embark_town','who'],axis=1)
    elif "Income 💵" in select_dataset:
        df = pd.read_csv("adult_income.csv")
    else:
        df = pd.read_csv("Student_Performance.csv")
    df = df.dropna()
    return select_dataset, df


DATA_SELECT = {
    "Regression": ["Income 💵", "Student Score 💯","Wine Quality 🍷"],
    "Classification": ["Wine Quality 🍷","Titanic 🛳️"]
}

MODELS = {
    "Linear Regression": LinearRegression,
    "Logistic Regression": LogisticRegression 
}
target_variable = {
    "Wine Quality 🍷": "quality",
    "Income 💵": "income",
    "Student Score 💯":"Performance Index",
    "Titanic 🛳️": "survived"
}


model_mode = st.sidebar.selectbox('🔎 Select Use Case',['Regression','Classification'])


select_data =  st.sidebar.selectbox('💾 Select Dataset',DATA_SELECT[model_mode])
select_dataset, df = get_dataset(select_data)


import streamlit as st
import pandas as pd

# Sidebar file uploader
st.sidebar.header("Upload CSV File")

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display success message
        st.sidebar.success("File uploaded successfully!")

        # Show file details
        st.sidebar.write(f"📁 File: `{uploaded_file.name}`")
        st.sidebar.write(f"🔢 Rows: `{df.shape[0]}` | Columns: `{df.shape[1]}`")

        # Show first few rows of the file
        st.write("### Preview of Uploaded CSV:")
        st.dataframe(df.head())

        # Optional: Provide a download button for the uploaded CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.sidebar.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name="uploaded_file.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")

else:
    st.sidebar.info("Upload a CSV file to see its content.")




if selected == "💽 01 Data":

    st.markdown("## :violet[Data Exploration 💽]")

    num = st.number_input('No. of Rows', 5, 10)
    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head':
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))
    

    st.text('(Rows,Columns)')
    st.write(df.shape)


    st.write("### Describe")
    st.dataframe(df.describe())

    with st.spinner("Cleaning dataset... This will take 3 seconds ⏳"):
        time.sleep(3)  # Simulate processing time
        cleaned_df = df.dropna()

    st.success("Cleaning complete! 🧹")



    import streamlit as st

    # Path to the uploaded HTML report
    html_file_path = "report.html"

    st.title("HTML Report Viewer")

    # Button to render the HTML file
    if st.button("Show Report"):
        with open(html_file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        
        # Display the HTML content inside an iframe
        st.components.v1.html(html_content, height=600, scrolling=True)



if selected == "📊 02 Viz":

    import streamlit as st
    import pandas as pd
    import plotly.express as px



    st.markdown("## :violet[Visualization 📊]")
        
    # Select dataset
    #dataset_name = st.sidebar.selectbox('💾 Select Dataset', ["Wine Quality 🍷", "Titanic 🛳️", "Student Score 💯", "Income 💵"])
    dataset_name, df = get_dataset(select_data)
        
    # Select only numeric columns
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    selected_vars = st.multiselect("Select variables for correlation matrix", numeric_columns, default=numeric_columns[:3])
        
    if len(selected_vars) > 1:
        tab_corr = st.tabs(["Correlation ⛖"])[0]
        tab_corr.subheader("Correlation Matrix ⛖")
            
        # Compute correlation
        corr = df[selected_vars].corr()
        fig = px.imshow(corr.values, x=corr.index, y=corr.columns, labels=dict(color="Correlation"))
        fig.layout.height = 700
        fig.layout.width = 700
        tab_corr.plotly_chart(fig, theme="streamlit", use_container_width=True)


 
if selected == "⚡️ 03 Pred":

    st.markdown("## :violet[ Live Model Prediction ⚡️]")


    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, r2_score, mean_absolute_error
    from pycaret.classification import setup as setup_clf, compare_models as compare_clf, tune_model as tune_clf
    from pycaret.regression import setup as setup_reg, compare_models as compare_reg, tune_model as tune_reg


    #dataset_choice = st.selectbox("Select Dataset", ["Wine Quality 🍷", "Titanic 🛳️", "Income 💵", "Student Score 💯"])

    # Load Dataset
    def load_data(dataset_choice):
        if dataset_choice == "Wine Quality 🍷":
            df = pd.read_csv("wine_quality_red.csv")
        elif dataset_choice == "Titanic 🛳️":
            df = sns.load_dataset('titanic').drop(['deck','embark_town','who'], axis=1)
        elif dataset_choice == "Income 💵":
            df = pd.read_csv("adult_income.csv")
        else:
            df = pd.read_csv("Student_Performance.csv")
        df = df.dropna()
        return df

    df = load_data(select_dataset)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    # Define Target Variable
    target_variable = {
        "Wine Quality 🍷": "quality",
        "Income 💵": "income",
        "Student Score 💯": "Performance Index",
        "Titanic 🛳️": "survived"
    }
    target = target_variable[select_dataset]

    # Splitting Data
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categorical variables if needed
    for col in X.select_dtypes(include=['object']).columns:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Select Mode
    mode = st.sidebar.radio("Select Mode", ["Prediction", "Hyperparameter Tuning"])
    dataset_choice = select_data

    if mode == "Prediction":
        # Determine task type (classification or regression)
        is_classification = dataset_choice in ["Wine Quality 🍷", "Titanic 🛳️"]
        
        # Model Selection
        if is_classification:
            model_choice = st.sidebar.selectbox("Select Model", ["Logistic Regression", "Random Forest Classifier"])
            model = LogisticRegression() if model_choice == "Logistic Regression" else RandomForestClassifier()
        else:
            model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest Regressor"])
            model = LinearRegression() if model_choice == "Linear Regression" else RandomForestRegressor()
        
        # Train Model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Display Metrics
        st.write("### Model Performance")
        if is_classification:
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            st.write(f"**Accuracy:** {accuracy:.2f}")
            st.write(f"**F1 Score:** {f1:.2f}")
            st.write(f"**Precision:** {precision:.2f}")
            st.write(f"**Recall:** {recall:.2f}")
        else:
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            st.write(f"**R² Score:** {r2:.2f}")
            st.write(f"**Mean Absolute Error:** {mae:.2f}")
        
        st.success("✅ Prediction completed successfully!")


        # Check if X_test is already a DataFrame. If not, convert it.
        if not isinstance(X_test, pd.DataFrame):
            # If X_test is a NumPy array or similar, create a DataFrame with default column names.
            X_test = pd.DataFrame(X_test, columns=[f"Feature_{i}" for i in range(X_test.shape[1])])

        # Create a copy of X_test to avoid modifying the original data.
        result_df = X_test.copy()

        # Add the predictions as a new column.
        result_df['Predicted'] = y_pred

        # ------------------------------
        # Create a download button for the CSV export
        # ------------------------------

        # Convert the DataFrame to CSV format and encode it to bytes.
        csv = result_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            label="Download Predictions CSV",
            data=csv,
            file_name='predictions.csv',
            mime='text/csv'
        )

    elif mode == "Hyperparameter Tuning":

    
        st.write("### 🔧 Hyperparameter Tuning with PyCaret")
        
        if dataset_choice in ["Wine Quality 🍷", "Titanic 🛳️"]:
            df2=df.sample(n=500)
            setup_clf(df2, target=target)
            st.success("✅ done")
            best_model = compare_clf()
            tuned_model = tune_clf(best_model)
        else:
            df2 = df.sample(n=500)
            setup(data=df2, target=target, session_id=123)
            st.success("✅ Setup Complete!")

            if st.button("🚀 Find Best Model"):
                st.write("### 🏆 Comparing Models... Please wait.")

                leaderboard_placeholder = st.empty()  # Placeholder for dynamic leaderboard updates
                progress_bar = st.progress(0)  # Progress bar for model comparison
                status_text = st.empty()  # Placeholder for live processing logs

                # Capture PyCaret's live output
                old_stdout = sys.stdout
                sys.stdout = mystdout = io.StringIO()

                with st.spinner("Comparing models live... ⏳"):
                    best_models = compare_models(n_select=5)  # Selecting top 5 models

                    # Simulate progress updates
                    for i in range(1, 101, 5):  # Update progress in increments
                        time.sleep(0.5)
                        progress_bar.progress(i / 100)  # Update progress bar
                        logs = mystdout.getvalue().split("\n")[-5:]  # Get last 5 log lines
                        status_text.text("\n".join(logs))  # Update live logs
                        
                        leaderboard = pull()  # Get current leaderboard

                        def highlight_best(val, col, df):
                            """Highlight the lowest MSE/MAE and the highest R²"""
                            if col in ["MSE", "MAE"] and val == df[col].min():
                                return "background-color: yellow"
                            elif col == "R2" and val == df[col].max():
                                return "background-color: yellow"
                            return ""

                        leaderboard_placeholder.dataframe(
                            leaderboard.style.apply(lambda col: col.map(lambda val: highlight_best(val, col.name, leaderboard)), axis=0)
                        )

                sys.stdout = old_stdout  # Restore stdout after capturing logs

                st.success("✅ Model comparison complete!")

                # Display the best model found
                st.write("### 🏅 Best Model Found")
                st.write(best_models[0])  # Show the best model









