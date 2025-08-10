import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="🌱 Plant Doctor - Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Dynamic theme-aware CSS
st.markdown("""
<style>
    /* CSS Variables for theme colors */
    :root {
        /* Light theme colors */
        --bg-primary: #ffffff;
        --bg-secondary: #f8f9fa;
        --bg-tertiary: #e9ecef;
        --text-primary: #333333;
        --text-secondary: #666666;
        --text-muted: #999999;
        --border-color: #dee2e6;
        --shadow-color: rgba(0,0,0,0.1);
        --success-color: #2E7D32;
        --success-bg: #E8F5E8;
        --warning-color: #FF9800;
        --warning-bg: #FFF3E0;
        --danger-color: #F44336;
        --danger-bg: #FFEBEE;
        --info-color: #2196F3;
        --info-bg: #E3F2FD;
        --primary-color: #4CAF50;
        --primary-hover: #388E3C;
    }

    /* Dark theme colors */
    [data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Dark mode detection and application */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1a1a1a;
            --bg-secondary: #2d2d2d;
            --bg-tertiary: #404040;
            --text-primary: #ffffff;
            --text-secondary: #cccccc;
            --text-muted: #999999;
            --border-color: #404040;
            --shadow-color: rgba(0,0,0,0.3);
            --success-color: #66bb6a;
            --success-bg: #1b5e20;
            --warning-color: #ffb74d;
            --warning-bg: #e65100;
            --danger-color: #ef5350;
            --danger-bg: #c62828;
            --info-color: #64b5f6;
            --info-bg: #1565c0;
            --primary-color: #66bb6a;
            --primary-hover: #4caf50;
        }
    }

    /* Streamlit dark mode override */
    .stApp[data-testid="stAppViewContainer"] {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    .main .block-container {
        background-color: var(--bg-primary) !important;
        color: var(--text-primary) !important;
    }

    /* Ensure all text elements use theme colors */
    .stMarkdown, .stText, p, h1, h2, h3, h4, h5, h6, span, div {
        color: var(--text-primary) !important;
    }

    /* Main header styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--success-color) !important;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        border-bottom: 3px solid var(--primary-color);
        text-shadow: 1px 1px 2px var(--shadow-color);
    }
    
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--success-color) !important;
        margin-bottom: 1rem;
        text-shadow: 1px 1px 2px var(--shadow-color);
    }
    
    /* Info box styling */
    .info-box {
        background-color: var(--success-bg);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid var(--primary-color);
        margin: 1rem 0;
        color: var(--text-primary) !important;
        box-shadow: 0 2px 8px var(--shadow-color);
    }
    
    .info-box h4 {
        color: var(--success-color) !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .info-box p, .info-box ul, .info-box li {
        color: var(--text-primary) !important;
        font-weight: 400;
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background-color: var(--bg-secondary);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background-color: var(--bg-tertiary);
        border-color: var(--primary-hover);
    }
    
    /* Result box styling */
    .result-box {
        background-color: var(--warning-bg);
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid var(--warning-color);
        margin: 1rem 0;
        box-shadow: 0 2px 8px var(--shadow-color);
    }
    
    .healthy-result {
        background-color: var(--success-bg);
        border-left: 5px solid var(--success-color);
    }
    
    .disease-result {
        background-color: var(--danger-bg);
        border-left: 5px solid var(--danger-color);
    }
    
    /* Step box styling */
    .step-box {
        background-color: var(--bg-secondary);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--info-color);
        color: var(--text-primary) !important;
        font-weight: 500;
        box-shadow: 0 1px 4px var(--shadow-color);
    }
    
    /* Metric card styling */
    .metric-card {
        background-color: var(--bg-secondary);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px var(--shadow-color);
        text-align: center;
        margin: 0.5rem;
        border: 1px solid var(--border-color);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    
    .metric-card h4 {
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-card p {
        color: var(--text-primary) !important;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: var(--primary-color);
        color: white !important;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px var(--shadow-color);
        min-width: 200px;
        margin: 0 auto;
        display: block;
    }
    
    .stButton > button:hover {
        background-color: var(--primary-hover);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px var(--shadow-color);
    }
    
    /* Button container for centering */
    .button-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 2rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: var(--bg-secondary);
        border-radius: 8px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--bg-tertiary);
        border-radius: 6px;
        color: var(--text-secondary) !important;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--primary-color);
        color: white !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background-color: var(--bg-secondary);
        border: 2px dashed var(--border-color);
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: var(--primary-color);
        background-color: var(--bg-tertiary);
    }
    
    /* Image styling */
    .stImage > img {
        max-height: 400px;
        object-fit: contain;
        border-radius: 10px;
        box-shadow: 0 4px 8px var(--shadow-color);
        display: block;
        margin: 0 auto;
        background-color: var(--bg-secondary);
        padding: 10px;
        max-width: 100%;
    }
    
    .stImage {
        text-align: center;
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100%;
    }
    
    .stImage > div {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        max-width: 100%;
    }
    
    .stImage img {
        margin: 0 auto;
        display: block;
        max-width: 100%;
        height: auto;
    }
    
    /* Custom image container for better centering */
    .image-display-container {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        margin: 2rem auto;
        padding: 0;
        text-align: center;
        position: relative;
        left: 60%;
        transform: translateX(50%);
    }
    
    .image-display-container > div {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100%;
        max-width: 500px;
        margin: 0 auto;
    }
    
    /* Image container */
    .image-container {
        max-width: 100%;
        overflow: hidden;
        border-radius: 10px;
        background-color: var(--bg-secondary);
        padding: 10px;
        border: 1px solid var(--border-color);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: var(--primary-color);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Text input styling */
    .stTextInput > div > div > input {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Number input styling */
    .stNumberInput > div > div > input {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Checkbox styling */
    .stCheckbox > div > div {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
    }
    
    /* Radio button styling */
    .stRadio > div > div {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
    }
    
    /* Multiselect styling */
    .stMultiSelect > div > div {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Slider styling */
    .stSlider > div > div > div > div {
        background-color: var(--primary-color);
    }
    
    /* Date input styling */
    .stDateInput > div > div > input {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Time input styling */
    .stTimeInput > div > div > input {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Color picker styling */
    .stColorPicker > div > div {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary);
        color: var(--text-primary) !important;
        border: 1px solid var(--border-color);
    }
    
    /* Alert styling */
    .stAlert {
        background-color: var(--info-bg);
        color: var(--text-primary) !important;
        border: 1px solid var(--info-color);
    }
    
    /* Success alert */
    .stAlert[data-baseweb="notification"] {
        background-color: var(--success-bg);
        border-color: var(--success-color);
    }
    
    /* Warning alert */
    .stAlert[data-baseweb="notification"][data-severity="warning"] {
        background-color: var(--warning-bg);
        border-color: var(--warning-color);
    }
    
    /* Error alert */
    .stAlert[data-baseweb="notification"][data-severity="error"] {
        background-color: var(--danger-bg);
        border-color: var(--danger-color);
    }
    
    /* Info alert */
    .stAlert[data-baseweb="notification"][data-severity="info"] {
        background-color: var(--info-bg);
        border-color: var(--info-color);
    }
    
    /* Code block styling */
    .stCodeBlock {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    
    /* JSON viewer styling */
    .stJson {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    
    /* Chart styling */
    .stPlotlyChart, .stAltairChart, .stVegaLiteChart {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
        padding: 10px;
    }
    
    /* Map styling */
    .stMap {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    
    /* Video styling */
    .stVideo {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    
    /* Audio styling */
    .stAudio {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-radius: 6px;
    }
    
    /* Balloons styling */
    .stBalloons {
        background-color: var(--bg-secondary);
    }
    
    /* Snow styling */
    .stSnow {
        background-color: var(--bg-secondary);
    }
    
    /* Sidebar navigation styling */
    .css-1d391kg {
        background-color: var(--bg-secondary);
    }
    
    /* Main content area */
    .main .block-container {
        background-color: var(--bg-primary);
        color: var(--text-primary);
    }
    
    /* Hide Streamlit elements */
    .stDeployButton {
        display: none;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
    
    /* Additional dark mode improvements */
    .stApp > header {
        background-color: var(--bg-secondary);
        border-bottom: 1px solid var(--border-color);
    }
    
    .stApp > footer {
        background-color: var(--bg-secondary);
        color: var(--text-secondary);
    }
    
    /* Improve contrast for better readability */
    .stMarkdown strong, .stMarkdown b {
        color: var(--text-primary);
    }
    
    .stMarkdown em, .stMarkdown i {
        color: var(--text-secondary);
    }
    
    /* Better focus states for accessibility */
    .stButton > button:focus,
    .stSelectbox > div > div:focus,
    .stTextInput > div > div > input:focus {
        outline: 2px solid var(--primary-color);
        outline-offset: 2px;
    }
    
    /* Smooth transitions for theme switching */
    * {
        transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease;
    }
    
    /* Ensure proper contrast ratios */
    .metric-card p {
        color: var(--text-primary) !important;
        font-weight: 500;
    }
    
    /* Better hover effects */
    .info-box:hover,
    .result-box:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px var(--shadow-color);
    }
    
    /* Loading spinner improvements */
    .stSpinner > div > div {
        border-color: var(--primary-color) transparent transparent transparent;
    }
    
    /* Better file uploader styling */
    .stFileUploader > div {
        background-color: var(--bg-secondary);
        border: 2px dashed var(--border-color);
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: var(--primary-color);
        background-color: var(--bg-tertiary);
    }
</style>
""", unsafe_allow_html=True)

# Model prediction function
@st.cache_resource
def load_model():
    """Load the trained model once and cache it"""
    try:
        model = tf.keras.models.load_model('trained_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def model_prediction(test_image):
    """Predict plant disease from uploaded image"""
    model = load_model()
    if model is None:
        return None
    
    try:
        # Load and preprocess image
        image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr])
        
        # Make prediction
        prediction = model.predict(input_arr)
        result_index = np.argmax(prediction)
        confidence = float(np.max(prediction))
        
        return result_index, confidence
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None, None

# Disease classes with farmer-friendly names
DISEASE_CLASSES = {
    0: {'name': 'Apple Scab', 'plant': 'Apple', 'status': 'disease', 'description': 'Fungal disease causing dark spots on leaves and fruit'},
    1: {'name': 'Black Rot', 'plant': 'Apple', 'status': 'disease', 'description': 'Fungal disease affecting fruit and leaves'},
    2: {'name': 'Cedar Apple Rust', 'plant': 'Apple', 'status': 'disease', 'description': 'Fungal disease with orange spots on leaves'},
    3: {'name': 'Healthy', 'plant': 'Apple', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    4: {'name': 'Healthy', 'plant': 'Blueberry', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    5: {'name': 'Powdery Mildew', 'plant': 'Cherry', 'status': 'disease', 'description': 'White powdery growth on leaves and stems'},
    6: {'name': 'Healthy', 'plant': 'Cherry', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    7: {'name': 'Gray Leaf Spot', 'plant': 'Corn', 'status': 'disease', 'description': 'Grayish-brown spots on corn leaves'},
    8: {'name': 'Common Rust', 'plant': 'Corn', 'status': 'disease', 'description': 'Rust-colored spots on corn leaves'},
    9: {'name': 'Northern Leaf Blight', 'plant': 'Corn', 'status': 'disease', 'description': 'Large brown lesions on corn leaves'},
    10: {'name': 'Healthy', 'plant': 'Corn', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    11: {'name': 'Black Rot', 'plant': 'Grape', 'status': 'disease', 'description': 'Black spots on grape leaves and fruit'},
    12: {'name': 'Black Measles', 'plant': 'Grape', 'status': 'disease', 'description': 'Dark spots on grape wood and leaves'},
    13: {'name': 'Leaf Blight', 'plant': 'Grape', 'status': 'disease', 'description': 'Brown spots and lesions on grape leaves'},
    14: {'name': 'Healthy', 'plant': 'Grape', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    15: {'name': 'Citrus Greening', 'plant': 'Orange', 'status': 'disease', 'description': 'Yellowing leaves and misshapen fruit'},
    16: {'name': 'Bacterial Spot', 'plant': 'Peach', 'status': 'disease', 'description': 'Dark spots on peach leaves and fruit'},
    17: {'name': 'Healthy', 'plant': 'Peach', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    18: {'name': 'Bacterial Spot', 'plant': 'Bell Pepper', 'status': 'disease', 'description': 'Dark spots on pepper leaves and fruit'},
    19: {'name': 'Healthy', 'plant': 'Bell Pepper', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    20: {'name': 'Early Blight', 'plant': 'Potato', 'status': 'disease', 'description': 'Dark brown spots with concentric rings'},
    21: {'name': 'Late Blight', 'plant': 'Potato', 'status': 'disease', 'description': 'Water-soaked lesions on potato leaves'},
    22: {'name': 'Healthy', 'plant': 'Potato', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    23: {'name': 'Healthy', 'plant': 'Raspberry', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    24: {'name': 'Healthy', 'plant': 'Soybean', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    25: {'name': 'Powdery Mildew', 'plant': 'Squash', 'status': 'disease', 'description': 'White powdery growth on squash leaves'},
    26: {'name': 'Leaf Scorch', 'plant': 'Strawberry', 'status': 'disease', 'description': 'Brown edges and spots on strawberry leaves'},
    27: {'name': 'Healthy', 'plant': 'Strawberry', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'},
    28: {'name': 'Bacterial Spot', 'plant': 'Tomato', 'status': 'disease', 'description': 'Dark spots on tomato leaves and fruit'},
    29: {'name': 'Early Blight', 'plant': 'Tomato', 'status': 'disease', 'description': 'Dark brown spots with concentric rings'},
    30: {'name': 'Late Blight', 'plant': 'Tomato', 'status': 'disease', 'description': 'Water-soaked lesions on tomato leaves'},
    31: {'name': 'Leaf Mold', 'plant': 'Tomato', 'status': 'disease', 'description': 'Yellow spots with gray mold on underside'},
    32: {'name': 'Septoria Leaf Spot', 'plant': 'Tomato', 'status': 'disease', 'description': 'Small brown spots with gray centers'},
    33: {'name': 'Spider Mites', 'plant': 'Tomato', 'status': 'disease', 'description': 'Tiny pests causing yellow stippling on leaves'},
    34: {'name': 'Target Spot', 'plant': 'Tomato', 'status': 'disease', 'description': 'Brown spots with target-like appearance'},
    35: {'name': 'Yellow Leaf Curl Virus', 'plant': 'Tomato', 'status': 'disease', 'description': 'Yellowing and curling of tomato leaves'},
    36: {'name': 'Mosaic Virus', 'plant': 'Tomato', 'status': 'disease', 'description': 'Mottled yellow and green pattern on leaves'},
    37: {'name': 'Healthy', 'plant': 'Tomato', 'status': 'healthy', 'description': 'Plant is healthy and disease-free'}
}

# Main app
def main():
    # Theme toggle in sidebar
    with st.sidebar:
        st.markdown("### 🎨 Theme Settings")
        st.markdown("Choose your preferred theme mode:")
        theme_mode = st.selectbox(
            "Theme Mode",
            ["Auto", "Light", "Dark"],
            help="Auto follows your system preference, or choose manually"
        )
        
        # Show current theme info
        if theme_mode == "Auto":
            st.info("🌓 Auto mode: Following your system preference")
        elif theme_mode == "Light":
            st.success("☀️ Light mode: Clean and bright interface")
        else:
            st.info("🌙 Dark mode: Easy on the eyes")
        
        # Apply theme based on selection
        if theme_mode == "Light":
            st.markdown("""
            <style>
                :root {
                    --bg-primary: #ffffff;
                    --bg-secondary: #f8f9fa;
                    --bg-tertiary: #e9ecef;
                    --text-primary: #333333;
                    --text-secondary: #666666;
                    --text-muted: #999999;
                    --border-color: #dee2e6;
                    --shadow-color: rgba(0,0,0,0.1);
                    --success-color: #2E7D32;
                    --success-bg: #E8F5E8;
                    --warning-color: #FF9800;
                    --warning-bg: #FFF3E0;
                    --danger-color: #F44336;
                    --danger-bg: #FFEBEE;
                    --info-color: #2196F3;
                    --info-bg: #E3F2FD;
                    --primary-color: #4CAF50;
                    --primary-hover: #388E3C;
                }
            </style>
            """, unsafe_allow_html=True)
        elif theme_mode == "Dark":
            st.markdown("""
            <style>
                :root {
                    --bg-primary: #1a1a1a;
                    --bg-secondary: #2d2d2d;
                    --bg-tertiary: #404040;
                    --text-primary: #ffffff;
                    --text-secondary: #cccccc;
                    --text-muted: #999999;
                    --border-color: #404040;
                    --shadow-color: rgba(0,0,0,0.3);
                    --success-color: #66bb6a;
                    --success-bg: #1b5e20;
                    --warning-color: #ffb74d;
                    --warning-bg: #e65100;
                    --danger-color: #ef5350;
                    --danger-bg: #c62828;
                    --info-color: #64b5f6;
                    --info-bg: #1565c0;
                    --primary-color: #66bb6a;
                    --primary-hover: #4caf50;
                }
            </style>
            """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🌱 Plant Doctor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: var(--text-secondary); margin-bottom: 2rem;">Smart Disease Detection for Your Crops</p>', unsafe_allow_html=True)
    
    # Navigation tabs
    tab1, tab2, tab3 = st.tabs(["🔍 Detect Disease", "📊 Statistics", "ℹ️ About"])
    
    with tab1:
        st.markdown('<h2 class="sub-header">Upload Plant Image</h2>', unsafe_allow_html=True)
        
        # Instructions
        st.markdown("""
        <div class="info-box">
            <h4>📋 How to use:</h4>
            <div class="step-box">1. Take a clear photo of the plant leaf you want to check</div>
            <div class="step-box">2. Upload the image below</div>
            <div class="step-box">3. Click "Analyze Plant" to get instant results</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Upload area
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png'],
                help="Upload a clear image of the plant leaf"
            )
        
        if uploaded_file is not None:
            st.markdown('<h3 style="text-align: center; margin: 2rem 0;">📸 Your Image</h3>', unsafe_allow_html=True)
            # Center image and button using columns
            img_col1, img_col2, img_col3 = st.columns([2, 3, 2])
            with img_col2:
                st.image(
                    uploaded_file,
                    width=400,
                    caption="Uploaded plant image",
                    use_container_width=False
                )
                analyze_button = st.button("🔬 Analyze Plant")
            if analyze_button:
                with st.spinner("🔍 Analyzing your plant..."):
                    result_index, confidence = model_prediction(uploaded_file)
                    if result_index is not None:
                        disease_info = DISEASE_CLASSES[result_index]
                        st.markdown('<h3 style="text-align: center; margin: 2rem 0;">📊 Analysis Results</h3>', unsafe_allow_html=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            plant_color = "var(--success-color)" if disease_info['status'] == 'healthy' else "var(--warning-color)"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🌿 Plant Type</h4>
                                <p style="font-size: 1.5rem; font-weight: bold; color: {plant_color};">{disease_info['plant']}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        with col2:
                            confidence_color = "var(--success-color)" if disease_info['status'] == 'healthy' else "var(--warning-color)"
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>🎯 Confidence</h4>
                                <p style="font-size: 1.5rem; font-weight: bold; color: {confidence_color};">{confidence:.1%}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        status_icon = "✅" if disease_info['status'] == 'healthy' else "⚠️"
                        status_color = "healthy-result" if disease_info['status'] == 'healthy' else "disease-result"
                        status_text = "HEALTHY" if disease_info['status'] == 'healthy' else "DISEASE DETECTED"
                        st.markdown(f"""
                        <div class="result-box {status_color}">
                            <h3 style="text-align: center; margin-bottom: 1rem;">
                                {status_icon} {status_text}
                            </h3>
                            <h4 style="text-align: center; color: var(--text-primary); margin-bottom: 1rem;">
                                {disease_info['name']}
                            </h4>
                            <p style="text-align: center; font-size: 1.1rem; color: var(--text-secondary);">
                                {disease_info['description']}
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        if disease_info['status'] == 'healthy':
                            st.markdown("""
                            <div class="info-box">
                                <h4>🎉 Great News!</h4>
                                <p>Your plant appears to be healthy. Continue with your current care routine:</p>
                                <ul>
                                    <li>Maintain regular watering schedule</li>
                                    <li>Ensure proper sunlight exposure</li>
                                    <li>Monitor for any changes</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="info-box">
                                <h4>💡 Recommended Actions:</h4>
                                <ul>
                                    <li>Isolate affected plants to prevent spread</li>
                                    <li>Remove and destroy severely infected leaves</li>
                                    <li>Consider appropriate treatment (organic or chemical)</li>
                                    <li>Monitor other plants for similar symptoms</li>
                                    <li>Consult with local agricultural expert if needed</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<h2 class="sub-header">📊 System Statistics</h2>', unsafe_allow_html=True)
        
        # Statistics cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h4>🌿 Plant Types</h4>
                <p style="font-size: 2rem; font-weight: bold; color: var(--info-color);">15</p>
                <p>Supported crops</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h4>🔍 Disease Types</h4>
                <p style="font-size: 2rem; font-weight: bold; color: var(--warning-color);">25</p>
                <p>Detectable diseases</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h4>📈 Accuracy</h4>
                <p style="font-size: 2rem; font-weight: bold; color: var(--success-color);">95%</p>
                <p>Detection accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h4>⚡ Speed</h4>
                <p style="font-size: 2rem; font-weight: bold; color: var(--info-color);">2s</p>
                <p>Average analysis time</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Supported plants
        st.markdown('<h3 style="margin: 2rem 0;">🌱 Supported Plants</h3>', unsafe_allow_html=True)
        
        plants = [
            "🍎 Apple", "🫐 Blueberry", "🍒 Cherry", "🌽 Corn", "🍇 Grape", 
            "🍊 Orange", "🍑 Peach", "🫑 Bell Pepper", "🥔 Potato", "🫐 Raspberry",
            "🫘 Soybean", "🎃 Squash", "🍓 Strawberry", "🍅 Tomato"
        ]
        
        cols = st.columns(4)
        for i, plant in enumerate(plants):
            cols[i % 4].markdown(f"<div class='step-box'>{plant}</div>", unsafe_allow_html=True)
    
    with tab3:
        st.markdown('<h2 class="sub-header">ℹ️ About Plant Doctor</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <h4>🎯 Our Mission</h4>
            <p>Plant Doctor helps farmers quickly identify plant diseases using advanced AI technology. 
            Our goal is to protect your crops and improve your harvest yields.</p>
        </div>
        
        <div class="info-box">
            <h4>🔬 How It Works</h4>
            <p>Our system uses a deep learning model trained on over 87,000 images of healthy and diseased plants. 
            It can identify 38 different plant conditions across 15 crop types with high accuracy.</p>
        </div>
        
        <div class="info-box">
            <h4>📊 Dataset Information</h4>
            <ul>
                <li><strong>Training Images:</strong> 70,295 images</li>
                <li><strong>Validation Images:</strong> 17,572 images</li>
                <li><strong>Test Images:</strong> 33 images</li>
                <li><strong>Total Classes:</strong> 38 (including healthy plants)</li>
            </ul>
        </div>
        
        <div class="info-box">
            <h4>💡 Tips for Best Results</h4>
            <ul>
                <li>Take photos in good lighting</li>
                <li>Focus on the affected leaves</li>
                <li>Ensure the image is clear and not blurry</li>
                <li>Include both healthy and diseased parts if possible</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
