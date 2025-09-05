import streamlit as st
import requests
from PIL import Image
import io
import os

# Configure page
st.set_page_config(
    page_title="🚜 RobCrop - Smart Farming Assistant",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# FINAL, AGGRESSIVE CSS TO REMOVE ALL EXTRA SPACING
st.markdown("""
<style>
    /* === REMOVE ALL UNWANTED STREAMLIT SPACING === */
    /* This targets the main app container */
    .main .block-container {
        padding-top: 1rem !important; /* Drastically reduce top padding */
        padding-bottom: 1rem !important;
    }
    /* This targets the vertical stacking of elements */
    [data-testid="stVerticalBlock"] {
        gap: 0.5rem !important; /* Reduces the gap between elements to a minimum */
    }

    /* === THEME AND COLOR STYLES (UNCHANGED) === */
    .stApp { background-color: #F0FFF0; }
    h1, h2, h3, h4, h5, h6 { color: #004d00 !important; }
    p, li, span, small { color: #2E8B57 !important; }
    .stButton > button { color: #ffffff !important; }
    
    .main-header {
        background: linear-gradient(90deg, #004d00 0%, #2E8B57 100%);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1rem;
    }
    .main-header h1 { color: white !important; }
    .main-header p { color: #C1FFC1 !important; }

    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        border-left: 6px solid #3CB371;
        margin-bottom: 1rem;
    }
    
    #MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)



# Improved CSS with better contrast and cleaner design
st.markdown("""
<style>
    /* Main theme colors - darker greens for better contrast */
    .stApp {
        background: linear-gradient(135deg, #f0f8f0 0%, #e8f5e8 100%);
    }
    
    /* Header styling with darker green */
    .main-header {
        background: linear-gradient(90deg, #1a3a1a 0%, #2d5a3d 50%, #1a3a1a 100%);
        padding: 2.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(26, 58, 26, 0.4);
        text-align: center;
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        font-weight: bold;
    }
    
    .main-header p {
        color: #b8d8b8;
        font-size: 1.3rem;
        margin: 0.8rem 0 0 0;
        font-weight: 500;
    }
    
    /* Instructions card with better contrast */
    .instructions-card {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #2d5a3d;
        margin-bottom: 2rem;
    }
    
    .instructions-card h3 {
        color: #1a3a1a;
        margin-top: 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .instructions-card ul {
        color: #2d5a3d;
        font-size: 1.1rem;
        line-height: 1.8;
    }
    
    .instructions-card li {
        margin: 0.8rem 0;
        font-weight: 500;
    }
    
    /* Upload section */
    .upload-section {
        background: white;
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #2d5a3d;
        margin-bottom: 2rem;
    }
    
    .upload-title {
        color: #1a3a1a;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    
    /* Enhanced button styling */
    .stButton > button {
        background: linear-gradient(45deg, #2d5a3d, #1a3a1a);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.8rem 2.5rem;
        font-weight: bold;
        font-size: 1.2rem;
        box-shadow: 0 6px 20px rgba(45, 90, 61, 0.4);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #1a3a1a, #2d5a3d);
        box-shadow: 0 8px 25px rgba(45, 90, 61, 0.6);
        transform: translateY(-3px);
    }
    
    /* Result cards with improved visibility */
    .result-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 5px solid #2d5a3d;
        margin: 1rem 0;
        text-align: center;
    }
    
    .healthy-result {
        border-left: 5px solid #28a745;
        background: #f8fff8;
    }
    
    .disease-result {
        border-left: 5px solid #dc3545;
        background: #fff8f8;
    }
    
    .warning-result {
        border-left: 5px solid #ffc107;
        background: #fffdf8;
    }
    
    /* File uploader enhancement */
    .stFileUploader > div > div {
        background: #f8fff8;
        border: 3px dashed #2d5a3d;
        border-radius: 12px;
        padding: 2.5rem;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #1a3a1a;
        background: #f0f8f0;
    }
    
    /* Image info styling */
    .image-info {
        background: #f0f8f0;
        padding: 1.5rem;
        border-radius: 10px;
        margin-top: 1rem;
        border: 1px solid #d0e0d0;
    }
    
    .image-info strong {
        color: #1a3a1a;
        font-size: 1.1rem;
    }
    
    .image-info span {
        color: #2d5a3d;
        font-weight: 500;
    }
    
    /* Treatment recommendation cards */
    .treatment-urgent {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #ffc107;
        margin: 1.5rem 0;
    }
    
    .treatment-urgent h4 {
        color: #856404;
        margin-top: 0;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .treatment-urgent p {
        color: #856404;
        font-size: 1.1rem;
        margin-bottom: 0;
        line-height: 1.6;
    }
    
    .treatment-healthy {
        background: linear-gradient(135deg, #d1f2eb 0%, #a8e6cf 100%);
        padding: 2rem;
        border-radius: 12px;
        border-left: 6px solid #17a2b8;
        margin: 1.5rem 0;
    }
    
    .treatment-healthy h4 {
        color: #0c5460;
        margin-top: 0;
        font-size: 1.3rem;
        font-weight: bold;
    }
    
    .treatment-healthy p {
        color: #0c5460;
        font-size: 1.1rem;
        margin-bottom: 0;
        line-height: 1.6;
    }
    
    /* Processing info */
    .processing-info {
        background: linear-gradient(135deg, #e7f3ff 0%, #cce7ff 100%);
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin-top: 1.5rem;
        border: 1px solid #b8d8ff;
    }
    
    .processing-info small {
        color: #004085;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #2d5a3d;
        padding: 2rem;
        margin-top: 3rem;
        background: white;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    .footer h4 {
        color: #1a3a1a;
        margin-bottom: 1rem;
        font-size: 1.4rem;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Dynamic API URL
API_URL = os.getenv('API_URL', 'http://localhost:8000')

# Main header
st.markdown("""
<div class="main-header">
    <h1>🚜 RobCrop - Smart Farming Assistant</h1>
    <p>🌱 AI-Powered Crop Disease Detection - Protecting Your Harvest with 96.45% Accuracy</p>
</div>
""", unsafe_allow_html=True)

# Instructions section
st.markdown("""
<div class="instructions-card">
    <h3>📋 How to Use RobCrop</h3>
    <ul>
        <li>📱 Take a clear photo of your crop leaf</li>
        <li>🔍 Make sure the leaf fills most of the frame</li>
        <li>☀️ Use good lighting (natural sunlight works best)</li>
        <li>📤 Upload the image below and click "Analyze Crop"</li>
        <li>⚡ Get instant results and treatment recommendations</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# File upload section
st.markdown('<div class="upload-section">', unsafe_allow_html=True)
st.markdown('<h3 class="upload-title">📸 Upload Your Crop Image</h3>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image of your crop leaf",
    type=['png', 'jpg', 'jpeg'],
    help="Supported formats: PNG, JPG, JPEG (Max size: 200MB)",
    label_visibility="collapsed"
)

st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Display uploaded image and analysis
    col_image, col_analysis = st.columns([1, 1], gap="large")
    
    with col_image:
        st.markdown("#### 📷 Your Crop Image")
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded crop image", use_container_width=True)
        
        # Image info with better styling
        st.markdown(f"""
        <div class="image-info">
            <strong>📋 Image Details:</strong><br>
            🏷️ <strong>Name:</strong> <span>{uploaded_file.name}</span><br>
            📐 <strong>Size:</strong> <span>{uploaded_file.size/1024:.1f} KB</span><br>
            🎨 <strong>Type:</strong> <span>{uploaded_file.type}</span>
        </div>
        """, unsafe_allow_html=True)
    
    with col_analysis:
        st.markdown("#### 🔬 Crop Health Analysis")
        
        if st.button("🌿 Analyze My Crop Health", type="primary"):
            # Progress bar for better UX
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            with st.spinner("🤖 Our AI farmer is examining your crop..."):
                try:
                    # Update progress
                    status_text.text("🔗 Connecting to RobCrop AI...")
                    progress_bar.progress(25)
                    
                    # Test backend connection
                    health_response = requests.get(f"{API_URL}/health", timeout=5)
                    
                    if health_response.status_code == 200:
                        status_text.text("✅ Connected! Analyzing crop...")
                        progress_bar.progress(50)
                        
                        # Prepare file for upload
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type or "image/jpeg"
                            )
                        }
                        
                        progress_bar.progress(75)
                        status_text.text("🧠 AI analyzing disease patterns...")
                        
                        # Make prediction request
                        response = requests.post(
                            f"{API_URL}/predict", 
                            files=files,
                            timeout=30
                        )
                        
                        progress_bar.progress(100)
                        status_text.text("✅ Analysis complete!")
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            # Display results with enhanced styling
                            st.markdown("---")
                            st.markdown("### 🎯 Diagnosis Results")
                            
                            # Result metrics
                            metric_col1, metric_col2, metric_col3 = st.columns(3)
                            
                            with metric_col1:
                                disease_name = result['disease'].replace('_', ' ').replace(',', ' ')
                                if 'healthy' in disease_name.lower():
                                    st.markdown(f"""
                                    <div class="result-card healthy-result">
                                        <h4>🌿 Plant Status</h4>
                                        <h3 style="color: #28a745; margin: 0.5rem 0;">{disease_name}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="result-card disease-result">
                                        <h4>🦠 Disease Found</h4>
                                        <h3 style="color: #dc3545; margin: 0.5rem 0;">{disease_name}</h3>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            with metric_col2:
                                confidence = result['confidence']
                                if confidence > 0.8:
                                    confidence_color = "#28a745"
                                    confidence_level = "Very High"
                                    card_class = "healthy-result"
                                elif confidence > 0.6:
                                    confidence_color = "#ffc107"
                                    confidence_level = "Good"
                                    card_class = "warning-result"
                                else:
                                    confidence_color = "#17a2b8"
                                    confidence_level = "Moderate"
                                    card_class = "result-card"
                                
                                st.markdown(f"""
                                <div class="result-card {card_class}">
                                    <h4>🎯 AI Confidence</h4>
                                    <h3 style="color: {confidence_color}; margin: 0.5rem 0;">{confidence:.1%}</h3>
                                    <p style="margin: 0; color: {confidence_color}; font-weight: bold;">{confidence_level}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with metric_col3:
                                action = result['action']
                                if action == 'SPRAY':
                                    st.markdown(f"""
                                    <div class="result-card disease-result">
                                        <h4>💊 Treatment Needed</h4>
                                        <h3 style="color: #dc3545; margin: 0.5rem 0;">🚨 {action}</h3>
                                        <p style="margin: 0; color: #dc3545; font-weight: bold;">RECOMMENDED</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="result-card healthy-result">
                                        <h4>✅ Plant Health</h4>
                                        <h3 style="color: #28a745; margin: 0.5rem 0;">🟢 NO TREATMENT</h3>
                                        <p style="margin: 0; color: #28a745; font-weight: bold;">NEEDED</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Treatment recommendations
                            st.markdown("### 💡 Treatment Recommendation")
                            
                            if 'treatment' in result:
                                if action == 'SPRAY':
                                    st.markdown(f"""
                                    <div class="treatment-urgent">
                                        <h4>⚠️ Immediate Action Required</h4>
                                        <p>{result['treatment']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                                else:
                                    st.markdown(f"""
                                    <div class="treatment-healthy">
                                        <h4>✅ Great News!</h4>
                                        <p>{result['treatment']}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            
                            # Processing information
                            if 'processing_time_ms' in result:
                                st.markdown(f"""
                                <div class="processing-info">
                                    <small>⚡ Analysis completed in {result['processing_time_ms']:.1f}ms using advanced AI technology</small>
                                </div>
                                """, unsafe_allow_html=True)
                                
                        else:
                            progress_bar.empty()
                            status_text.empty()
                            error_detail = response.json() if response.headers.get('content-type') == 'application/json' else response.text
                            st.error(f"❌ Analysis failed: {error_detail}")
                    else:
                        progress_bar.empty()
                        status_text.empty()
                        st.error(f"❌ Cannot connect to RobCrop AI system")
                        
                except requests.exceptions.ConnectionError:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("❌ **Connection Error**: Cannot reach RobCrop AI")
                    st.info("🔧 **Troubleshooting**: Make sure the AI backend is running")
                    
                except requests.exceptions.Timeout:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("⏱️ **Timeout Error**: Analysis is taking too long")
                    
                except Exception as e:
                    progress_bar.empty()
                    status_text.empty()
                    st.error(f"❌ **Error**: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div class="footer">
    <h4>🌱 RobCrop - Empowering Farmers with AI Technology</h4>
    <p><strong>Developed for Smart Agriculture • Protecting crops, Supporting farmers • SIH 2025</strong></p>
    <p><small>🔬 Powered by Deep Learning • 🚜 Built for Indian Farmers • 🌍 Sustainable Agriculture</small></p>
</div>
""", unsafe_allow_html=True)
