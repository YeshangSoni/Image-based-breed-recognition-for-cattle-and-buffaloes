import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- CONFIGURATION ---
# IMPORTANT: This path should be relative to where you run the streamlit app.
MODEL_PATH = 'models/my_final_model.keras'
CLASS_NAMES = ['AYRSHIRE', 'BANNI', 'GIR', 'GUERNSEY', 'HALLIKAR', 'HOLSTEIN_FRIESIAN', 'JERSEY', 'MURRAH', 'SAHIWAL', 'THARPARKAR']
IMAGE_SIZE = (224, 224)
# ---

# --- TRANSLATION AND CONTENT DATABASE ---
TRANSLATIONS = {
    "English": {
        "page_title": "Bharat Pashudhan AI",
        "app_title": "🇮🇳 Bharat Pashudhan AI",
        "app_subtitle": "An AI-powered tool for recognizing breeds of Indian cattle and buffaloes.",
        "file_uploader_label": "Upload an Image to Identify Breed",
        "image_caption": "Your Image",
        "spinner_text": "Analyzing Breed...",
        "prediction_header": "Prediction Result",
        "breed_info_header": "Breed Information",
        "origin_label": "🌍 Origin",
        "features_label": "⭐ Key Features",
        "use_label": "🥛 Primary Use",
        "tutorial_header": "💡 How to Get the Best Results",
        "tutorial_dos_header": "Do's ✅",
        "tutorial_donts_header": "Don'ts ❌",
        "tutorial_do_1": "Take a clear, side or full-body shot of the animal.",
        "tutorial_do_2": "Ensure only one animal is in the photo.",
        "tutorial_do_3": "Make sure the view is not obstructed.",
        "tutorial_dont_1": "Avoid blurry or out-of-focus images.",
        "tutorial_dont_2": "Avoid images where the animal is very far away.",
        "footer_text": "Made with ❤️ by Techno Alliance for SIH 2025",
        "feedback_question": "Was this prediction correct?",
        "feedback_correct": "Correct Prediction ✅",
        "feedback_incorrect": "Incorrect Prediction ❌",
        "feedback_thanks": "Thank you for your feedback!",
        "resources_header": "📞 Helpful Resources",
        "helpline_text": "National Animal Disease Control Helpline:",
        "website_header": "Official Govt. Portal:",
        "website_text": "Dept. of Animal Husbandry and Dairying"
    },
    "हिन्दी": {
        "page_title": "भारत पशुधन एआई",
        "app_title": "🇮🇳 भारत पशुधन एआई",
        "app_subtitle": "भारतीय मवेशियों और भैंसों की नस्लों को पहचानने के लिए एक एआई-संचालित उपकरण।",
        "file_uploader_label": "नस्ल की पहचान के लिए एक छवि अपलोड करें",
        "image_caption": "आपकी छवि",
        "spinner_text": "नस्ल का विश्लेषण किया जा रहा है...",
        "prediction_header": "भविष्यवाणी परिणाम",
        "breed_info_header": "नस्ल की जानकारी",
        "origin_label": "🌍 मूल",
        "features_label": "⭐ मुख्य विशेषताऐं",
        "use_label": "🥛 प्राथमिक उपयोग",
        "tutorial_header": "💡 सर्वोत्तम परिणाम कैसे प्राप्त करें",
        "tutorial_dos_header": "क्या करें ✅",
        "tutorial_donts_header": "क्या न करें ❌",
        "tutorial_do_1": "जानवर का स्पष्ट, साइड या पूरे शरीर का शॉट लें।",
        "tutorial_do_2": "सुनिश्चित करें कि फोटो में केवल एक ही जानवर हो।",
        "tutorial_do_3": "सुनिश्चित करें कि दृश्य बाधित न हो।",
        "tutorial_dont_1": "धुंधली या आउट-ऑफ-फोकस छवियों से बचें।",
        "tutorial_dont_2": "ऐसी छवियों से बचें जिनमें जानवर बहुत दूर हो।",
        "footer_text": "SIH 2025 के लिए टेक्नो एलायंस द्वारा ❤️ से बनाया गया",
        "feedback_question": "क्या यह भविष्यवाणी सही थी?",
        "feedback_correct": "सही भविष्यवाणी ✅",
        "feedback_incorrect": "गलत भविष्यवाणी ❌",
        "feedback_thanks": "आपकी प्रतिक्रिया के लिए धन्यवाद!",
        "resources_header": "📞 उपयोगी संसाधन",
        "helpline_text": "राष्ट्रीय पशु रोग नियंत्रण हेल्पलाइन:",
        "website_header": "आधिकारिक सरकारी पोर्टल:",
        "website_text": "पशुपालन और डेयरी विभाग"
    }
}
# (Breed info databases remain the same, redacted for brevity)
BREED_INFO_EN = { "AYRSHIRE": {"origin": "Ayrshire, Scotland", "features": "Distinctive red, brown, and white markings. Known for strong constitution.", "use": "Dairy"},"BANNI": {"origin": "Kutch, Gujarat, India", "features": "Resilient buffalo breed adapted to arid conditions. Tightly curled horns.", "use": "Dairy"},"GIR": {"origin": "Gir hills, Gujarat, India", "features": "Prominent, convex forehead and long, pendulous ears that curl at the tip.", "use": "Dairy"},"GUERNSEY": {"origin": "Isle of Guernsey (Channel Islands)", "features": "Known for its rich, golden-colored milk. Coat is fawn or red and white.", "use": "Dairy"},"HALLIKAR": {"origin": "Mysore, Karnataka, India", "features": "A draught breed with long, vertical horns that curve backward at the tips.", "use": "Draught"},"HOLSTEIN_FRIESIAN": {"origin": "Netherlands", "features": "World's highest-production dairy animal. Distinctive black-and-white markings.", "use": "Dairy"},"JERSEY": {"origin": "Isle of Jersey (Channel Islands)", "features": "Small dairy breed with a fawn-colored coat. Milk has very high butterfat content.", "use": "Dairy"},"MURRAH": {"origin": "Haryana and Punjab, India", "features": "Premier Indian dairy buffalo. Jet black with distinctive, tightly curled horns.", "use": "Dairy"},"SAHIWAL": {"origin": "Punjab region (India/Pakistan)", "features": "Heat-tolerant dairy breed. Typically reddish-dun with loose skin (dewlap).", "use": "Dairy"},"THARPARKAR": {"origin": "Tharparkar District (Pakistan/India)", "features": "Dual-purpose breed known for hardiness. Coat is typically white to grey.", "use": "Dairy & Draught"}}
BREED_INFO_HI = {"AYRSHIRE": {"origin": "आयरशायर, स्कॉटलैंड", "features": "विशिष्ट लाल, भूरे और सफेद निशान। मजबूत संविधान के लिए जाना जाता है।", "use": "दुग्धालय"},"BANNI": {"origin": "कच्छ, गुजरात, भारत", "features": "शुष्क परिस्थितियों के अनुकूल लचीला भैंस नस्ल। कसकर मुड़े हुए सींग।", "use": "दुग्धालय"},"GIR": {"origin": "गिर पहाड़ियाँ, गुजरात, भारत", "features": "उत्तल माथा और लंबे, लटकते हुए कान जो सिरे पर मुड़ जाते हैं।", "use": "दुग्धालय"},"GUERNSEY": {"origin": "ग्वेर्नसे द्वीप (चैनल द्वीप समूह)", "features": "अपने समृद्ध, सुनहरे रंग के दूध के लिए जाना जाता है। कोट आमतौर पर हल्का पीला या लाल और सफेद होता है।", "use": "दुग्धालय"},"HALLIKAR": {"origin": "मैसूर, कर्नाटक, भारत", "features": "लंबे, ऊर्ध्वाधर सींगों वाली एक मसौदा नस्ल जो सिरों पर पीछे की ओर मुड़ती है।", "use": "ड्राफ्ट"},"HOLSTEIN_FRIESIAN": {"origin": "नीदरलैंड", "features": "दुनिया का सबसे अधिक उत्पादन वाला डेयरी पशु। विशिष्ट काले और सफेद निशान।", "use": "दुग्धालय"},"JERSEY": {"origin": "जर्सी द्वीप (चैनल द्वीप समूह)", "features": "हल्के पीले रंग के कोट वाली छोटी डेयरी नस्ल। बहुत अधिक मक्खन वसा वाले दूध के लिए जाना जाता है।", "use": "दुग्धालय"},"MURRAH": {"origin": "हरियाणा और पंजाब, भारत", "features": "प्रमुख भारतीय डेयरी भैंस। विशिष्ट, कसकर मुड़े हुए सींगों के साथ जेट ब्लैक।", "use": "दुग्धालय"},"SAHIWAL": {"origin": "पंजाब क्षेत्र (भारत/पाकिस्तान)", "features": "गर्मी सहन करने वाली डेयरी नस्ल। आमतौर पर ढीली त्वचा (गलकम्बल) के साथ लाल-भूरा।", "use": "दुग्धालय"},"THARPARKAR": {"origin": "थारपारकर जिला (पाकिस्तान/भारत)", "features": "कठोरता के लिए जानी जाने वाली दोहरे उद्देश्य वाली नस्ल। कोट का रंग आमतौर पर सफेद से ग्रे होता है।", "use": "डेयरी और ड्राफ्ट"}}
BREED_NAME_TRANSLATIONS = {"AYRSHIRE": "ायरशायर", "BANNI": "बन्नी", "GIR": "गिर", "GUERNSEY": "ग्वेर्नसे", "HALLIKAR": "हल्लीकर","HOLSTEIN_FRIESIAN": "होलस्टीन-फ्रेशियन", "JERSEY": "जर्सी", "MURRAH": "मुर्रा", "SAHIWAL": "साहीवाल","THARPARKAR": "थारपारकर"}
# ---

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_keras_model(model_path):
    """Loads the trained Keras model by rebuilding the architecture and loading weights."""
    try:
        base_model = tf.keras.applications.MobileNetV2(input_shape=IMAGE_SIZE + (3,), include_top=False, weights='imagenet')
        base_model.trainable = False
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=IMAGE_SIZE + (3,)),
            tf.keras.layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])
        model.load_weights(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        return None

def predict(model, image_to_predict):
    """Takes a model and a PIL image, and returns the prediction."""
    img_resized = image_to_predict.resize(IMAGE_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_batch = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_batch)
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence_score = np.max(predictions[0])
    return predicted_class_name, confidence_score

# --- MAIN APP LOGIC ---

# Initialize session state
if 'language' not in st.session_state:
    st.session_state.language = 'English'
if 'feedback_given' not in st.session_state:
    st.session_state.feedback_given = False

# --- PAGE CONFIG AND STYLING ---
st.set_page_config(page_title="Bharat Pashudhan AI", page_icon="🐄", layout="wide")

st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    .result-card {
        background-color: #1E293B;
        border: 1px solid #384251;
        border-radius: 10px;
        padding: 25px;
        margin-top: 20px;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
    }
    .result-card h3 {
        color: #38BDF8; /* Light Blue */
        margin-bottom: 15px;
    }
    .prediction {
        font-size: 2.2rem;
        font-weight: bold;
        color: #4ADE80; /* Bright Green */
        text-align: center;
    }
    .confidence {
        font-size: 1.2rem;
        color: #A1A1AA;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-item {
        font-size: 1rem;
        margin-bottom: 10px;
    }
    .info-item strong {
        color: #93C5FD;
    }
    .stExpander {
        background-color: #1E293B;
        border-radius: 10px;
    }
    .st-emotion-cache-1hver42 {
        background-color: #1E293B;
    }
    .sidebar-content {
        background-color: #1E293B;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.title("⚙️ Settings")
    selected_language = st.selectbox('Select Language / भाषा चुनें', ('English', 'हिन्दी'))
    if st.session_state.language != selected_language:
        st.session_state.language = selected_language
        st.rerun()
    
    st.write("---")
    
    # --- UPDATED: HELPFUL RESOURCES SECTION ---
    lang = TRANSLATIONS[st.session_state.language]
    st.subheader(lang["resources_header"])
    st.markdown(f"**{lang['helpline_text']}**")
    st.markdown(f"<h3 style='text-align: center; color: #4ADE80;'>1962</h3>", unsafe_allow_html=True)
    st.markdown(f"**{lang['website_header']}**")
    st.markdown(f"[{lang['website_text']}](https://dahd.gov.in/)", unsafe_allow_html=False)


# --- Load translations for the main page ---
lang = TRANSLATIONS[st.session_state.language]
breed_info_db = BREED_INFO_EN if st.session_state.language == 'English' else BREED_INFO_HI

# --- HEADER ---
st.title(lang["app_title"])
st.markdown(f"<p style='font-size: 1.2rem; color: #A1A1AA;'>{lang['app_subtitle']}</p>", unsafe_allow_html=True)
st.write("---")

# --- TUTORIAL EXPANDER ---
with st.expander(lang["tutorial_header"]):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(lang["tutorial_dos_header"])
        st.markdown(f"- {lang['tutorial_do_1']}")
        st.markdown(f"- {lang['tutorial_do_2']}")
        st.markdown(f"- {lang['tutorial_do_3']}")
    with col2:
        st.subheader(lang["tutorial_donts_header"])
        st.markdown(f"- {lang['tutorial_dont_1']}")
        st.markdown(f"- {lang['tutorial_dont_2']}")
        
st.write("") # Add some space

# Load the AI model
model = load_keras_model(MODEL_PATH)

# --- UPLOADER AND PREDICTION ---
uploaded_file = st.file_uploader(lang["file_uploader_label"], type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # ** BUG FIX **
    # Create a unique key for the file to correctly handle state
    current_file_key = f"{uploaded_file.name}-{uploaded_file.size}"
    if 'file_key' not in st.session_state or st.session_state.file_key != current_file_key:
        st.session_state.file_key = current_file_key
        st.session_state.feedback_given = False

    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([2, 3])
    
    with col1:
        st.image(image, caption=lang["image_caption"], use_container_width=True, clamp=True)
    
    with col2:
        if model:
            with st.spinner(lang["spinner_text"]):
                predicted_breed, confidence = predict(model, image)
                
                display_breed_name = predicted_breed
                if st.session_state.language == 'हिन्दी' and predicted_breed in BREED_NAME_TRANSLATIONS:
                    display_breed_name = BREED_NAME_TRANSLATIONS[predicted_breed]

                st.markdown('<div class="result-card">', unsafe_allow_html=True)
                st.markdown(f"<h3>{lang['prediction_header']}</h3>", unsafe_allow_html=True)
                st.markdown(f"<p class='prediction'>{display_breed_name}</p>", unsafe_allow_html=True)
                st.markdown(f"<p class='confidence'>{confidence:.2%}</p>", unsafe_allow_html=True)
                
                if predicted_breed in breed_info_db:
                    st.write("---")
                    info = breed_info_db[predicted_breed]
                    st.markdown(f"<h3>{lang['breed_info_header']}</h3>", unsafe_allow_html=True)
                    st.markdown(f"<div class='info-item'><strong>{lang['origin_label']}:</strong> {info['origin']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='info-item'><strong>{lang['features_label']}:</strong> {info['features']}</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='info-item'><strong>{lang['use_label']}:</strong> {info['use']}</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

            # --- FEEDBACK UI ---
            st.write("") # Add some space
            if not st.session_state.feedback_given:
                st.write(lang["feedback_question"])
                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button(lang["feedback_correct"], use_container_width=True):
                        st.session_state.feedback_given = True
                        st.rerun()
                with btn_col2:
                    if st.button(lang["feedback_incorrect"], use_container_width=True):
                        st.session_state.feedback_given = True
                        st.rerun()
            
            if st.session_state.feedback_given:
                st.success(lang["feedback_thanks"])
                
        else:
            st.warning(lang["prediction_warning"])

# --- FOOTER ---
st.write("---")
st.markdown(f"<p style='text-align: center; color: #A1A1AA;'>{lang['footer_text']}</p>", unsafe_allow_html=True)


