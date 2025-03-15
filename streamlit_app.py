#YOLO ONLY
import subprocess
import sys
# Function to install packages
def install_packages(requirements_file="requirements.txt"):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_file])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")

# Run installation
install_packages()
import streamlit as st
from PIL import Image
import numpy as np
import torch
from ultralytics import YOLO
import cv2
import io
import torchvision.transforms as transforms
import os
from datetime import timedelta
import torchvision.models as models
import time
import onnxruntime as ort

# Set page configuration
st.set_page_config(
    page_title="Precision Agriculture Solution",
    page_icon="🌱",
    layout="wide"
)

# Sidebar CSS for enhanced aesthetics
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #D3D3D3; /* Light grey background */
        border: 8px solid #228B22; /* Thick dark Green border */
        padding: 15px; /* Add padding inside the sidebar */
        border-radius: 10px; /* Rounded corners for a modern look */
    }
    [data-testid="stSidebar"] h1 {
        color: #333333; /* Darker grey for title text */
        text-align: center; /* Center the title */
        font-family: Arial, sans-serif; /* Clean font */
        font-size: 24px; /* Adjust font size */
        font-weight: bold; /* Make the title bold */
    }
    [data-testid="stSidebar"] .css-1v3fvcr {
        color: #4F4F4F; /* Medium grey for radio button text */
        font-family: 'Roboto', sans-serif; /* Modern font for options */
        font-size: 16px; /* Slightly larger text size */
    }
    [data-testid="stSidebar"] .css-1v3fvcr:hover {
        color: #FF6347; /* Tomato color on hover for navigation options */
        font-weight: bold; /* Bold hover effect */
    }
    
    /* Additional styling for pest detection page */
    .main {
        background-color: #f8f9fa;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .card {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 20px;
    }
    .header {
        font-size: 30px;
        font-weight: bold;
        color: #333;
        margin-bottom: 20px;
        text-align: center;
    }
    .subheader {
        font-size: 20px;
        font-weight: bold;
        color: #555;
        margin-bottom: 10px;
    }
    .upload-btn {
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .upload-icon {
        margin-right: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title("Navigation Pane")
page = st.sidebar.radio(
    "Go to", 
    ["Main Menu", "Pest Detection"]
)



pest_info = {
    "ant": {
        "description": "Small insects that live in colonies and can damage crops by farming aphids or directly damaging plants.",
        "danger_level": "Medium",
        "control_methods": "Use diatomaceous earth around plants, set up ant baits, or create a barrier with cinnamon or coffee grounds."
    },
    "aphids": {
        "description": "Small sap-sucking insects that can transmit plant viruses and cause leaf curling or yellowing.",
        "danger_level": "High for vegetables and ornamentals",
        "control_methods": "Spray with neem oil, introduce ladybugs or lacewings, or use insecticidal soap."
    },
    "cricket": {
        "description": "Jumping insects that can feed on seedlings and young plants, especially at night.",
        "danger_level": "Medium",
        "control_methods": "Use diatomaceous earth, set up cricket traps, or introduce natural predators like birds."
    },
    "fruitfly": {
        "description": "Small flies that lay eggs in ripening fruit, causing decay and spoilage.",
        "danger_level": "High for fruit crops",
        "control_methods": "Use apple cider vinegar traps, harvest fruit early, or cover plants with fine mesh."
    },
    "leafbeetle": {
        "description": "Beetles that chew leaves, creating holes and reducing plant vigor.",
        "danger_level": "Medium to High",
        "control_methods": "Handpick beetles, use neem oil spray, or introduce beneficial nematodes."
    },
    "grasshopper": {
        "description": "Large jumping insects that can consume large amounts of plant material quickly.",
        "danger_level": "High during outbreaks",
        "control_methods": "Use row covers, apply garlic spray, or introduce birds and other natural predators."
    },
    "leafhopper": {
        "description": "Small, wedge-shaped insects that suck plant sap and can spread plant diseases.",
        "danger_level": "Medium to High",
        "control_methods": "Use sticky traps, apply insecticidal soap, or introduce beneficial insects like parasitic wasps."
    },
    "mites": {
        "description": "Tiny arachnids that suck plant sap, causing stippling, discoloration, and webbing on leaves.",
        "danger_level": "High for many crops",
        "control_methods": "Spray plants with water, apply neem oil, or use predatory mites."
    },
    "armyworm": {
        "description": "Caterpillars that feed on foliage and can move in large groups, causing extensive damage.",
        "danger_level": "Very High during infestations",
        "control_methods": "Apply Bacillus thuringiensis (Bt), introduce natural predators, or use row covers for protection."
    },
    "weevil": {
        "description": "Beetles with distinctive snouts that feed on various plant parts and lay eggs in plant tissue.",
        "danger_level": "Medium to High",
        "control_methods": "Apply beneficial nematodes to soil, use sticky traps, or practice crop rotation."
    },
    "stinkbug": {
        "description": "Shield-shaped bugs that pierce plant tissues to feed, causing deformed fruits and vegetables.",
        "danger_level": "High for fruit and vegetable crops",
        "control_methods": "Use kaolin clay spray, set up pheromone traps, or introduce parasitic wasps."
    },
    "slug": {
        "description": "Soft-bodied mollusks that create holes in leaves and fruits, especially in wet conditions.",
        "danger_level": "Medium to High",
        "control_methods": "Set up beer traps, apply diatomaceous earth around plants, or use copper tape barriers."
    },
    "snail": {
        "description": "Similar to slugs but with shells, they feed on plant material and create holes in leaves.",
        "danger_level": "Medium",
        "control_methods": "Hand pick at night, set up beer traps, or create copper barriers around plants."
    },
    "whiteflies": {
        "description": "Small, white flying insects that suck plant sap and excrete honeydew, promoting sooty mold growth.",
        "danger_level": "High for many crops",
        "control_methods": "Use yellow sticky traps, apply insecticidal soap, or introduce natural predators like ladybugs."
    },
    "thrips": {
        "description": "Tiny, slender insects that rasp plant surfaces and suck sap, causing silvering and distortion.",
        "danger_level": "High for ornamentals and some vegetables",
        "control_methods": "Use blue sticky traps, apply neem oil, or introduce beneficial insects like minute pirate bugs."
    }
}

# Function to load YOLO model with caching
@st.cache_resource
def load_yolo_model():
    try:
        model_path = "yolo_detectorv9.pt"  # Use your existing model path
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# List of pest classes
pest_classes = [
    'ant', 'aphids', 'armyworm', 'cricket', 'fruitfly', 'grasshopper', 
    'leafbeetle', 'leafhopper', 'mites', 'slug', 'snail', 
    'stinkbug', 'thrips', 'weevil', 'whiteflies'
]

# Function to detect pests in image using only YOLOv8
def detect_pests(image, yolo_model):
    # Convert PIL Image to numpy array for YOLOv8
    img_np = np.array(image)
    
    # YOLOv8 detection with confidence threshold
    results = yolo_model(img_np, conf=0.25)
    
    # Get bounding boxes and detections
    detections = []
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            cls = int(box.cls[0].item())
            cls_name = result.names[cls]
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(conf),
                'class': cls_name,
                'class_confidence': float(conf)
            })
    
    # Debug: Save the original image with YOLOv8 detections
    if len(detections) > 0:
        debug_dir = "debug_images"
        os.makedirs(debug_dir, exist_ok=True)
        debug_img = img_np.copy()
        for detection in detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            cv2.rectangle(debug_img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.imwrite(f"{debug_dir}/debug_yolo_detections.jpg", cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))
    
    if len(detections) == 0:
        # If no pests detected, try splitting the image into smaller segments
        segments = split_image(image)
        
        for i, segment in enumerate(segments):
            segment_np = np.array(segment)
            segment_results = yolo_model(segment_np, conf=0.25)
            
            for result in segment_results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    cls = int(box.cls[0].item())
                    cls_name = result.names[cls]
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class': cls_name,
                        'class_confidence': float(conf),
                        'segment_index': i
                    })
            
            # Debug: Save segments with detections
            if len(detections) > 0:
                debug_dir = "debug_images"
                os.makedirs(debug_dir, exist_ok=True)
                debug_seg = segment_np.copy()
                for detection in detections:
                    if detection.get('segment_index') == i:
                        x1, y1, x2, y2 = map(int, detection['bbox'])
                        cv2.rectangle(debug_seg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.imwrite(f"{debug_dir}/debug_segment_{i}_detections.jpg", cv2.cvtColor(debug_seg, cv2.COLOR_RGB2BGR))
    
    return detections, results

# Function to split image into smaller segments with overlap
def split_image(image, segment_size=640, overlap=100):
    width, height = image.size
    segments = []
    
    for y in range(0, height, segment_size - overlap):
        for x in range(0, width, segment_size - overlap):
            # Define segment boundaries
            right = min(x + segment_size, width)
            bottom = min(y + segment_size, height)
            
            # Crop segment
            segment = image.crop((x, y, right, bottom))
            segments.append(segment)
    
    return segments

# Function to draw bounding boxes on image
def draw_boxes(image, detections):
    img_np = np.array(image)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection['bbox'])
        label = f"{detection['class']} ({detection['confidence']:.2f})"
        
        # Draw bounding box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw label
        cv2.putText(img_np, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)

# Function to process video
def process_video(video_bytes, yolo_model):
    # Create a temporary file to save the uploaded video
    temp_file = "temp_video.mp4"
    with open(temp_file, "wb") as f:
        f.write(video_bytes)
    
    # Open the video file
    cap = cv2.VideoCapture(temp_file)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer for output
    output_file = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Initialize variables for tracking
    frame_count = 0
    detected_pests = {}  # Dictionary to track detected pests
    pest_timestamps = {}  # Dictionary to store timestamps of first detection
    
    # Process the video
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Update progress
        progress = int((frame_count / total_frames) * 100)
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count}/{total_frames} ({progress}%)")
        
        # Convert frame to PIL Image for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)
        
        # Detect pests in the frame
        detections, _ = detect_pests(pil_image, yolo_model)
        
        # Track detected pests and record timestamps
        timestamp = frame_count / fps
        formatted_timestamp = str(timedelta(seconds=int(timestamp)))
        
        for detection in detections:
            pest_class = detection['class']
            
            # If this pest class hasn't been detected before, record its timestamp
            if pest_class not in pest_timestamps:
                pest_timestamps[pest_class] = formatted_timestamp
                detected_pests[pest_class] = detection
        
        # Draw bounding boxes on frame
        annotated_frame = draw_boxes(pil_image, detections)
        
        # Add timestamp
        cv2.putText(annotated_frame, formatted_timestamp, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Write the frame to output video
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    os.remove(temp_file)
    
    # Return the output video and detected pests
    with open(output_file, "rb") as f:
        processed_video = f.read()

# os.remove(output_file)  # This line is removed or commented out

# You may want to inform the caller where the file is located
    return processed_video, pest_timestamps, detected_pests

# Load model using Streamlit cached resource
yolo_model = load_yolo_model()
models_loaded = yolo_model is not None

# MAIN MENU PAGE
if page == "Main Menu":
    # Title
    st.title("🌱 Welcome to the Pest Detection and Tracking App")

    # Subtitle
    st.subheader("Empowering Precision Agriculture for a Sustainable Future")

    # Main Content
    st.write("""
    Precision Agriculture represents a cutting-edge approach to farming that leverages technology and data analytics to optimize crop production, minimize waste, and enhance sustainability. 

    Why Pest Detection Matters:
    -   Pests cause significant crop damage and economic losses.
    -   Traditional methods like manual inspections and broad-spectrum treatments can be time-consuming and environmentally harmful.
    -   Early and accurate detection is critical for targeted interventions, reducing chemical use and promoting sustainable farming practices.

    What This App Offers:
    - 🚀 Advanced YOLO-based Detection for pests.
    - 📸 Upload images or videos to analyze, track and classify pests.
    - 💾 Save results for future use in your research or farm management.
    
    By using this system, you can ensure healthier crops, better yields, and a step forward in sustainable farming practices. Join the revolution in smart agriculture!
    """)

    # Add an image or icon for visual appeal
    try:
        st.image(
            "/workspaces/pests-ui/images/main.jpg",  
            caption="Smart Farming: The Future of Agriculture",
            use_container_width=True
        )
    except:
        st.warning("Main image not found. Please check the path to the image file.")

    # Add a call-to-action button
    st.write("""Click on Pest Detection in Navigation Pane to Get Started 🔍""")

# PEST DETECTION PAGE
elif page == "Pest Detection":
    st.markdown("<div class='header'>Agricultural Pest Detection and Tracking</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>About This Tool</div>", unsafe_allow_html=True)
    st.write("""
    This application helps farmers and gardeners identify and track agricultural pests in images and videos.
    Upload an image or video of your plants, and our AI will detect and classify any pests present.
    """)

    # Show detected pests list
    st.markdown("### Detectable Pests")
    col1, col2, col3 = st.columns(3)
    pest_columns = [col1, col2, col3]
    for i, pest in enumerate(pest_classes):
        pest_columns[i % 3].write(f"- {pest.capitalize()}")
    st.markdown("</div>", unsafe_allow_html=True)
    
    if not models_loaded:
        st.error("YOLOv8 model could not be loaded. Please check the model path and try again.")
    else:
        # Create tabs for Image and Video processing
        tab1, tab2 = st.tabs(["Image Analysis", "Video Analysis"])

        with tab1:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='subheader'>Upload an Image</div>", unsafe_allow_html=True)
            
            # Image upload instructions
            with st.expander("📷 Photo Capture Instructions", expanded=False):
                st.markdown("""
                ### How to Take the Best Photo for Pest Detection:
                - **Image Size**: Aim for 640x640 pixels for optimal detection
                - **Background**: Use a white or light-colored background when possible
                - **Lighting**: Ensure good, even lighting without harsh shadows
                - **Focus**: Make sure the pest is clearly visible and in focus
                - **Distance**: Capture from about 10-15 cm away from the pest
                """)
            
            # Upload options: camera or file upload
            upload_option = st.radio("Choose input method:", ["Take a Photo", "Upload Image File"], horizontal=True)
            
            uploaded_image = None
            
            if upload_option == "Take a Photo":
                uploaded_image = st.camera_input("Take a photo", key="camera_input")
            else:
                uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image_uploader")
            
            if uploaded_image is not None:
                # Display original image
                image = Image.open(uploaded_image).convert('RGB')
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Process image on button click
                if st.button("Detect Pests", key="detect_image_btn"):
                    with st.spinner("Analyzing image..."):
                        # Detect pests using only YOLOv8
                        detections, yolo_results = detect_pests(image, yolo_model)
                        
                        if not detections:
                            st.warning("No pests detected. Please try another image or adjust lighting and focus.")
                        else:
                            # Draw bounding boxes on image
                            result_image = draw_boxes(image, detections)
                            
                            # Display result
                            st.image(result_image, caption="Detected Pests", use_column_width=True)
                            
                            # Display detection details
                            st.markdown("### Detection Results")
                            for i, detection in enumerate(detections):
                                pest_class = detection['class']
                                confidence = detection['confidence']
                                
                                st.markdown(f"#### {i+1}. {pest_class.capitalize()}")
                                st.markdown(f"**Confidence Score**: {confidence:.2f}")
                                
                                # Display pest information
                                if 'pest_info' in locals() and pest_class in pest_info:
                                    info = pest_info[pest_class]
                                    st.markdown(f"**Description**: {info['description']}")
                                    st.markdown(f"**Danger Level**: {info['danger_level']}")
                                    st.markdown(f"**Control Methods**: {info['control_methods']}")
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='subheader'>Upload a Video</div>", unsafe_allow_html=True)
            
            uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"], key="video_uploader")
            
            if uploaded_video is not None:
                video_bytes = uploaded_video.read()
                st.video(video_bytes)
                
                if st.button("Track Pests in Video", key="track_video_btn"):
                    with st.spinner("Processing video... This may take a while depending on the video length."):
                        # Process video using only YOLOv8
                        processed_video, pest_timestamps, detected_pests = process_video(video_bytes, yolo_model)
                        st.download_button(
                            label="Download Processed Video",
                            data=processed_video,
                            file_name="pest_detection_results.mp4",
                            mime="video/mp4"
                        )

                        if not pest_timestamps:
                            st.warning("No pests detected in the video.")
                        else:
                            # Display processed video
                            st.video(processed_video)
                            
                            # Display detected pests with timestamps
                            st.markdown("### Pest Detection Timeline")
                            
                            for pest_class, timestamp in pest_timestamps.items():
                                st.markdown(f"#### {pest_class.capitalize()}")
                                st.markdown(f"**First Detected at**: {timestamp}")
                                
                                # Display pest information
                                if 'pest_info' in locals() and pest_class in pest_info:
                                    info = pest_info[pest_class]
                                    st.markdown(f"**Description**: {info['description']}")
                                    st.markdown(f"**Danger Level**: {info['danger_level']}")
                                    st.markdown(f"**Control Methods**: {info['control_methods']}")
            st.markdown("</div>", unsafe_allow_html=True)

        # Footer
        st.markdown("""
        <div style="text-align: center; margin-top: 30px; padding: 10px; color: #666;">
            <p>Agricultural Pest Detection System • Powered by YOLOv8</p>
        </div>
        """, unsafe_allow_html=True)
