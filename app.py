import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from datetime import datetime
import os
import pandas as pd 


# --- Model Path Verification ---
MODEL_PATH = "/Users/shanumsharief/smart_shelf_multiclass/multiclass_final_v3/high_res_instance_train/weights/best.pt" 

model = None 
try:
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL ERROR: Trained model not found at expected path: {MODEL_PATH}. Loading default YOLOv8n.")
        model = YOLO('yolov8n.pt') 
    else:
        model = YOLO(MODEL_PATH)
        print(f"SUCCESS: Loaded trained multi-class model from: {MODEL_PATH}")

    CLASS_NAMES = model.names 
    if len(CLASS_NAMES) < 2:
        print("WARNING: Model reports few classes. Verify custom training was multi-class.")

except Exception as e:
    print(f"CRITICAL ERROR loading model: {e}. Using placeholders.")
    CLASS_NAMES = {i: f"Product {i}" for i in range(12)}
    model = None 

# --- Inventory & Alert Settings ---
threshold_values = [10, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5]
LOW_STOCK_THRESHOLDS = {
    name: threshold_values[i]
    for i, name in enumerate(CLASS_NAMES.values())
    if i < len(threshold_values) 
}
SNAPSHOTS_DIR = "snapshots_multiclass"
os.makedirs(SNAPSHOTS_DIR, exist_ok=True)


# --- Core Processing Function ---
def process_frame(frame, conf_threshold):
    """Run YOLO on each webcam frame, count multiple classes, and manage alerts."""
    if frame is None or model is None:
        return None, 0, 0, "Model Not Loaded or No Frame.", None, None, "Check setup.", [] 

    try:
        # YOLO inference
        # IoU=0.4 is set low to prevent aggressive NMS on tightly packed shelf items (CRITICAL FIX for counting)
        results = model.predict(frame, imgsz=640, conf=conf_threshold, iou=0.4, verbose=False) 
        annotated = results[0].plot()

        # 1. Counting and Metrics
        class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        product_counts = {}
        detected_class_names = set()
        
        for class_id in class_ids:
            class_name = CLASS_NAMES.get(class_id, "Unknown")
            product_counts[class_name] = product_counts.get(class_name, 0) + 1
            detected_class_names.add(class_name) 

        total_items = len(class_ids)
        low_stock_count = 0
        low_stock_items = []
        low_stock_triggered = False

        # 2. Check for Low Stock Alerts (Context-Aware Logic)
        for name, threshold in LOW_STOCK_THRESHOLDS.items():
            current_count = product_counts.get(name, 0)
            
            # CRITICAL LOGIC: Only check for low stock if the item is VISIBLE and the count is low
            if name in detected_class_names: 
                if current_count < threshold:
                    low_stock_items.append(f"• {name}: {current_count} (Goal: >{threshold})")
                    low_stock_triggered = True
                    low_stock_count += 1
        
        # 3. Generate Status/Alert Message
        alert_msg = ""
        alert_sound = None
        snapshot_path = None
        
        if low_stock_triggered:
            alert_summary = f"🚨 LOW STOCK ({low_stock_count} items): " + ", ".join([item.split(': ')[0].replace('• ', '') for item in low_stock_items])
            alert_details = "\n".join(low_stock_items)
            alert_sound = "alert.mp3" 
            
            # Save snapshot (No logging CSV used)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_path = f"{SNAPSHOTS_DIR}/low_stock_{timestamp}.jpg"
            cv2.imwrite(snapshot_path, cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
            
            status_text = alert_summary
        
        else:
            if detected_class_names:
                status_text = f"🟢 STOCK OK | Total Items: {total_items}"
            else:
                status_text = "⚪ Waiting for products to enter view..."

            alert_details = "None currently."

        # 4. Prepare Live Inventory Table Data
        live_table_data = []
        for name, threshold in LOW_STOCK_THRESHOLDS.items():
            count = product_counts.get(name, 0)
            status_icon = "⚪ N/A"
            
            if name in detected_class_names:
                if count < threshold:
                    status_icon = "🔴 LOW"
                else:
                    status_icon = "✅ OK"
            
            live_table_data.append([name, count, threshold, status_icon])
        
        return annotated, total_items, low_stock_count, status_text, alert_sound, snapshot_path, alert_details, live_table_data

    except Exception as e:
        return None, 0, 0, f"ERROR: {str(e)}", None, None, f"An unexpected error occurred: {e}", []


# --- GRADIO UI (Single Tab) ---
with gr.Blocks(title="Smart Shelf Inventory System") as demo:

    total_items_detected_state = gr.State(0)
    low_stock_count_state = gr.State(0)
    LIVE_TABLE_HEADERS = ["Product", "Current Count", "Threshold", "Status"]

    # --- CSS for Visual Style ---
    gr.Markdown("""
    <style>
    body { background-color: #f7f9fb; font-family: 'Inter', sans-serif; }
    .main-header { background-color: #fff; padding: 15px 30px; border-bottom: 1px solid #e0e0e0; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    .card { background-color: #fff; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid #f0f0f0; transition: border 0.3s; }
    .card-alert { border: 2px solid #ef4444 !important; background-color: #fef2f2; }
    .metric-value { font-size: 2.5rem; font-weight: 700; color: #1f2937; margin-bottom: 5px; }
    .metric-label { font-size: 1rem; color: #6b7280; text-transform: uppercase; }
    .title-text { color: #1f2937; font-weight: 800; font-size: 2rem; }
    .subtitle-text { color: #4b5563; }
    </style>
    """)

    # --- Header ---
    gr.HTML('<div class="main-header"><div class="title-text">Smart Shelf Inventory Management</div><div class="subtitle-text">Real-time Vision & Stock Analytics Dashboard</div></div>')

    with gr.Row(variant="compact", elem_id="main-layout"):
        
        with gr.Column(scale=5):
            
            # --- OVERVIEW Metrics Row ---
            with gr.Row(elem_id="overview-metrics"):
                
                with gr.Column(min_width=150, scale=1, elem_classes=["card"]):
                    gr.HTML('<div class="metric-label">Total Items Detected</div>')
                    total_items_output = gr.Number(value=total_items_detected_state, label="Total Items", elem_classes=["metric-value"], interactive=False)
                    gr.HTML('<div class="metric-label text-green-500">In Camera View</div>')

                with gr.Column(min_width=150, scale=1, elem_classes=["card"]):
                    gr.HTML('<div class="metric-label">Total Categories Tracked</div>')
                    gr.Number(value=len(LOW_STOCK_THRESHOLDS), label="Categories", elem_classes=["metric-value"], interactive=False)
                    gr.HTML('<div class="metric-label text-blue-500">Multi-Class System</div>')

                with gr.Column(min_width=150, scale=1, elem_classes=["card", "card-alert"]):
                    gr.HTML('<div class="metric-label">Items Below Threshold</div>')
                    low_stock_output = gr.Number(value=low_stock_count_state, label="Low Stock", elem_classes=["metric-value"], interactive=False)
                    gr.HTML('<div class="metric-label text-red-600">Action Required!</div>')


            # --- LIVE MONITORING ROW ---

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Live Webcam Feed")
                    webcam = gr.Image(
                        label="Webcam Input",
                        type="numpy",
                        sources=["webcam"],
                        streaming=True,
                        height=400,
                    )
                with gr.Column(scale=1):
                    gr.Markdown("### AI Detections")
                    detected_img = gr.Image(label="Annotated Detections", height=400)

            with gr.Row():
                # Confidence Slider (Control Panel)
                conf_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.4,  
                    step=0.01,
                    label="Detection Confidence Threshold (conf)",
                    interactive=True
                )

            # --- NEW ROW FOR LIVE INVENTORY TABLE ---
            with gr.Row():
                gr.Markdown("### Real-Time Product Status")
                live_inventory_table = gr.DataFrame(
                    headers=LIVE_TABLE_HEADERS,
                    interactive=False,
                    label="Live Inventory Counts"
                )


            with gr.Row(elem_id="alert-and-status"):
                with gr.Column(scale=2, elem_classes=["card"]):
                    gr.Markdown("### System Status")
                    status_box = gr.Textbox(label="Current Alert Status", interactive=False)
                    alert_details_box = gr.Textbox(label="Low Stock Details", lines=3, interactive=False, placeholder="Lists items below their required threshold.")

                with gr.Column(scale=1, elem_classes=["card"]):
                    gr.Markdown("### Alert Artifacts")
                    alert_audio = gr.Audio(label="Audio Alert", autoplay=True) 
                    snapshot_img = gr.Image(label="Low Stock Snapshot", height=100)

            
            # --- Stream Event Linkage (Continuous Updates) ---
            webcam.stream(
                fn=process_frame,
                inputs=[webcam, conf_slider], 
                outputs=[
                    detected_img, 
                    total_items_detected_state, 
                    low_stock_count_state, 
                    status_box, 
                    alert_audio, 
                    snapshot_img, 
                    alert_details_box, 
                    live_inventory_table
                ]
            )


# --- LAUNCH APP ---
if __name__ == "__main__":
    demo.launch()