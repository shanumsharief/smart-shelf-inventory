import os
from ultralytics import YOLO

# --- Configuration for Training (Optimized for Instance Detection) ---

# 1. Model to use (YOLOv8n is lightweight and fast)
MODEL_TO_LOAD = 'yolov8n.pt'     

# 2. Dataset config file (Located in project root)
DATA_CONFIG_PATH = 'data.yaml'   

# 3. Training Hyperparameters - INCREASED RESOLUTION AND EPOCHS
# WARNING: Training at 1024x1024 requires a GPU and significant time.
EPOCHS = 200          # Increased training time for better feature learning
IMAGE_SIZE = 1024     # Increased resolution for detecting individual item boundaries (CRITICAL FIX)
BATCH_SIZE = 8        # Lowered to 8 to prevent memory errors at 1024 resolution

# 4. Output project/run names (saved under runs/detect/)
PROJECT_NAME = 'multiclass_final_v3'
RUN_NAME = 'high_res_instance_train' 

# --- Training Logic ---

def run_training():
    """Loads the YOLOv8 model and starts the high-resolution training process."""

    if not os.path.exists(DATA_CONFIG_PATH):
        print(f"❌ ERROR: '{DATA_CONFIG_PATH}' not found in project folder.")
        return

    # 1. Load model weights (Start clean to ensure no semantic generalization)
    print(f"🔄 Loading model: {MODEL_TO_LOAD}")
    model = YOLO(MODEL_TO_LOAD) 

    # 2. Train model
    print(f"\n🚀 Starting YOLOv8 high-res training ({IMAGE_SIZE}x{IMAGE_SIZE})...\n")
    try:
        model.train(
            data=DATA_CONFIG_PATH,
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            project=PROJECT_NAME,
            name=RUN_NAME
        )

        # 3. Report save path
        final_path = f"runs/detect/{PROJECT_NAME}/{RUN_NAME}/weights/best.pt"
        print("\n✅ Training completed!")
        print(f"Weights saved at: {final_path}")
        print("\nNEXT STEP: Update MODEL_PATH in app.py to this path.")

    except Exception as e:
        print(f"\n❌ ERROR DURING TRAINING: {e}")
        print("🔎 Check dataset paths in data.yaml and GPU/CPU memory availability.")

if __name__ == '__main__':
    run_training()