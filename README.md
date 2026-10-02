Smart Shelf Inventory

A YOLOv8-based computer vision project for detecting and counting selected retail products from shelf images/video and generating low-stock alerts.

The system was developed and evaluated using a custom-labelled dataset collected from a single retail store.

Demo

Suggested file:

demo/demo.jpeg

The demo shows live shelf detection, product counting, inventory status, and low-stock alerts through a Gradio interface.

What It Does

Multi-class product detection using YOLOv8

Product counting from live camera input

Per-class inventory threshold checking

Low-stock alerts

Snapshot saving for alert events

Gradio-based monitoring interface

Project Structure

smart_shelf_inventory/
│
├── app.py                         # Gradio UI + inference loop
├── train.py                       # YOLOv8 training script
├── data.yaml                      # Dataset configuration
├── requirements.txt               # Python dependencies
├── yolov8n.pt                     # Base YOLOv8 model
├── alert.mp3                      # Alert sound
│
├── data/                          # Dataset (not included in the public repo)
│
├── multiclass_final_v3/           # Training output / trained model
├── multiclass_new_run/            # Additional training run
├── snapshots_multiclass/          # Generated alert snapshots
│
└── README.md

Dataset

The dataset was collected and labelled manually from a single retail store.

It contains a limited set of product classes and shelf conditions. Because the images were collected from one store and the product/background diversity is limited, the trained model is primarily suited to products and visual conditions represented in this dataset.

The raw dataset is not included in this repository.

Results

On the validation set from this training run:

mAP@50: 26.15%

mAP@50–95: 14.27%

Precision: 53.50%

Recall: 24.03%

The highest recorded validation mAP values occurred around epochs 185–187.

These metrics reflect performance on the project's validation set and should not be interpreted as general performance across unseen stores or products.

How It Works

Camera / Image
      ↓
YOLOv8 Detection
      ↓
Product Classification + Counting
      ↓
Per-class Threshold Check
      ↓
Inventory Status
      ↓
Low-stock Alert + Snapshot

Setup

1. Create a virtual environment

python3 -m venv .venv
source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

Run the Application

Start the Gradio dashboard with:

python app.py

The application accepts live camera input and displays detected products, counts, inventory status, and low-stock alerts.

Training

To train a new YOLOv8 model:

python train.py

The training configuration used for the reported run was:

Base model: yolov8n.pt

Image size: 1024 × 1024

Epochs: 200

Batch size: 8

Training outputs are saved to the configured project/run directory.

Configuration

Low-stock thresholds are configured in app.py.

The confidence threshold can be adjusted through the Gradio interface during inference.

Training parameters such as image size, epochs, and batch size can be changed in train.py.

Limitations

This project has an intentionally narrow evaluation scope.

The training images were collected and labelled by me from one retail store, so the dataset does not capture enough variation across stores, products, shelf layouts, lighting, or camera conditions.

As a result:

The model works best on product classes represented in the training data.

It is not reliable for arbitrary products from unseen stores.

Performance can change with different shelf layouts or visual conditions.

Improving generalization would require a larger and more diverse dataset.

What I Learned

This project helped me work through the full computer-vision pipeline:

Data Collection
      ↓
Manual Labelling
      ↓
Dataset Preparation
      ↓
YOLOv8 Training
      ↓
Model Evaluation
      ↓
Real-time Inference
      ↓
Counting + Application Logic

A key limitation I identified during testing was the relationship between dataset diversity and real-world generalization. Testing on the same store showed good detection performance, while changing the environment exposed the limits of the training data.

Tech Stack

Python

YOLOv8 / Ultralytics

OpenCV

Gradio

Repository Notes

The public repository contains the application and training code needed to understand and reproduce the project workflow.

The original image dataset is not included because it was collected personally and is tied to a specific retail environment.