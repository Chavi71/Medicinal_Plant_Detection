# Medicinal Plant Identifier 

A smart plant recognition system that uses deep learning to detect and classify plants from images, providing users with insightful recommendations and information in real-time.

## Overview

This project integrates:
- **Convolutional Neural Networks (CNNs)** for accurate plant image classification.
- **Mobile Camera Interface** to capture leaves/flowers.
- **Recommendation System** that suggests care tips and medicinal or agricultural uses.
- **User-friendly Interface** for easy identification and interaction.

## Key Features
- Identify plant species using leaf or flower images.
- Provide botanical details and usage info (medicinal, ecological, etc.).
- Recommender engine for best growing conditions or applications.
- Responsive interface optimized for mobile devices.

## Tech Stack
- Python
- TensorFlow / PyTorch
- OpenCV
- Flask / Streamlit (for web/mobile UI)
- Custom-trained CNN model
- SQLite / Firebase (for plant data storage)

## Sample Workflow
1. User captures a leaf/flower.
2. Model predicts the plant species.
3. System displays:
   - Plant name & description
   - Ideal growing conditions
   - Medicinal or cultural significance
4. (Optional) Save to personal plant collection

## Model Performance
- Accuracy: ~93% on test dataset of common Indian flora.
- Real-time prediction latency: <1s on mid-range smartphones.

## Applications
- Educational field guides
- Smart gardening assistants
- Ayurvedic and herbal medicine reference
- Environmental conservation tools
