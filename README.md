# Weed Detection using YOLOv8

This repository contains a Jupyter Notebook demonstrating the use of the **YOLOv8 object detection model** to identify and classify *weed* and *crop* instances from agricultural field images. The goal is to leverage deep learning for precision agriculture by automating weed detection, which can lead to more targeted herbicide use and improved crop yields.

## 🧠 Model & Framework

- **Model**: YOLOv8 (via SuperGradients)
- **Framework**: SuperGradients Training Library
- **Backend**: PyTorch
- **Data Format**: YOLO annotation style (bounding boxes + labels)

## 📁 Dataset

The notebook utilizes a weed detection dataset with the following structure:

```
Weed Detection/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
```

Each image has a corresponding YOLO-formatted label file. The dataset consists of two classes:
- `crop`
- `weed`

## 🔧 Project Workflow

1. **Environment Setup**
   - Uses Kaggle environment with common libraries pre-installed
   - Installs `super-gradients` library for training

2. **Dataset Preparation**
   - Loads the YOLO-formatted dataset and organizes it into `train`, `val`, and `test` sets
   - Sets dataset paths and class names

3. **Model Definition**
   - Loads the `PP-YOLOE` model from SuperGradients
   - Applies YOLO-specific loss and evaluation metrics (e.g., mAP@0.5, mAP@0.5:0.95)

4. **Training Configuration**
   - Defines training parameters like learning rate, batch size, image size, and number of epochs
   - Trains the model using SuperGradients' `Trainer` module

5. **Evaluation & Visualization**
   - Evaluates the best saved model on the test set
   - Visualizes ground truth and prediction overlaps

## 📊 Evaluation Metrics

- **mAP@0.5**: Measures precision across object confidence thresholds
- **mAP@0.5:0.95**: A more rigorous evaluation averaging over multiple IoU thresholds

## 🖼️ Visualizations

The notebook includes visualization examples for:
- Ground truth bounding boxes
- Predicted detections after training
- Model performance comparison

## 🚀 How to Run

You can run the notebook on [Kaggle Notebooks](https://www.kaggle.com/) or in a local Jupyter environment with GPU support. Ensure you have the following installed:

```bash
pip install super-gradients
```

Make sure to place the dataset in the expected directory structure before running the notebook.

## 📌 Dependencies

- Python ≥ 3.8
- numpy
- pandas
- matplotlib
- opencv-python
- super-gradients
- tqdm

## 📂 File Structure

```
.
├── weed-detection-using-yolo-v8.ipynb
├── README.md
└── data-set.zip/
    └── Weed Detection/
```

## 🧑‍💻 Author

*This project notebook was created by **Samarth Kulkarni**. If you find this helpful, feel free to star 🌟 the repo and share feedback!*

---

## 📄 License

This project is open-sourced under the MIT License.

## 🔗 Actual Implementation is on Kaggle:

- 📓 Kaggle Notebook: [Weed Detection using YOLOv8](https://www.kaggle.com/code/samrocks03/weed-detection-using-yolo-v8)
- 📂 Kaggle Dataset: [Weed Detection Dataset](https://www.kaggle.com/datasets/samrocks03/weed-detection/)
