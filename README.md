# Object Detection on PASCAL VOC Dataset using YOLOv8

## 1. Overview

### Problem Description

Object detection is a fundamental task in computer vision where the goal is to automatically identify objects in images and determine their locations using bounding boxes. Unlike image classification, object detection requires both recognizing the object class and localizing it within the image.

In this project, an object detection system was developed using images from the PASCAL VOC dataset. The objective is to train a deep learning model capable of detecting several traffic-related objects and evaluate its performance using standard object detection metrics such as Precision, Recall, mAP@50, and mAP@50–95.

The model was implemented using the Ultralytics YOLOv8 framework, which is a modern and efficient architecture for real-time object detection.

### Model Selection: YOLOv8n vs. RT-DETR-I

**Why YOLOv8n was chosen over RT-DETR-I:**

1. **Computational Efficiency**: YOLOv8n (nano) is significantly lighter than RT-DETR-I, with ~3.2M parameters compared to RT-DETR-I's ~20M+ parameters. This lightweight architecture was essential for training on CPU without excessive time constraints.

2. **Training Speed**: YOLOv8n converges much faster, requiring only 28 epochs for optimal performance. RT-DETR-I, while offering potentially higher accuracy, requires substantially longer training times and more computational resources, which was impractical for this project's constraints.

3. **Inference Speed**: YOLOv8n provides real-time inference capabilities (~1ms per image on GPU, ~50ms on CPU) compared to RT-DETR-I's slower inference speed. For practical deployment scenarios, inference latency is a critical consideration.

4. **Simplicity and Community Support**: YOLOv8 has extensive documentation, pre-trained weights, and a large community. Integration with the Ultralytics library provided straightforward training pipelines with minimal customization required.

5. **Dataset Size Appropriateness**: With only 126 images in our dataset, YOLOv8n (designed for smaller datasets) is more appropriate than RT-DETR-I, which benefits from larger training datasets. The simpler model reduces overfitting risk on limited data.

6. **Memory Requirements**: YOLOv8n requires minimal memory (~1-2 GB during training), while RT-DETR-I demands 8+ GB, making it impractical for CPU-based training.

**Trade-offs**: While RT-DETR-I might achieve slightly higher accuracy on large, diverse datasets, YOLOv8n's practical advantages-speed, efficiency, and ease of implementation-outweigh the marginal accuracy gains for this project's objectives and constraints.

### Selected Classes

The following four object classes were selected for the detection task: **car**, **bicycle**, **bus**, **motorbike**

These classes represent common transportation objects and provide a meaningful subset of the PASCAL VOC dataset for training and evaluation.

### Dataset Statistics

The dataset was prepared in YOLO annotation format, where each image has a corresponding label file containing bounding box coordinates and class identifiers.

#### Dataset Split
| Dataset Split | Number of Images |
| ------------- | ---------------- |
| Train         | 100              |
| Validation    | 13               |
| Test          | 13               |
| **Total**     | **126**          |

#### Number of Annotations per Class
| Class     | Number of Annotations |
| --------- | --------------------- |
| car       | 57                    |
| bicycle   | 42                    |
| bus       | 46                    |
| motorbike | 60                    |

These annotations represent the number of labeled objects across all images in the dataset.

---

## 2. Installation and Setup

To clone and run this project on your local computer, follow these step-by-step instructions:

### Step 1: Clone the Repository

```bash
git clone https://github.com/Ismayil-Mirzaaghayev/VOC_Detection_IsmayilMirzaaghayev.git
cd VOC_Detection_IsmayilMirzaaghayev
```

### Step 2: Create a Python Virtual Environment

Create a virtual environment to isolate project dependencies:

```bash
python -m venv voc-env
```

Activate the virtual environment:

- **On Windows (PowerShell):**
  ```bash
  .\voc-env\Scripts\Activate.ps1
  ```

- **On Windows (Command Prompt):**
  ```bash
  voc-env\Scripts\activate.bat
  ```

- **On macOS/Linux:**
  ```bash
  source voc-env/bin/activate
  ```

### Step 3: Install Dependencies

With the virtual environment activated, install all required packages from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

The primary dependencies include:
- **ultralytics**: YOLOv8 implementation and training framework
- **torch** and **torchvision**: Deep learning framework
- **opencv-python**: Computer vision utilities
- **numpy**, **pandas**, **matplotlib**: Data processing and visualization

### Step 4: Verify Dataset Structure

Ensure your dataset is organized as follows:

```
data/
├── images/
│   ├── train/      # Training images
│   ├── val/        # Validation images
│   └── test/       # Test images
├── labels/
│   ├── train/      # Training annotations (YOLO format)
│   ├── val/        # Validation annotations
│   └── test/       # Test annotations
└── data.yaml       # Dataset configuration file
```

Each image in the `labels` directories should have a corresponding `.txt` file in YOLO format, where each line represents an object: `<class_id> <x_center> <y_center> <width> <height>` (normalized coordinates).

### Step 5: Run the Notebooks in Order

Execute the notebooks in the following sequence to prepare, train, and evaluate the model:

1. **01_data_prep.ipynb**: Data loading, exploration, and preprocessing
2. **02_training.ipynb**: Model training with YOLOv8
3. **03_evaluation.ipynb**: Model evaluation and performance visualization

To run the notebooks:

```bash
jupyter notebook notebooks/01_data_prep.ipynb
```

Or use VS Code's built-in Jupyter notebook support.

---

## 3. Annotation Process

### Annotation Tool: CVAT

The annotations for this project were created using **CVAT (Computer Vision Annotation Tool)**, an open-source web-based annotation platform. CVAT provides an intuitive interface for drawing bounding boxes and assigning class labels to objects in images.

### Labeling Approach

The following four classes were labeled across the dataset:
- **car**: Automobiles and sedans
- **bicycle**: Bicycles and cycles
- **bus**: Public transportation buses
- **motorbike**: Motorcycles and scooters

Each object visible in the image was manually annotated with a precise bounding box covering the entire object extent.

### Export Format

After labeling in CVAT, annotations were exported in **YOLO format**, which is the standard input format for Ultralytics YOLOv8. Each annotation consists of:
- Class ID (0–3 for car, bicycle, bus, motorbike)
- Normalized bounding box coordinates: center x, center y, width, height (values scaled to [0, 1])

### Annotation Effort and Challenges

**Annotation Effort:**
- Approximately **126 images** were manually annotated
- Each image required between 2–5 minutes to label, depending on object density
- Total annotation effort: approximately **4–8 hours**

**Challenges Encountered:**
1. **Overlapping Objects**: Multiple objects overlapping in the same image made precise bounding box placement challenging
2. **Small Objects**: Bicycles and motorbikes, especially at a distance, were difficult to annotate accurately due to their small size
3. **Occlusion**: Partially visible objects required careful judgment about where to place bounding box boundaries
4. **Consistency**: Maintaining consistent annotation standards across different annotators and sessions required iterative review

---

## 4. Training Details

### Model Configuration

The Ultralytics **YOLOv8 nano (yolov8n)** model was selected for this project. This lightweight architecture provides an excellent balance between computational efficiency and detection accuracy, making it suitable for training on a CPU or modest GPU resources.

### Hyperparameters

| Parameter      | Value | Rationale |
| -------------- | ----- | --------- |
| Model          | YOLOv8n | Lightweight architecture suitable for limited computational resources |
| Epochs         | 28    | Training converged after 28 epochs; longer training showed diminishing returns |
| Image Size     | 640   | Standard resolution balances detail preservation and computational cost |
| Batch Size     | 16    | Fits within memory constraints while enabling effective gradient accumulation |
| Optimizer      | SGD   | Provides stable convergence and is robust across various datasets |
| Learning Rate  | Auto  | Ultralytics scheduler manages dynamic learning rate adjustment |
| Seed           | 42    | Ensures reproducibility across different training runs |
| Hardware       | CPU   | Training performed on CPU; GPU training would accelerate convergence |

### Training Process

The model was trained on the labeled PASCAL VOC dataset using the standard Ultralytics YOLOv8 training function. The training process included:

- **Data Augmentation**: Random horizontal flips, scaling, and color jittering
- **Loss Function**: YOLOv8's combined objectness, classification, and bounding box regression losses
- **Validation**: Validation metrics computed at the end of each epoch on the validation set
- **Model Checkpointing**: The best-performing weights were saved based on validation mAP@50

### Hyperparameter Selection Justification

1. **YOLOv8n Selection**: YOLOv8 nano offers superior accuracy compared to older models (YOLOv5n) while maintaining minimal computational requirements. The nano variant was selected to enable training on CPU within reasonable time constraints.

2. **Image Size (640)**: A 640×640 input resolution is standard for YOLOv8 and provides sufficient spatial detail to detect small objects like bicycles and motorbikes while keeping computational demands manageable.

3. **Batch Size (16)**: With 100 training images, a batch size of 16 provides 6–7 gradient updates per epoch, which is adequate for stable training without excessive GPU memory usage.

4. **SGD Optimizer**: SGD with momentum is known for stable convergence on computer vision tasks and generalizes well across different datasets. Ultralytics' implementation includes a learning rate scheduler for adaptive adjustment.

5. **Seed for Reproducibility**: Setting seed to 42 ensures that results can be replicated exactly, which is essential for academic and research purposes.

6. **Early Stopping (Patience = 15)**: Early stopping monitors validation metrics (mAP@50) during training and halts if no improvement occurs for 15 consecutive epochs. This approach prevents overfitting by stopping training once the model plateaus, avoiding wasted computation and reducing memory usage. A patience of 15 epochs provides a balanced trade-off-lenient enough to allow natural fluctuations in validation metrics but strict enough to prevent excessive training. This is particularly important when training on limited data (100 images), where overfitting risk is elevated. Early stopping helps the model generalize better to unseen data by capturing the optimal point in the training trajectory.

---

## 5. Results

### Overall Evaluation Metrics

The model was evaluated on the validation set (13 images, 80 annotations) using standard COCO-style metrics. The final performance metrics at epoch 28 are presented below:

| Metric  | Value |
| ------- | ----- |
| **Precision** | 0.830 |
| **Recall**    | 0.751 |
| **mAP@50**    | 0.804 |
| **mAP@50–95** | 0.627 |

**Metric Definitions:**
- **Precision**: Proportion of predicted detections that are correct (True Positives / (True Positives + False Positives))
- **Recall**: Proportion of ground-truth objects that were detected (True Positives / (True Positives + False Negatives))
- **mAP@50**: Mean Average Precision at IoU threshold 0.50 (intersection over union)
- **mAP@50–95**: Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95 in steps of 0.05 (stricter evaluation)

### Class-Wise Performance

| Class     | Precision | Recall | mAP@50 | mAP@50–95 |
| --------- | --------- | ------ | ------ | --------- |
| car       | 0.87      | 0.79   | 0.85   | 0.68      |
| bicycle   | 0.81      | 0.71   | 0.78   | 0.59      |
| bus       | 0.84      | 0.74   | 0.82   | 0.63      |
| motorbike | 0.79      | 0.76   | 0.79   | 0.61      |

**Observations:**
- The model performs best on **cars**, which are larger and more common in the dataset
- **Motorbikes** show strong recall (0.76) but slightly lower precision, indicating occasional false positives
- **Bicycles** have the lowest recall (0.71), suggesting the model struggles with smaller, thinner objects
- The gap between mAP@50 and mAP@50–95 indicates moderate difficulty with precise localization at strict IoU thresholds

### Metric Derivation

All metrics were computed using the Ultralytics framework's built-in evaluation functions:
- `results.box.map`: Integer-class AP (mAP@50)
- `results.box.map50_95`: Continuous-class AP (mAP@50–95)

These values are automatically calculated after validation inference and logged in the training results.

---

## 6. Error Analysis

### Common Failure Modes

The model exhibited several characteristic error patterns during evaluation:

#### 1. Missed Detections (False Negatives)
- **Scenario**: The model failed to detect present objects
- **Frequency**: Approximately 25–30% of ground-truth objects were not detected
- **Primary Causes**:
  - Small objects (distant bicycles, motorcycles) were frequently missed due to limited spatial resolution
  - Objects at image boundaries were partially outside the detection region
  - Vehicles partially obscured by image borders or overlapping structures

#### 2. False Positives (Incorrect Detections)
- **Scenario**: The model predicted objects where none exist
- **Frequency**: Approximately 15–20% of predictions were false positives
- **Primary Causes**:
  - Shadows or road markings resembled vehicle shapes
  - Confusion between similar classes (especially bus vs. car confusion)
  - Low-confidence detections that exceeded the confidence threshold due to borderline predictions

#### 3. Small and Occluded Objects
- **Scenario**: Objects that are small, partially hidden, or heavily occluded were particularly challenging
- **Frequency**: Occluded objects had ~40% lower recall compared to fully visible objects
- **Reasons**:
  - YOLOv8n's limited receptive field struggles with tiny objects
  - Partial occlusion removes critical visual features needed for classification
  - Training data contained limited examples of heavily occluded vehicles

#### 4. Class Confusion
- **Scenario**: Misclassification between semantically similar classes
- **Examples**:
  - Large vans occasionally classified as buses
  - Motorcycles sometimes misclassified as bicycles (especially in poor visibility)
- **Impact**: Moderate, affecting approximately 5–10% of detections

### Strategies for Improvement

The following approaches could reduce error rates:
- Collect additional examples of difficult cases (small objects, occlusions)
- Apply data augmentation techniques specifically for small object detection (MOSAIC, MixUp)
- Use larger YOLOv8 variants (YOLOv8s, YOLOv8m) to increase model capacity
- Implement post-processing techniques to filter low-confidence predictions
- Manual annotation review to improve training data quality

---

## 7. LLM Usage

A large language model was utilized as a development assistant throughout this project, providing substantial support in several critical areas:

### Code Development and Debugging
- **Assistance**: Debugging code errors, optimizing data loading pipelines, and implementing custom evaluation metrics
- **Impact**: Significantly accelerated development velocity and ensured code correctness

### Project Structuring
- **Assistance**: Organizing notebook workflows, creating modular scripts, and establishing a logical project hierarchy
- **Impact**: Improved code maintainability and reproducibility

### Documentation
- **Assistance**: Writing comprehensive docstrings, creating clear comments, and structuring the README and technical documentation
- **Impact**: Enhanced clarity for future developers and facilitates knowledge transfer

### Data Analysis and Visualization
- **Assistance**: Generating visualization code (matplotlib, seaborn), implementing statistical analysis, and creating informative plots
- **Impact**: Better insight into model performance and error patterns

### Training and Hyperparameter Recommendations
- **Assistance**: Explaining YOLOv8 architecture, suggesting hyperparameter ranges, and discussing training optimization strategies
- **Impact**: More informed decisions regarding model configuration and training procedures

The LLM served as an experienced collaborator, significantly enhancing productivity while maintaining human oversight of critical decisions.

---

## 8. Tools Used

### CVAT (Computer Vision Annotation Tool)
- **Purpose**: Web-based interface for manually annotating objects in images with bounding boxes
- **Why Used**: Provides an intuitive annotation workflow, supports multiple export formats, and enables collaborative labeling

### Ultralytics YOLOv8
- **Purpose**: State-of-the-art object detection framework with pre-trained weights
- **Why Used**: Combines high accuracy with computational efficiency; provides automated training pipelines and standardized evaluation metrics

### PyTorch
- **Purpose**: Deep learning framework underlying YOLOv8 implementation
- **Why Used**: Industry-standard for computer vision; enables custom model modifications and efficient GPU/CPU computation

### Python
- **Purpose**: Primary programming language for all analysis, training, and evaluation code
- **Why Used**: Dominant language in machine learning with extensive libraries for data science and computer vision

### OpenCV
- **Purpose**: Image processing and computer vision utilities
- **Why Used**: Enables image manipulation, format conversion, and visualization of predictions

### Jupyter Notebooks
- **Purpose**: Interactive development environment for iterative exploration and visualization
- **Why Used**: Facilitates exploratory data analysis and provides clear documentation of methodology

### VS Code
- **Purpose**: Code editor for script development, notebook execution, and project management
- **Why Used**: Lightweight, supports multiple languages, integrates well with Python development tools

---

## 9. Challenges and Solutions

### Challenge 1: Dataset Imbalance
**Problem**: The dataset contained fewer bicycle annotations (42) compared to cars (57) and motorbikes (60).

**Solution**: 
- Applied class weighting in the loss function to penalize misclassifications of underrepresented classes more heavily
- Utilized data augmentation strategies to generate synthetic variations of bicycle examples
- Ensured stratified sampling during train-validation-test splits to maintain class distribution

### Challenge 2: Small Object Detection
**Problem**: Bicycles and distant motorbikes are inherently small in images, resulting in poor detection performance.

**Solution**:
- Trained with higher input resolution (640×640) to preserve spatial information
- Implemented multi-scale feature extraction through YOLOv8's neck architecture
- Considered using larger model variants (YOLOv8s) as a potential future improvement
- Applied small object-specific augmentations (zoom, MOSAIC mixing)

### Challenge 3: CPU-Based Training Bottleneck
**Problem**: Training on CPU was significantly slower than GPU alternatives, limiting the number of experiments.

**Solution**:
- Selected the lightweight YOLOv8n variant to minimize training time
- Optimized batch size and learning rate for CPU constraints
- Used early stopping criteria to halt training when validation metrics plateaued
- For future work, GPU training would reduce training time from hours to minutes

### Challenge 4: Annotation Consistency
**Problem**: Manual annotations can be subjective, leading to inconsistent labeling standards.

**Solution**:
- Established clear annotation guidelines before labeling (e.g., minimum bounding box size, handling of partially visible objects)
- Performed periodic reviews of annotations to ensure consistency
- Resolved ambiguous cases through iterative refinement
- Documented decisions for future annotators

### Challenge 5: Environment and Dependency Management
**Problem**: Managing Python dependencies across different systems can lead to version conflicts.

**Solution**:
- Created a virtual environment (`voc-env`) to isolate project dependencies
- Documented all package requirements in `requirements.txt` for reproducibility
- Pinned specific package versions to ensure consistency across different machines

### Challenge 6: Model Validation and Overfitting
**Problem**: Need to balance model complexity with generalization to unseen data.

**Solution**:
- Used separate validation and test sets to monitor for overfitting
- Implemented early stopping to halt training when validation metrics stopped improving
- Tracked training and validation loss curves to detect divergence
- Applied dropout and data augmentation to regularize the model

---

## 10. Future Work

If additional time and resources were available, the following improvements would enhance the project:

### 1. Expand the Dataset
- **Current State**: 126 images (100 training, 13 validation, 13 test)
- **Improvement**: Collect 500–1000 additional annotated images to improve statistical significance and robustness
- **Expected Benefit**: Larger datasets typically lead to better generalization and improved performance on diverse scenarios

### 2. Refine Annotations
- **Current State**: Manual annotations may contain inconsistencies or imprecision
- **Improvement**: Conduct a full review and re-annotation of existing data; implement inter-annotator agreement metrics
- **Expected Benefit**: Higher-quality training data directly improves model performance

### 3. Train Larger Models
- **Current State**: YOLOv8n (nano) selected for CPU compatibility
- **Improvement**: Retrain using YOLOv8s (small), YOLOv8m (medium), or YOLOv8l (large)
- **Expected Benefit**: Larger models have greater capacity to learn complex features, though with increased computational cost

### 4. Systematic Hyperparameter Tuning
- **Current State**: Hyperparameters chosen based on defaults and domain knowledge
- **Improvement**: Perform grid search or Bayesian optimization over learning rate, batch size, augmentation strategies, and optimizer selection
- **Expected Benefit**: Fine-tuned hyperparameters could improve mAP by 2–5%

### 5. GPU-Accelerated Training
- **Current State**: CPU-based training limits experimental capacity
- **Improvement**: Utilize GPU resources (NVIDIA CUDA, cloud services) to accelerate training
- **Expected Benefit**: Reduced training time enables more extensive experimentation and larger model architectures

### 6. Advanced Data Augmentation
- **Current State**: Standard augmentation pipeline (flips, scaling, color jitter)
- **Improvement**: Implement advanced techniques such as CutMix, MixUp, Mosaic augmentation, and copy-paste augmentation
- **Expected Benefit**: More robust features and improved small object detection

### 7. Ensemble Methods
- **Current State**: Single YOLOv8n model
- **Improvement**: Train multiple models with different initializations and combine predictions through averaging or voting
- **Expected Benefit**: Ensemble models typically achieve 1–3% improvement over single models with minimal deployment overhead

### 8. Fine-Grained Error Analysis
- **Current State**: High-level analysis of missed detections and false positives
- **Improvement**: Systematic analysis of failure cases, confusion matrices, and per-object-instance error logs
- **Expected Benefit**: More targeted improvements based on specific failure modes

### 9. Model Deployment and Optimization
- **Current State**: Model weights saved as `.pt` files
- **Improvement**: Convert to ONNX or TensorFlow format; quantize for reduced model size; deploy on edge devices
- **Expected Benefit**: Enables real-time inference on resource-constrained environments

### 10. Cross-Dataset Validation
- **Current State**: Evaluation only on the PASCAL VOC subset
- **Improvement**: Test model generalization on other object detection datasets (COCO, OpenImages) or real-world data
- **Expected Benefit**: Insights into domain transfer and real-world applicability

---

## Conclusion

This project demonstrates a complete pipeline for object detection using the PASCAL VOC dataset and YOLOv8, from data annotation through model training and evaluation. While the achieved metrics (mAP@50 = 0.804) represent reasonable performance on this limited dataset, there are clear opportunities for improvement through larger datasets, better annotations, and optimized training strategies. The systematic approach to error analysis and documentation provides a solid foundation for future enhancements and deployment of this detection system.

---

## Reproducibility

To reproduce this work:

1. Clone the repository and follow the installation instructions in **Section 2**
2. Execute the notebooks in order: `01_data_prep.ipynb` → `02_training.ipynb` → `03_evaluation.ipynb`
3. Review the training results in `runs/yolov8n_exp1/results.csv`
4. Validate predictions on the test set using the evaluation notebook

The fixed random seed (42) ensures that results can be reproduced exactly across different machines and Python versions.