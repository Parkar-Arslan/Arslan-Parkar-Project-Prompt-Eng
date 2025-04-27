# Chest X-ray Analysis with RAG in Google Colab

This guide explains how to run the Chest X-ray Analysis with RAG implementation in Google Colab using a T4 GPU.

## Setup Instructions

### 1. Prepare the Notebook

- Upload the `Xray_analysis.ipynb` file to Google Drive
- Right-click the file and select "Open with" > "Google Colaboratory"
- Alternatively, create a new Colab notebook and copy-paste the code

### 2. Enable T4 GPU

- Click on "Runtime" in the menu
- Select "Change runtime type"
- Set Hardware accelerator to "GPU"
- Ensure "GPU type" is set to "T4" (this is typically the default)
- Click "Save"

### 3. Kaggle API Setup

The notebook requires downloading the chest X-ray dataset from Kaggle. Set up your Kaggle credentials:

```python
# Run this in a new cell at the beginning of the notebook
from google.colab import files
uploaded = files.upload()  # Upload your kaggle.json file

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

To get your kaggle.json:
1. Go to kaggle.com and sign in
2. Go to Account > API > Create New API Token
3. Upload the downloaded file when prompted

## Running the Notebook

### Important: Run Cells One by One

For optimal execution and to avoid memory issues, run the notebook cells sequentially instead of all at once:

1. Start with Section 1: Environment Setup
   - This installs required libraries and creates project directories
   - Wait for all installations to complete before proceeding

2. Run Section 2: Download Data from Kaggle
   - This downloads the chest X-ray dataset
   - Ensure the download completes before continuing

3. Run Section 3: Enhanced Data Exploration and Analysis
   - This creates visualizations of the dataset
   - These visualizations help understand the data distribution

4. Run Section 4: Enhanced Multi-Condition Model Building and Training
   - This trains the deep learning model
   - Note: This section takes ~20-30 minutes on a T4 GPU
   - Watch for potential memory issues; reduce batch size if needed

5. Run Section 5: Enhanced Model Evaluation with Detailed Metrics
   - This evaluates the model performance
   - Review the metrics to understand model accuracy

6. Run Section 6: Enhanced Model Testing with Detailed Lung Analysis
   - This tests the model on sample images
   - You can upload your own X-ray images for testing

7. Run Section 7: Enhanced RAG with LangChain and Advanced Medical Knowledge
   - This builds the retrieval system for medical knowledge
   - Note: This section takes ~10-15 minutes to run

8. Run Section 8: X-ray Analysis Integrated with Advanced RAG-Based Diagnostic Assistant
   - This integrates the model with the RAG system
   - Creates a unified diagnostic pipeline

9. Run Section 9: Enhanced Web Interface with Chatbot-Style Q&A
   - This creates the Gradio interface
   - **IMPORTANT**: At the end of this section, you will receive a public URL link
   - The link will be displayed as: `Running on public URL: https://xxx.gradio.app`
   - Use this link to access the web interface

## Accessing the Gradio Interface

After running the final section, you will see a message similar to:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxx-xxxx-xxxx.gradio.app
```

1. Click on the public URL link to open the web interface
2. The interface has three tabs:
   - **Detailed Analysis**: Upload an X-ray and get a comprehensive report
   - **ChatBot Q&A**: Talk with the AI about the X-ray findings
   - **About**: Information about the system

3. The link will remain active for about 72 hours or until your Colab session ends

## Memory Management Tips

The T4 GPU has 16GB of memory. If you encounter memory issues:

- Run cells one by one as suggested
- Reduce batch size in Section 4 (`batch_size = 16` instead of `32`)
- Use a smaller language model in Section 7
- Restart the runtime if you get CUDA out of memory errors
- Save intermediate results to Google Drive

## Saving Your Results

To save models and results to Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Save model to Drive
!mkdir -p /content/drive/MyDrive/chest_xray_project
!cp -r ./chest_xray_project/models /content/drive/MyDrive/chest_xray_project/
```

## Troubleshooting

- **CUDA out of memory**: Restart runtime and reduce batch sizes
- **Kaggle dataset download fails**: Check your API token and internet connection
- **Gradio link not appearing**: Make sure to include `share=True` in the Gradio launch command
- **Language model loading fails**: Try a smaller model like Flan-T5-small

Remember that your Colab session will eventually time out. If this happens, you'll need to re-run the notebook.
