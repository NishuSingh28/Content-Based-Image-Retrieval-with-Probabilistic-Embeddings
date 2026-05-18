# Content-Based Image Retrieval with Probabilistic Embeddings

A production-ready Streamlit application for intelligent image retrieval using deep learning and probabilistic embeddings.

## Features
- Deep feature extraction using ResNet50
- Probabilistic embeddings with uncertainty estimation
- FAISS-powered fast similarity search
- Interactive Streamlit dashboard
- Retrieval evaluation metrics
- Clean modular architecture for resume-ready presentation

## Project Structure
```
cbir_streamlit_app/
│── app.py
│── requirements.txt
│── README.md
│── pages/
│   ├── 1_Dataset_Explorer.py
│   ├── 2_Image_Retrieval.py
│   ├── 3_Probabilistic_Embeddings.py
│   └── 4_Model_Evaluation.py
│── src/
│   ├── feature_extractor.py
│   ├── retrieval_engine.py
│   ├── probabilistic_model.py
│   └── utils.py
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
