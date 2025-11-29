
# Knowledge Base Agent

Developed by **Suchithra G**

This project is submitted for the **AI Agent Development Challenge**.

## Overview
The Knowledge Base Agent allows users to upload PDF, DOCX, or TXT documents and ask questions. The system retrieves the most relevant text chunks using embeddings and answers using Gemini with citations.

## Features
- PDF/DOCX/TXT upload
- Text extraction & chunking
- Embeddings (text-embedding-004)
- Vector search with cosine similarity
- Gemini 2.5 Flash for answer generation
- Citations included in outputs
- Clean Streamlit UI with chat history

## Setup
```
pip install -r requirements.txt
streamlit run app.py
```

## Tech Stack
- Streamlit
- Google Generative AI (Gemini)
- PyPDF2, python-docx
- numpy, scikit-learn
## 🎥 Project Demo Video
Watch the full demo video on Google Drive:

🔗 https://drive.google.com/file/d/1m9AmJNDRIw4dClkTMrkmcqVvk6Ekvax8/view?usp=drive_link

The working link:
https://suchithra481--knowledge-base-agent--app-mafplu.streamlit.app/


