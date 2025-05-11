# MedAssist: AI-Powered Automated Nurse System

## 🏥 Overview

MedAssist is an AI-powered automated nurse system designed to streamline communication between patients and healthcare providers. The system addresses critical challenges in healthcare delivery including nursing shortages, emergency room inefficiencies, and communication gaps between patients and medical staff.

### The Problem

The healthcare industry faces:
- Critical nursing shortages leading to burnout and reduced quality of care
- Long emergency room wait times and inefficient patient processing
- Communication barriers between patients and healthcare providers
- Patients struggling to articulate symptoms effectively
- Patients often misunderstanding or forgetting treatment details

### Our Solution

MedAssist serves as an automated communication interface that:
- Enables patients to describe their conditions through natural conversation
- Extracts critical medical information and classifies patient severity
- Forwards structured patient data to doctors before arrival
- Records and summarizes doctor-patient interactions
- Generates personalized treatment summaries, recovery timelines, and medication reminders

## ✨ Features

- **Intelligent Patient Intake**
  - Natural language chatbot interface
  - Symptom recognition and extraction
  - Triage severity classification
  - Pre-arrival information forwarding

- **Medical Interaction Processing**
  - Speech-to-text conversion of doctor consultations
  - Real-time medical entity recognition
  - Automatic generation of consultation summaries

- **Patient Follow-up System**
  - Recovery timeline generation
  - Medication reminders
  - Simplified treatment explanations
  - Email or chat-based delivery of follow-up information

## 🛠️ Technical Architecture

### Chatbot Intake Module

**Baseline Implementation:**
- T5 pre-trained transformer for abstractive summarization
- MedSpaCy for Named Entity Recognition

**Advanced Implementation:**
- Custom transformer model for structured information extraction
- Patient condition classification system
- Enhanced medical entity recognition with MedSpaCy

### Doctor Summary Module

**Baseline Implementation:**
- Whisper model for speech-to-text conversion
- T5 pre-trained transformer for abstractive summarization
- MedSpaCy for Named Entity Recognition

**Advanced Implementation:**
- Whisper for optimized medical speech recognition
- Custom transformer for medical text summarization
- Specialized medical entity recognition

### System Integration

- Python backend for data processing and model integration
- React/Next.js frontend for user interface
- Secure data transmission protocols
- Email and notification systems

## 📊 Dataset

This project utilizes the [augmented-clinical-notes dataset](https://huggingface.co/datasets/AGBonnet/augmented-clinical-notes) from Hugging Face, which provides:
- Medical conversations between healthcare providers and patients
- Clinical notes from medical professionals
- Summarized medical information

## 📋 Evaluation Metrics

We evaluate our system using:
- **ROUGE scores** for measuring textual overlap
- **BERTScore** for assessing semantic similarity
- **F1 scores** for structured component accuracy
- **Exact match accuracy** for critical medical fields

## 👥 Team

- [Pranav Cheraku](https://github.com/pranav-cheraku)
- [Jeffrey Guo](https://github.com/Jeffrey-F-Guo)


*This project is intended for educational purposes only.*