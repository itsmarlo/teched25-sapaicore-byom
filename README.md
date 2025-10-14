# 🧠 TechEd 2025 – Blocked Invoice Classification (BYOM on SAP AI Core)

This repository demonstrates how to **train, deploy, and serve a Hugging Face Transformer model** for the **Blocked Invoice Classification** use case on **SAP AI Core**.  
It was created as part of **SAP TechEd 2025** to showcase Bring-Your-Own-Model (BYOM) capabilities on **SAP Business Technology Platform (BTP)**, integrating **AI Core**, **AI Launchpad**, and **S/4HANA** scenarios.

---

## 📘 Overview

Blocked invoices occur in SAP S/4HANA when mismatches arise between purchase orders, goods receipts, and supplier invoices.  
The goal of this project is to automate classification of these blocked invoices using a fine-tuned **DistilBERT/DeBERTa** model to predict reasons for the block (e.g., quantity mismatch, price mismatch, missing PO).

This repository includes:

- **Data preparation** for invoice and PO text data.  
- **Training pipeline** for Hugging Face models integrated into **SAP AI Core**.  
- **Serving pipeline** for inference using AI Launchpad endpoints.  
- **Docker images** for training and serving with workflow templates in YAML.  

---

## 🏗️ Repository Structure

