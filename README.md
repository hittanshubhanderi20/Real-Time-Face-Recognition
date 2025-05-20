# 🧠 Real-Time Face Recognition System

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Enabled-green.svg)
![Face Recognition](https://img.shields.io/badge/Face%20Recognition-Real%20Time-orange.svg)

---

## 🎯 Project Overview

This project implements a **real-time facial recognition system** using computer vision and machine learning. Built with OpenCV and `face_recognition`, it detects and recognizes known faces live from a webcam feed and can be extended to support automatic attendance systems, surveillance, and more.

---

## 🔍 Features

- 📸 Real-time webcam video feed with face bounding boxes
- 🧠 Face encoding and comparison using `face_recognition` (dlib)
- 📂 Auto-saves unknown faces into a directory for review
- ✅ Easily extensible for access control or identity verification use cases
- 💾 Face data stored as encodings for fast future recognition

---

## 🛠️ Tech Stack

| Component      | Library/Tool        |
|----------------|---------------------|
| Face Detection | `face_recognition`, `dlib` |
| Video Feed     | `OpenCV`            |
| Model Backend  | CNN / HOG (via `dlib`) |
| Data Handling  | Python (`os`, `numpy`) |
| Deployment     | Standalone `.py` script |

---

## 🗂 Project Structure

```

Real-Time-Face-Recognition/
├── known\_faces/                # 📁 Folder with labeled images for known people
├── unknown\_faces/              # 📁 Folder to store captured unknown faces
├── face\_recognition\_main.py    # 🎯 Main script to run real-time face recognition
├── encode\_faces.py             # 🧠 Preprocessing: encodes known faces
├── requirements.txt            # 📦 Python dependencies
├── README.md                   # 📘 Project documentation

````

---

## 📸 Sample Workflow

1. Load known faces from `known_faces/`
2. Run webcam and detect faces in real-time
3. Compare each face to known encodings
4. Annotate names or label as "Unknown"
5. Save unknown faces for future training

---

## ▶️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/hittanshubhanderi20/Real-Time-Face-Recognition.git
cd Real-Time-Face-Recognition
````

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Encode your known faces

Add labeled images (one face per image) to the `known_faces/` folder, then run:

```bash
python encode_faces.py
```

### 4. Start recognition system

```bash
python face_recognition_main.py
```

---

## ⚠️ Notes

* Ensure your webcam is connected and functional
* Lighting conditions significantly affect recognition accuracy
* Resize large images in `known_faces/` for performance
* Best results come with high-quality, front-facing images

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙋‍♂️ Contact

Built with ❤️ by Hittanshu Bhanderi
Connect via [LinkedIn](https://www.linkedin.com/in/hittanshubhanderi/)

---
