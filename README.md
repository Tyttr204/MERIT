# MERIT



MERIT is a multimodal emotion-recognition system that integrates EEG, audio, and video information with modality-quality assessment and quality-aware multimodal fusion.



MERIT provides two execution modes:



- **MERIT-Offline Model** - designed for previously recorded multimodal data. This mode is recommended for reviewers because it does not require live sensing hardware.

- **MERIT-Realtime Model** - designed for realtime multimodal emotion recognition when compatible EEG acquisition hardware, a microphone, and a camera are available.



> **For peer review and reproducibility testing, we recommend using MERIT-Offline Model.**



---



# Quick Start



## 1. Download the Repository



Open PowerShell and run:



```powershell

git clone https://github.com/Tyttr204/MERIT.git

cd MERIT



## 2. System Requirements



The validated reviewer environment is:



- Windows 10/11

- Python 3.11

- FFmpeg and FFprobe available on `PATH`

- PyTorch 2.7.1

- CUDA 12.8 for NVIDIA GPU execution



A CPU-only mode is also supported, although video inference can be substantially slower.



Check the installed Python version with:



```powershell

python --version



## 3. Download the Required Runtime Assets



Large model files and demonstration EEG files are distributed separately through the GitHub Release `review-v1.0.0`.



Run:



```powershell

python .\\download\_reviewer\_assets.py



## 4. Create the Reviewer Environment



For a computer with an NVIDIA GPU and CUDA support, run:



```powershell

python .\\bootstrap\_reviewer.py --device cuda



---



# Recommended Reviewer Demo - Offline Mode



The Offline Mode is the recommended execution path for peer review because it does not require live EEG, camera, or microphone hardware.



Launch the graphical interface with:



```powershell

\& ".\\.venv-reviewer\\Scripts\\python.exe" `

&#x20; ".\\MERIT-Offline Model\\03\_reviewer\_demo.py" `

&#x20; --python ".\\.venv-reviewer\\Scripts\\python.exe" `

&#x20; --device auto `

&#x20; --fusion-device cpu



---



# Realtime Mode



`MERIT-Realtime Model` is intended for natural human-robot interaction scenarios with live multimodal acquisition.



The realtime system can use:



- live camera input;

- live microphone input;

- compatible EEG acquisition hardware.



Camera and microphone devices can first be inspected with:



```powershell

\& ".\\.venv-reviewer\\Scripts\\python.exe" `

&#x20; ".\\MERIT-Realtime Model\\01\_probe\_av\_devices.py" `

&#x20; --list-only



---



# Repository Structure



```text

MERIT/

|-- EEG/

|-- Audio/

|-- Video/

|-- Quality/

|-- Fusion/

|-- MERIT-Offline Model/

|   |-- testing Data/

|   |-- 03\_reviewer\_demo.py

|   `-- 08\_reviewer\_runtime\_probe.py

|-- MERIT-Realtime Model/

|-- main.py

|-- bootstrap\_reviewer.py

|-- download\_reviewer\_assets.py

|-- requirements-reviewer.txt

|-- RELEASE\_ASSETS\_MANIFEST.json

`-- COPYRIGHT.md



---



# Troubleshooting



## Check Runtime Assets



If MERIT reports missing model files or demonstration data, run:



```powershell

python .\\download\_reviewer\_assets.py --check



---



# Copyright and Review Anonymity



Copyright (c) 2026 MERIT Authors. All rights reserved.



This copyright notice applies to the original software, source code, system integration, user interfaces, quality-aware multimodal fusion implementation, deployment utilities, and documentation developed as part of the MERIT project.



Third-party software, pretrained models, datasets, and other external components included in or referenced by this repository remain subject to their respective original licenses, copyright notices, and terms of use.



The authors are identified anonymously as **MERIT Authors** during the double-blind review process. Author attribution may be updated after the review process concludes.



For additional information, see:



```text

COPYRIGHT.md
