# Reviewer Quick Start

This repository contains two reviewer-facing modes:

- `MERIT-Offline Model` — offline EAV replay and time-aligned emotion recognition.
- `MERIT-Realtime Model` — realtime camera/microphone capture utilities for the same frozen runtime.

## 1. Create the reviewer environment

From the repository root:

```powershell
python .\bootstrap_reviewer.py --device cuda
```

For a CPU-only machine:

```powershell
python .\bootstrap_reviewer.py --device cpu
```

The bootstrap creates `.venv-reviewer`, validates dependencies, and runs the
runtime probe.

## 2. Offline mode

Windows PowerShell:

```powershell
& ".\.venv-reviewer\Scripts\python.exe" `
  ".\MERIT-Offline Model\03_reviewer_demo.py" `
  --python ".\.venv-reviewer\Scripts\python.exe" `
  --device auto `
  --fusion-device cpu
```

The spaces in `MERIT-Offline Model` are intentional. Keep the path quoted.

## 3. Realtime mode preflight

Before realtime capture, list/check the available camera and microphone devices:

```powershell
& ".\.venv-reviewer\Scripts\python.exe" `
  ".\MERIT-Realtime Model\01_probe_av_devices.py" `
  --list-only
```

A full realtime run can require operating-system permission for camera and
microphone access.

## Rename compatibility

The public folder names are exactly:

```text
MERIT-Offline Model
MERIT-Realtime Model
```

Historical internal words such as `ReviewerDemo` or `LiveInteraction` may still
appear in version/schema identifiers. They are not filesystem paths and do not
affect execution.

The folder rename does not alter model weights, Quality V1, thresholds,
F4/AF4-B, class order, or inference formulas.
