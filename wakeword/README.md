# wakeword/

openWakeWord detection loop (**M4**).

Listens continuously for **"Viky"**. Until a trained `viky.onnx` exists, a
pre-trained fallback model (`hey_jarvis`) is used, controlled by
`VIKY_WAKEWORD_FALLBACK=true`. Emits an earcon on detection.
