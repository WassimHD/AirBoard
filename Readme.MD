🚀 Introducing AirBoard — an AI-powered touchless whiteboard experience

After weeks of iteration and testing, I’m excited to share AirBoard, a computer-vision-based system that lets you draw, write, and interact using only hand gestures — no mouse, keyboard, or touchscreen needed.

💡 What it does:

Tracks hand and finger movements in real time using MediaPipe + OpenCV

Lets you switch between a Virtual Keyboard and an Air Drawer

Supports hover-based typing, gesture-based drawing, and AI-assisted text search

Integrates with LLaMA 4 Scout (via OpenRouter) to process queries directly from the interface

Saves your sketches as mini-previews, which can be cleared or reviewed anytime

🧠 Tech Stack:

Python for the main logic

OpenCV + MediaPipe for gesture detection & hand tracking

NumPy for image and coordinate processing

OpenRouter API for connecting with Meta’s LLaMA 4 model

Threading for smooth asynchronous AI responses
