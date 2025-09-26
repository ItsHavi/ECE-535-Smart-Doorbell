# ECE-535-Smart-Doorbell
A smart doorbell system using a raspberry pi and a camera module.

Motivation:
Our main motivation for the project is to create another member of the smart home's devices using a readily available edge device such as the Raspberry Pi 

Design Goals:
We will attempt to make a compact and reasonable addon to a doorbell by leaving space for a Raspberry Pi and its camera module. It should be able to detect people who approach the door and recognize them as a known or unknown person. 

Deliverables:
• Learn how to deploy ML models on Raspberry Pi
• Implement a lightweight face detection or person detection model using TensorFlow Lite.
• Build a simple system that, when someone appears at the door, captures an image and classify
• Optional: Add an alert mechanism (e.g., send a notification to a phone or log it in a file)
• Fun feature!: Can it detect a delivery person? E.g., pizza or package delivery worker?
• The final output should be a code snippet and demonstration running inference directly on         Raspberry Pi

System Blocks:

Hardware and Software Requirements:
Raspberry Pi, camera module. For training, Google Colab can be used. You
can use any model for face detection. There are various approaches. One simple way is to compute embeddings of known/trusted people using a CNN and store them. When a person approaches the door, compare their embedding with known embeddings. If the embeddings match, you know it is a trusted person.

TEAM MEMBERS and roles:
Havi

Nicholas

Angad
