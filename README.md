# ECE-535-Smart-Doorbell
A smart doorbell system using a raspberry pi and a camera module.

**Motivation:**
Our main motivation for the project is to create another member of the smart home's devices, the smart doorbell, using a readily available edge device, the Raspberry Pi.

**Design Goals:**
We will attempt to make a compact and reasonable addon to a doorbell by leaving space for a Raspberry Pi and its camera module. It should be able to detect people who approach the door and recognize them as a known or unknown person. 

**Deliverables:**
• Learn how to deploy ML models on Raspberry Pi
• Implement a lightweight face detection or person detection model using TensorFlow Lite.
• Build a simple system that, when someone appears at the door, captures an image and classify
• Optional: Add an alert mechanism (e.g., send a notification to a phone or log it in a file)
• Fun feature!: Can it detect a delivery person? E.g., pizza or package delivery worker?
• The final output should be a code snippet and demonstration running inference directly on Raspberry Pi

**System Blocks:**
   Input of Camera & Motion Detector: Input: Raw video from camera module. Does: captures frames and signifies events if there's enough motion. -> Outputs: Image from that event frame and sends trigger signal.

   Face Detector: Input: Frame of event image. Does: Runs a face detection model to identify coordinates of any faces in the image. Outputs: Bounding box or cropped image of face, or nothing if it cant find anything.

   Face Recognizer: Input: Cropped image of face. Does: Runs the image on a trained CNN designed for face recognition. Output: A way to identify the faces features.

   Database: Input: That way to identify faces, the recognized faces database. Does: If close enough to the measure on the identified faces, we will call it a match, else it isn't. Outputs: Label on if it is a known person with their name, or unknown, like "Known Person: Havi" 

**Hardware and Software Requirements:**
Hardware:
- Raspberry Pi
- Camera Module 
Software:
- Google Colab
- Linux
- Python
- Tensorflow (determining face recognition model)

You can use any model for face detection. There are various approaches. One simple way is to compute embeddings of known/trusted people using a CNN and store them. When a person approaches the door, compare their embedding with known embeddings. If the embeddings match, you know it is a trusted person.

**TEAM MEMBERS and roles:**

Havi
 - Software
 
Nicholas 
 - Algorithm Design
 
Angad
 - Networking
 
Shared roles: Setup, Writing, Research

**Timeline:**
   All through:
     - Work on documents as we go.
     - Research constantly, not just in October.
     
   October:
    - Setting up hardware, and testing. 
    - Research.
    - Collecting a dataset for our faces.
    
   November:
    - Train model using Colab.
    - Add model on Raspberry Pi
    - Make software for image capture and detection.
    
   December: 
    - Polish Polish
    - See if we can add optional or additional features.
    - Finalize the documents, and report.
    - Present. 
    
**References:**
- A CNN paper on edge vision: https://arxiv.org/abs/1704.04861
