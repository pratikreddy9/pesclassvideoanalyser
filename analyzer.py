import cv2
import base64
import json
import math
import time
from openai import OpenAI

# OpenAI API Key
api_key = ""
client = OpenAI(api_key=api_key)

# Input paths
video_path = r"C:\Users\krish\Downloads\recordings\B 0204_NoClassess.mp4"
reference_image_path = r"C:\Users\krish\Downloads\recordings\B 0204_Before_Classess.jpg"

print("Starting video analysis...")
all_responses = []  # Stores valid responses

# Function to encode an image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Step 1: Extract frames (1 per minute)
print(f"Opening video: {video_path}")
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    raise ValueError("Error: Could not read video FPS. Check the video file.")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_seconds = frame_count / fps
total_minutes = math.ceil(duration_seconds / 60.0)
print(f"Video duration: {total_minutes} minutes, extracting 1 frame per minute...")

frames = []  # Holds (minute_index, image_base64)

for minute in range(total_minutes):
    cap.set(cv2.CAP_PROP_POS_MSEC, minute * 60 * 1000)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: No frame captured at minute {minute}.")
        continue

    frame_resized = cv2.resize(frame, (320, 180))  
    success, buffer = cv2.imencode(".jpg", frame_resized, [int(cv2.IMWRITE_JPEG_QUALITY), 30])  
    if not success:
        print(f"Error: Failed to encode frame at minute {minute}. Skipping...")
        continue

    frame_b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    frames.append((minute, frame_b64))

cap.release()
print(f"Total frames extracted: {len(frames)}")

# Step 2: Encode the reference empty classroom image
print(f"Loading reference image: {reference_image_path}")
reference_b64 = encode_image(reference_image_path)

# Step 3: Process frames in batches of 5 with retry logic
batch_size = 5
print("Processing frames in batches of 5...")

for i in range(0, len(frames), batch_size):
    batch = frames[i:i + batch_size]
    print(f"Sending batch {i // batch_size + 1} to OpenAI API...")

    # **Updated prompt with better explanation**
    content = [
        {
            "type": "text",
            "text": (
                "You are analyzing classroom images to determine if a lecture is happening. "
                "A lecture is happening if a teacher is addressing students and using the blackboard or whiteboard. "
                "The **first image is a reference of an empty classroom**—it shows the room when no class is happening. "
                "The remaining images are frames from a lecture video. Compare them to the empty classroom image "
                "and return a structured JSON response indicating whether a class is taking place in each frame. \n\n"
                "**Return JSON in this exact format:**\n"
                "{\n"
                "  \"analysis\": [\n"
                "    {\"minute\": 1, \"flag\": \"Yes\", \"confidence_score\": 0.95},\n"
                "    {\"minute\": 2, \"flag\": \"No\", \"confidence_score\": 0.90}\n"
                "  ]\n"
                "}"
            ),
        }
    ]
    
    # Attach reference image (empty classroom)
    content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/jpeg;base64,{reference_b64}",
            "detail": "low"
        }
    })

    # Attach batch of frames
    for minute, frame_b64 in batch:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{frame_b64}",
                "detail": "low"
            }
        })

    # API Retry Logic (Max 3 Attempts)
    retries = 3
    success = False
    while retries > 0:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze classroom images. Return JSON array under the key 'analysis'."
                    },
                    {"role": "user", "content": content}
                ],
                max_tokens=2000
            )

            print(f"RAW RESPONSE (Batch {i // batch_size + 1}): {response}")

            # Ensure we have a valid JSON response
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                response_text = response.choices[0].message.content
                try:
                    result_json = json.loads(response_text)
                    if "analysis" in result_json:
                        for res in result_json["analysis"]:
                            res["minute"] = int(res["minute"])  # Convert to integer
                        all_responses.extend(result_json["analysis"])
                        print(f"Batch {i // batch_size + 1} processed successfully.")
                        success = True
                        break
                    else:
                        print(f"Unexpected API response format: {response_text}")

                except json.JSONDecodeError:
                    print(f"Error decoding JSON for batch {i // batch_size + 1}: {response_text}")

        except Exception as e:
            print(f"Error processing batch {i // batch_size + 1}, retrying ({3 - retries + 1}/3): {e}")

        retries -= 1
        time.sleep(2)  # Wait before retry

# Step 4: Determine final decision based on majority vote
print("Finalizing results...")

def determine_final_decision(responses):
    if not responses:
        return {"lecture_detected": "No"}  # Default to "No" if no valid responses

    yes_count = sum(1 for res in responses if res["flag"] == "Yes")
    no_count = sum(1 for res in responses if res["flag"] == "No")

    lecture_detected = "Yes" if yes_count > no_count else "No"

    return {"lecture_detected": lecture_detected}

final_output = determine_final_decision(all_responses)

# Print final JSON output
print("Final decision:")
print(json.dumps(final_output, indent=2))
