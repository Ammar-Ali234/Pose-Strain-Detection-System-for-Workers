# Install required packages if not already
# !pip install ultralytics

from ultralytics import YOLO
import cv2
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from google.colab.patches import cv2_imshow

# Load and fuse the models
pose_model = YOLO("yolo11n-pose.pt")
pose_model.fuse()

obj_model = YOLO("best 22")  # Replace with your model path
obj_model.fuse()

# Email details
sender_email = ""
password = "aagi svyu hviw ztat"
receiver_email = ""
subject = "Violation Detected"
body = "A person with ID: {} has been detected with violation: {}. Please take immediate action."

# Helper function to send email with an image attachment
def send_email_with_image(person_id, violation, frame):
    try:
        image_filename = f"violation_{person_id}_{int(time.time())}.jpg"
        cv2.imwrite(image_filename, frame)

        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = receiver_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body.format(person_id, violation), "plain"))

        with open(image_filename, "rb") as f:
            img_data = f.read()
            image_attachment = MIMEText(img_data, "base64", "utf-8")
            image_attachment.add_header("Content-Disposition", f"attachment; filename={image_filename}")
            image_attachment.add_header("Content-Type", "image/jpeg")
            msg.attach(image_attachment)

        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        print(f"✅ Email sent for Person ID: {person_id}, Violation: {violation} with image attached.")

        os.remove(image_filename)

    except Exception as e:
        print(f"❌ Error sending email: {e}")

    finally:
        server.quit()

# Class names for the object detection model
class_names = {
    0: 'Hardhat',
    2: 'NO-Hardhat',
    4: 'NO-Safety Vest',
    7: 'Safety Vest',
    5: 'Person'
}

# Open the video file
cap = cv2.VideoCapture("v.mp4")

frame_rate = cap.get(cv2.CAP_PROP_FPS)
frame_time = 1.0 / frame_rate

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_size = (frame_width, frame_height)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_combined_detection.mp4', fourcc, frame_rate, output_size)

# Track email sent status
ppe_email_sent_ids = set()  # For PPE violations (send only once)

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --------- Object Detection (PPE Check) ---------
    obj_results = obj_model.track(frame, stream=True)
    for result in obj_results:
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy()
            ids = result.boxes.id.cpu().numpy() if result.boxes.id is not None else None
            confs = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box[:4])
                class_id = int(cls[i])
                confidence = confs[i] * 100

                person_id = int(ids[i]) if ids is not None else -1

                if class_id in class_names:
                    label = class_names[class_id]
                    color = (0, 255, 0)

                    if label in ['NO-Hardhat', 'NO-Safety Vest']:
                        color = (0, 0, 255)

                        # Only send one-time email for PPE violations
                        if person_id not in ppe_email_sent_ids:
                            send_email_with_image(person_id, label, frame)
                            ppe_email_sent_ids.add(person_id)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label_text = f"ID: {person_id} {label} {confidence:.1f}%"
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # --------- Pose Detection (Ergonomic Check) ---------
    pose_results = pose_model.predict(frame, stream=True)
    for result in pose_results:
        if result.keypoints is not None:
            keypoints = result.keypoints.xy.cpu().numpy()

            for i, keypoint in enumerate(keypoints):
                person_id = i  # Pose model usually doesn't track ID separately, so just use i

                # Example condition: Check if head (keypoint 0) is very low compared to body (keypoint 5)
                if keypoint.shape[0] > 5:
                    head_y = keypoint[0][1]
                    mid_body_y = keypoint[5][1]

                    if head_y > mid_body_y + 50:  # Example threshold for bad posture
                        send_email_with_image(person_id, "Ergonomic Issue Detected", frame)

                # Draw skeleton points
                for point in keypoint:
                    x, y = map(int, point)
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

    # Save the frame
    out.write(frame)

    # Resize and show frame
    display_frame = cv2.resize(frame, (640, 480))
    cv2_imshow(display_frame)

cap.release()
out.release()
cv2.destroyAllWindows()
