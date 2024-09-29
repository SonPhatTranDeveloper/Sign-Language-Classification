import cv2
import mediapipe as mp
import argparse

from utils.features import *
from utils.model import ASLClassificationModel

# Temporarily ignore warning
import warnings
warnings.filterwarnings("ignore")

# Initialize MediaPipe Holistic
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if __name__ == "__main__":
    # Get the pose name from argument
    parser = argparse.ArgumentParser("Pose Data Capture")

    # Add and parse the arguments
    parser.add_argument("--model_path", help="Path of the ASL classification model",
                        type=str, default="models/simple_5_expression_model.pkl")
    parser.add_argument("--confidence", help="Confidence of the model",
                        type=float, default=0.6)
    args = parser.parse_args()

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    # Load model
    print("Initialising model ...")
    model = ASLClassificationModel.load_model(args.model_path)

    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=args.confidence,
                                      min_tracking_confidence=args.confidence)

    # Initialize MediaPipe Hands
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2,
                           min_detection_confidence=args.confidence,
                           min_tracking_confidence=args.confidence)

    # Initialize drawing utility
    mp_drawing = mp.solutions.drawing_utils
    drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    # Starting the application
    print("Starting application")

    # Set up the holistic model
    while cap.isOpened():
        # Check if getting frame is successful
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the image to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image and find faces
        face_results = face_mesh.process(image)

        # Process the image and find hands
        hand_results = hands.process(image)

        # Extract feature from face and hand results
        feature = extract_features(mp_hands, face_results, hand_results)
        expression = model.predict(feature)

        # Prepare text to display
        text = expression

        # Get frame dimensions
        height, width, _ = image.shape

        # Set text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3
        font_color = (255, 255, 255)  # White color
        font_thickness = 2

        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        # Calculate text position (top right corner with padding)
        padding = 10
        text_x = width - text_size[0] - padding
        text_y = text_size[1] + padding

        # Draw black background rectangle
        cv2.rectangle(image, (text_x - 5, text_y - text_size[1] - 5),
                      (width - 5, text_y + 5), (0, 0, 0), -1)

        # Draw text on the frame
        cv2.putText(image, text, (text_x, text_y), font, font_scale, font_color, font_thickness)

        # Draw the face mesh annotations on the image
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                )

        # Draw the hand annotations on the image
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
                )

        # Convert back to BGR to render
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Display the image
        cv2.imshow('ASL Classification Model', cv2.flip(image, 1))

        # Press 'q' to quit
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
