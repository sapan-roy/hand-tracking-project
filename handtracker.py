import mediapipe as mp
import cv2


class HandTracker:
    def __init__(self,
                 static_image_mode=False,
                 max_num_hands=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        """Initializes a HandTracker object.

        Args:
          static_image_mode: Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream. See details in
            https://solutions.mediapipe.dev/hands#static_image_mode.
          max_num_hands: Maximum number of hands to detect. See details in
            https://solutions.mediapipe.dev/hands#max_num_hands.
          min_detection_confidence: Minimum confidence value ([0.0, 1.0]) for hand
            detection to be considered successful. See details in
            https://solutions.mediapipe.dev/hands#min_detection_confidence.
          min_tracking_confidence: Minimum confidence value ([0.0, 1.0]) for the
            hand landmarks to be considered tracked successfully. See details in
            https://solutions.mediapipe.dev/hands#min_tracking_confidence.
        """

        self.static_image_mode = static_image_mode
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode,
                                         max_num_hands,
                                         min_detection_confidence,
                                         min_tracking_confidence)

        self.results = None
        self.landmarks = []

    def detect_hands(self, image, draw=True):
        """Detects hands in the video

        :param image: Input video frame
        :param draw: Whether to draw hands connecting all landmarks or not. Defaults to True.
        :return: Original input video frame
        """

        # Flip image horizontally and convert image from BGR to RGB
        # mediapipe.Hands.process method expects a RGB image
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image)

        # Convert image back to BGR (original)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if draw:
            if self.results.multi_hand_landmarks:
                for hand_landmarks in self.results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

        return image

    def record_landmarks(self, image):
        """Finds hand landmarks and saves them as a list

        :param image: Original input video frame
        :return: None
        """
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[0]
            for _id, landmark in enumerate(my_hand.landmark):
                h, w, c = image.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                self.landmarks.append([_id, cx, cy])

    def show_landmark(self, image, landmark_num):
        """Highlights particular hand landmark

        :param image: Original video frame
        :param landmark_num: Hand landmark index
        :return: None
        """
        for landmarks in self.landmarks:
            cx, cy = self.landmarks[landmark_num][1:]
            cv2.circle(image, (cx, cy), 8, (255, 255, 0), cv2.FILLED)

    def open_finger_count(self):
        """Counts and displays number of open fingers

        :return: None
        """
        finger_tip_indexes = [4, 8, 12, 16, 20]
        open_fingers = []
        if len(self.landmarks) > 0:
            if self.landmarks[finger_tip_indexes[0]][1] < self.landmarks[finger_tip_indexes[0] - 1][1]:
                open_fingers.append(1)
            else:
                open_fingers.append(0)

            for index in range(1, 5):
                if self.landmarks[finger_tip_indexes[index]][2] < self.landmarks[finger_tip_indexes[index] - 2][2]:
                    open_fingers.append(1)
                else:
                    open_fingers.append(0)
        return open_fingers.count(1)


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            # If loading a video, use 'break' instead of 'continue'.
            break

        ht = HandTracker(max_num_hands=1, min_detection_confidence=0.85)
        image = ht.detect_hands(image)

        ht.record_landmarks(image)
        # landmarks sample value
        # [[0, 566, 443], [1, 492, 441], [2, 453, 417], [3, 448, 394], [4, 447, 376], [5, 447, 372], [6, 409, 310],
        #  [7, 392, 267], [8, 374, 224], [9, 495, 353], [10, 483, 264], [11, 482, 200], [12, 474, 142], [13, 539, 347],
        #  [14, 549, 266], [15, 554, 207], [16, 550, 152], [17, 578, 349], [18, 573, 307], [19, 566, 306], [20, 557, 304]]

        if len(ht.landmarks) > 0:
            ht.show_landmark(image, 8)

        count = ht.open_finger_count()

        # cv2.namedWindow("window", cv2.WND_PROP_VISIBLE)
        # cv2.setWindowProperty("window", cv2.WND_PROP_VISIBLE, cv2.WINDOW_NORMAL)
        cv2.putText(image, str(count), (100, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        cv2.imshow("window", image)
        cv2.waitKey(1)

        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()


if __name__ == '__main__':
    main()
