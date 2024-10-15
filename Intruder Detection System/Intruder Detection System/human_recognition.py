from deepface import DeepFace
import os
import pandas as pd


class FaceRecognition:
    def __init__(self, data_set_dir='dataSet', threshold=0.4):
        self.data_set_dir = data_set_dir
        self.threshold = threshold

        self.username = {
            "user1": "woo",
            "user2": "obama",
            "user3": "trump",
            "user9": "chang"
        }

        # Ensure the dataset directory exists
        os.makedirs(self.data_set_dir, exist_ok=True)

    def recognize_face(self, image):
        try:
            name="Unknown"
            # Perform face recognition with DeepFace
            result = DeepFace.find(image, db_path=self.data_set_dir, enforce_detection=False,
                                   distance_metric="cosine", threshold=self.threshold, silent=True)
            if result and isinstance(result, list) and len(result) > 0:
                df = result[0]  # Get the first DataFrame from the result list
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # Get the identity from the best match
                    identity = os.path.basename(df.loc[df['distance'].idxmin()]['identity'])
                    user_id = identity.split('.')[0]  
                    name = self.username.get(user_id, "Unknown")
                    
            return name
        
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return "Unknown"
