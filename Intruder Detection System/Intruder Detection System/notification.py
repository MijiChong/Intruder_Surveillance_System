import requests

class TelegramBot:
    def __init__(self, chat_id):
        self.token = "7205505226:AAFftBbIj-3Qda9tIHZo14GTxG2aC8MFhcQ"
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{self.token}"

    def send_message(self, message):
        url = f"{self.base_url}/sendMessage?chat_id={self.chat_id}&text={message}"
        response = requests.post(url).json()
        return response

    def send_photo(self, photo_path):
        url = f"{self.base_url}/sendPhoto?chat_id={self.chat_id}"
        with open(photo_path, "rb") as photo_file:
            response = requests.post(url, files={"photo": photo_file}).json()
        return response

    def send_video(self, video_path):
        url = f"{self.base_url}/sendVideo?chat_id={self.chat_id}"
        with open(video_path, "rb") as video_file:
            response = requests.post(url, files={"video": video_file}).json()
        return response

