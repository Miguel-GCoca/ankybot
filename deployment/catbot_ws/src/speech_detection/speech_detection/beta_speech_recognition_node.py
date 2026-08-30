#!/usr/bin/env python3
import os
import time

import rclpy
from rclpy.node import Node
from example_interfaces.msg import String
import azure.cognitiveservices.speech as speechsdk


class SpeechRecognitionNode(Node):
    def __init__(self):
        super().__init__('speech_recognizer')

        self.publisher_ = self.create_publisher(String, 'speech_to_text', 10)
        self.is_recognizing = False
        self.timer = self.create_timer(0.1, self.run_recognition_loop)
        self.api_key = '***REMOVED-AZURE-KEY***'
        self.region='eastus'

        if not self.api_key or not self.region:
            self.get_logger().error("Could not find api_key or region in DinoCommandVariables.txt")
            raise RuntimeError("Missing Azure credentials")

        self.is_active = False
        self.last_wake_time = 0.0
        self.wake_window = 15.0

        self.wake_phrases = [
            "hey anky", "hey anki", "hey ankylo", "hey ankylosaurus",
            "hey, anki", "hey, anky", "hey, ankylo", "hey, ankylosaurus"
        ]

        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key,
            region=self.region
        )
        speech_config.speech_recognition_language = "en-US"

        wav_path = os.path.expanduser(
            "~/catbot_ws/src/speech_detection/wav_file_generator/test.wav"
            )

        self.get_logger().info(f"Using wav: {wav_path}")
        self.get_logger().info(f"Exists: {os.path.exists(wav_path)}")

        audio_config = speechsdk.audio.AudioConfig(filename=wav_path)
        self.speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )

        self.speech_recognizer.properties.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_InitialSilenceTimeoutMs,
            "60000"
        )
        self.speech_recognizer.properties.set_property(
            speechsdk.PropertyId.SpeechServiceConnection_EndSilenceTimeoutMs,
            "20000"
        )

        self.get_logger().info("System Online. Say 'Hey Anky' to begin.")

    def publish_text(self, text):
        msg = String()
        msg.data = text
        self.publisher_.publish(msg)
        self.get_logger().info(f'Published: "{text}"')

    def run_recognition_loop(self):

        if self.is_recognizing:
            return

        self.is_recognizing = True

        try:
            result = self.speech_recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                recognized_text = result.text.lower().strip(".,! ")
                current_time = time.time()

                self.get_logger().info(f"Recognized: {recognized_text}")

                if any(phrase in recognized_text for phrase in self.wake_phrases):
                    self.is_active = True
                    self.last_wake_time = current_time
                    self.get_logger().info("Dino activated.")
                    self.publish_text("activated")
                    return

                if self.is_active and (current_time - self.last_wake_time) > self.wake_window:
                    self.is_active = False

                if self.is_active:
                    if "roar" in recognized_text:
                        self.publish_text("roar")
                        self.last_wake_time = current_time
                    elif "sleep" in recognized_text:
                        self.publish_text("sleep")
                        self.is_active = False
                    elif "walk" in recognized_text:
                        self.publish_text("walk")
                        self.last_wake_time = current_time

            elif result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                self.get_logger().error(f"Canceled: {details.reason}")

        finally:
            self.is_recognizing = False

def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()



