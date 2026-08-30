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

        self.api_key = '***REMOVED-AZURE-KEY***'
        self.region='eastus'

        if not self.api_key or not self.region:
            raise RuntimeError("Missing Azure credentials")

        self.is_active = False
        self.last_wake_time = 0.0
        self.wake_window = 15.0

        self.wake_phrases = [
            "hey anky", "hey anki", "hay anky", "hay anki",
            "hey, anki", "hey, anky", "hay, anky", "hay, anki",
            "hey yankee", "hey, yankee","hay yankee","hay, yankee",
            "hey enki", "hey, enki","hay enki","hay, enki",
            "hanky", "henky", "hen key", "yankee",
            "hey ankle", "hay ankle", "hey uncle", "hay uncle",
            "hey angie", "hay angie", "hey annie", "hay annie",
            "hey inky", "hay inky", "hey onkey", "hay onkey",
            "a anky", "a anki", "k anky", "okay anky",
            "hey nanky", "hey ankie", "hankie", "anky", "anki"
        ]

        # Stop-related phrases work at all times, with or without the
        # wake word, so the dino can always be halted immediately.
        self.stop_phrases = [
            "stop", "slop", "stand", "stand up", "stand-up", "still",
            "stops", "stopped", "stahp",
            "top", "shop",
            "standing", "stan", "stan up",
            "steel", "stil"
        ]

        speech_config = speechsdk.SpeechConfig(
            subscription=self.api_key,
            region=self.region
        )
        speech_config.speech_recognition_language = "en-US"

        try:
            audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
            self.get_logger().info("Default microphone configured successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to configure default microphone: {e}")

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
        while rclpy.ok():
            result = self.speech_recognizer.recognize_once_async().get()

            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                recognized_text = result.text.lower().strip(".,! ")
                current_time = time.time()

                self.get_logger().info(f"Recognized: {recognized_text}")

                # Stop is a safety command - always active, no wake word required.
                if any(phrase in recognized_text for phrase in self.stop_phrases):
                    self.get_logger().info("Stop command received (wake word not required).")
                    self.publish_text("stop")
                    self.last_wake_time = current_time
                    continue

                if any(phrase in recognized_text for phrase in self.wake_phrases):
                    self.is_active = True
                    self.last_wake_time = current_time
                    self.get_logger().info("Dino activated. Awaiting command.")
                    self.publish_text("activated")
                    continue

                if self.is_active and (current_time - self.last_wake_time) > self.wake_window:
                    self.is_active = False
                    self.get_logger().info("Wake window expired.")

                if self.is_active:
                    if any(phrase in recognized_text for phrase in [
                        "roar", "speak",
                        "rawr", "roared", "roars", "raw",
                        "speech", "speaker"
                            ]):
                        self.get_logger().info("Dino roars.")
                        self.publish_text("roar")
                        self.last_wake_time = current_time

                    elif any(phrase in recognized_text for phrase in [
                          "sleep", "lie down", "lay down", "lie-down",
                          "sleepy", "sleeping", "sleap",
                          "lay-down", "lie town", "lay town", "lied down",
                          "line down", "lion down", "light down"
                      ]):
                        self.get_logger().info("Dino goes lays down.")
                        self.publish_text("sleep")
                        self.is_active = False

                    elif any(phrase in recognized_text for phrase in [
                            "turn left", "go left", "left turn",
                            "turn-left", "turnleft", "left"
                        ]):
                        self.get_logger().info("Dino turns left.")
                        self.publish_text("turn left")
                        self.last_wake_time = current_time

                    elif any(phrase in recognized_text for phrase in [
                            "turn right", "go right", "right turn",
                            "turn-right", "turnright", "right"
                        ]):
                        self.get_logger().info("Dino turns right.")
                        self.publish_text("turn right")
                        self.last_wake_time = current_time

                    elif any(phrase in recognized_text for phrase in [
                            "walk","start","move","forward",
                            "walked", "walking", "wok", "walc",
                            "started", "starting", "star",
                            "moves", "moving", "move it",
                            "forwards", "for word", "foreword"
                        ]):
                        self.get_logger().info("Dino walks/moves.")
                        if any(phrase in recognized_text for phrase in [
                                "speed up", "speed-up","fast","quickly","high speed","quick",
                                "speed it up", "spead up", "speeded up",
                                "faster", "quicker", "hi speed", "high-speed"
                            ]):
                            self.publish_text("walk faster")
                            self.last_wake_time = current_time
                        elif any(phrase in recognized_text for phrase in [
                                "slower","slow", "down",
                                "slow down", "slow-down", "sloe", "slo",
                                "go slow", "go down"
                            ]):
                            self.publish_text("walk slower")
                            self.last_wake_time = current_time
                        else:
                            self.publish_text("walk normal")

                    elif any(phrase in recognized_text for phrase in [
                            "push up", "push ups", "pushup", "pushups", "push-up", "push-ups",
                            "push it up", "pushed up",
                            "bush up", "push app", "push a", "poosh up",
                            "push ap"
                        ]):
                        self.get_logger().info("Dino flexes on em.")
                        self.publish_text("push up")
                        self.last_wake_time = current_time
                else:
                    self.get_logger().debug("Dino is inactive.")

            elif result.reason == speechsdk.ResultReason.NoMatch:
                pass

            elif result.reason == speechsdk.ResultReason.Canceled:
                details = result.cancellation_details
                self.get_logger().error(f"Speech recognition canceled: {details.reason}")
                self.publish_text("not listening")
                if details.reason == speechsdk.CancellationReason.Error:

                    self.get_logger().error(f"Error details: {details.error_details}")
                break


def main(args=None):
    rclpy.init(args=args)
    node = SpeechRecognitionNode()

    try:
        node.run_recognition_loop()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
