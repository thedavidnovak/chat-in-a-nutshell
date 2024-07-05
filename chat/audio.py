#!/usr/bin/env python3

import logging

logging.basicConfig(encoding="utf-8", format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)

class CreateAudioError(Exception):
    pass

class SaveAudioError(Exception):
    pass

class Speaker:
    def __init__(self, client, model: str = "tts-1-hd", voice: str = "nova"):
        """Initialize the Speaker with a client, model, and voice."""
        self.model = model
        self.voice = voice
        self.client = client

    def _save_audio(self, audio_response, audio_file_path: str) -> None:
        """Save the audio response to a file."""
        try:
            audio_response.stream_to_file(audio_file_path)
        except Exception as e:
            logging.error(f'Could not save file: {e}')
            raise SaveAudioError('Could not save file.') from e

    def create_audio(self, text: str, audio_save: bool = False, audio_file_path: str = 'tts.aac') -> None:
        """Create audio from text using the specified model and voice."""
        try:
            response = self.client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                input=text
            )
        except Exception as e:
            logging.error(f'Failed to create audio: {e}')
            raise CreateAudioError('Failed to create audio.') from e

        if audio_save:
            try:
                self._save_audio(response, audio_file_path)
            except SaveAudioError:
                logging.error(f'Could not save file: {e}.')
                raise
            else:
                logging.info(f'Successfully saved to: {audio_file_path}')