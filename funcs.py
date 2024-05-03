import os
import logging
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OpenAIAssistant:
    def __init__(self, vector_store_id, assistant_id):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store_id = vector_store_id
        self.assistant_id = assistant_id
        self.thread = self.create_thread()

    def create_thread(self):
        logging.info("Creating conversation thread.")
        return self.client.beta.threads.create(tool_resources={"file_search": {"vector_store_ids": [self.vector_store_id]}})

    def upload_file_and_send_message(self, path):
        logging.info("Uploading file to Vector Storage.")
        absolute_path = os.path.abspath(path)
        with open(absolute_path, "rb") as file:
            file_id = self.client.files.create(file=file, purpose="assistants").id
        self.client.beta.vector_stores.files.create_and_poll(vector_store_id=self.vector_store_id, file_id=file_id)
        self.enqueue_message(file_id)

    def enqueue_message(self, file_id):
        logging.info("Enqueuing message.")
        message = f"create a detailed minute of this file: file_id:{file_id}"
        attachments = [{"file_id": file_id, "tools": [{"type": "file_search"}]}]
        message = {
            "thread_id": self.thread.id,
            "message": {
                "role": "user",
                "content": message,
                "attachments": attachments
            }
        }
        self.client.beta.threads.messages.create(**message)

    def get_response(self):
        run = self.client.beta.threads.runs.create(thread_id=self.thread.id, assistant_id=self.assistant_id)
        while True:
            run = self.client.beta.threads.runs.retrieve(run_id=run.id, thread_id=self.thread.id)
            if run.status == "completed":
                messages = self.client.beta.threads.messages.list(thread_id=self.thread.id, order="asc")
                for msg in messages.data:
                    if msg.role == 'assistant':
                        return msg.content[0].text.value
                return "No response available."
            time.sleep(2)
# Path: funcs.py

