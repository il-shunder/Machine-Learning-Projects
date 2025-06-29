import os
import random
from datetime import datetime

import chatbot_assistant
import streamlit as st
import streamlit.components.v1 as components

MODEL_PATH = "assistant.pth"
DIMENSIONS_PATH = "dimensions.json"

if __name__ == "__main__":

    def get_time():
        current_time = datetime.now().strftime("%I:%M %p")
        return f"It is {current_time}"

    def get_date():
        current_date = datetime.now().strftime("%Y-%m-%d")
        return f"Today's date is {current_date}"

    def get_stock():
        stocks = ["AAPL", "GOOGL", "MSFT"]
        return random.choice(stocks)

    assistant = None

    if not os.path.exists(MODEL_PATH):
        assistant = chatbot_assistant.ChatbotAssistant(
            "intents.json", method_mappings={"get_time": get_time, "get_date": get_date, "get_stock": get_stock}
        )
        assistant.parse_intents()
        assistant.prepare_data()
        assistant.train_model(batch_size=8, lr=0.001, epochs=100)

        assistant.save_model(MODEL_PATH, DIMENSIONS_PATH)
    else:
        assistant = chatbot_assistant.ChatbotAssistant(
            "intents.json", method_mappings={"get_time": get_time, "get_date": get_date, "get_stock": get_stock}
        )
        assistant.parse_intents()
        assistant.load_model(MODEL_PATH, DIMENSIONS_PATH)

    if assistant:
        counter = 0

        st.write(
            """ <style>
                [class^="st-key-delete-button-"] {
                    /* Your CSS styles here */
                }
                .user-message {
                    margin-left: auto;
                }
            </style>""",
            unsafe_allow_html=True,
        )
        st.title("Chatbot")

        # while True:
        counter += 1
        placeholder = st.empty()
        message = placeholder.text_input(
            "Enter your message (to quit, enter '/quit') 👇",
            placeholder="Enter your message",
            key=counter,
            disabled=False,
        )

        components.html(
            f"""
                <script>
                    console.log({counter});
                    function focusInput() {{
                        const input = window.parent.document.querySelector('input');
                        if (input) {{
                            input.focus();
                        }} else {{
                            setTimeout(focusInput, 100);
                        }}
                    }}
                    focusInput();
                </script>
            """,
            height=0,
            width=0,
        )

        if message:
            # if message == "/quit":
            #     break

            html_str = f"""
            <div class="user-message">You: {message}</div>
            <div class="chatbot-message">Chatbot: {assistant.process_message(message)}</div>
            """

            st.markdown(html_str, unsafe_allow_html=True)
            placeholder.empty()
        else:
            st.stop()
