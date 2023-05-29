import tkinter as tk
from chat import ChatBot


class BotApp:
    def __init__(self, master):
        self.master = master
        self.master.title("ChatBot")

        # create chatbot object
        self.chatbot = ChatBot('intents.json')

        # create widgets
        self.label = tk.Label(self.master, text="ChatBot")
        self.label.pack()

        self.chat_history = tk.Text(self.master, width=50, height=20)
        self.chat_history.pack()

        self.input_box = tk.Entry(self.master, width=50)
        self.input_box.pack()
        self.input_box.bind('<Return>', self.send_message)

        self.send_button = tk.Button(self.master, text="Send", command=self.send_message)
        self.send_button.pack()

    def send_message(self, event=None):
        # get user input
        user_input = self.input_box.get()
        self.input_box.delete(0, tk.END)

        # display user input in chat history
        self.chat_history.insert(tk.END, "You: " + user_input + "\n")

        if user_input.lower() == "quit":
            self.master.quit()
            return

        # get bot response
        bot_response = self.chatbot.get_response(user_input)

        # display bot response in chat history
        self.chat_history.insert(tk.END, "Bot: " + str(bot_response) + "\n")

        # scroll to the end of the chat history
        self.chat_history.see(tk.END)


root = tk.Tk()
app = BotApp(root)
root.mainloop()
