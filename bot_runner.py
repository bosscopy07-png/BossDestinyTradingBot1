import os
import telebot
import threading

BOT_TOKEN = os.getenv("BOT_TOKEN")
bot = telebot.TeleBot(BOT_TOKEN)

def start_bot_polling():
    @bot.message_handler(commands=["start"])
    def start_message(message):
        bot.reply_to(message, "🔥 Welcome to Boss Destiny Trading Empire!\nUse /help to see available commands.")

    @bot.message_handler(commands=["help"])
    def help_message(message):
        bot.reply_to(
            message,
            "📊 Available Commands:\n"
            "/start - Begin the bot\n"
            "/signal - Get latest trading signal\n"
            "/status - Check bot status\n"
            "/trending - Show trending pairs"
        )

    @bot.message_handler(commands=["signal"])
    def get_signal(message):
        bot.reply_to(message, "📈 Generating signal... please wait.")
        # Placeholder: later add your AI + market logic here
        bot.send_message(message.chat.id, "✅ Signal: BUY BTCUSDT (1h timeframe)\nLeverage: x20")

    print("🤖 Telegram bot polling started.")
    bot.infinity_polling(timeout=60, long_polling_timeout=30)

def stop_existing_bot_instances():
    try:
        os.system("pkill -f telebot")  # kills other bot processes
    except Exception:
        pass
