# ============================================================
# DecodeLabs — Project 1: Rule-Based AI Chatbot
# Architecture: IPO Model | Engine: Dictionary (O1 lookup)
# ============================================================

# --- KNOWLEDGE BASE (Dictionary — O(1) lookup) ---------------
responses = {
    # Greetings
    "hello":        "Hello! I'm DecodeBot. How can I help you today?",
    "hi":           "Hey there! What can I do for you?",
    "hey":          "Hey! DecodeBot online. What's on your mind?",
    "good morning": "Good morning! Ready to learn some AI today?",
    "good evening": "Good evening! How can I assist you?",

    # About
    "who are you":      "I'm DecodeBot, a rule-based AI built at DecodeLabs.",
    "what are you":     "I'm a rule-based chatbot — pure logic, no machine learning.",
    "what can you do":  "I can answer questions, greet you, and tell you about AI!",

    # AI topics
    "what is ai":           "AI is the simulation of human intelligence by machines.",
    "what is machine learning": "ML is AI that learns patterns from data automatically.",
    "what is a chatbot":    "A chatbot is a program that simulates conversation with humans.",
    "what is python":       "Python is the #1 programming language for AI development.",
    "what is a dictionary": "In Python, a dictionary stores key-value pairs for O(1) lookup.",

    # DecodeLabs
    "what is decodelabs":   "DecodeLabs is your AI training hub — building real engineers.",
    "tell me about this project": "Project 1 teaches rule-based AI using control flow and logic.",

    # Feelings / smalltalk
    "how are you":      "I'm running at 100% efficiency. All systems operational!",
    "what is your name":"My name is DecodeBot. Pleased to meet you.",
    "tell me a joke":   "Why did the AI break up with the if-statement? Too many conditions.",
    "thank you":        "You're welcome! Keep building!",
    "thanks":           "Anytime! Happy to help.",

    # Help
    "help":  "Try: hello / what is ai / tell me a joke / what is python / bye",
}

# --- FALLBACK RESPONSE ----------------------------------------
FALLBACK = "I don't understand that yet. Try 'help' to see what I know."

# --- EXIT KEYWORDS --------------------------------------------
EXIT_COMMANDS = {"bye", "exit", "quit", "goodbye", "stop"}

# --- GREETING -------------------------------------------------
print("=" * 55)
print("  DecodeBot v1.0 | DecodeLabs AI Training Kit")
print("  Type 'help' for commands | Type 'bye' to exit")
print("=" * 55)

# --- MAIN LOOP (The Heartbeat) --------------------------------
while True:

    # PHASE 1: INPUT & SANITIZATION
    raw_input_text = input("\nYou: ")
    clean_input = raw_input_text.lower().strip()

    # Skip empty inputs
    if not clean_input:
        continue

    # PHASE 2: EXIT STRATEGY (kill command)
    if clean_input in EXIT_COMMANDS:
        print("Bot: Goodbye! Keep coding. Session terminated.")
        break

    # PHASE 3: PROCESS — O(1) dictionary lookup with fallback
    reply = responses.get(clean_input, FALLBACK)

    # PHASE 4: OUTPUT
    print(f"Bot: {reply}")