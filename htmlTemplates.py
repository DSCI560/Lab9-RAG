css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

* { box-sizing: border-box; }

.chat-message {
    padding: 1.2rem 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    font-family: 'IBM Plex Sans', sans-serif;
    animation: fadeIn 0.3s ease;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

.chat-message.user {
    background: linear-gradient(135deg, #1e2433 0%, #2b3245 100%);
    border-left: 3px solid #4a9eff;
}

.chat-message.bot {
    background: linear-gradient(135deg, #1a2a1a 0%, #243324 100%);
    border-left: 3px solid #4ade80;
}

.chat-message .avatar {
    flex-shrink: 0;
    width: 48px;
    height: 48px;
}

.chat-message .avatar img {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid rgba(255,255,255,0.15);
}

.chat-message .message {
    flex: 1;
    color: #e8eaf0;
    font-size: 0.95rem;
    line-height: 1.65;
    padding-top: 0.2rem;
}

.chat-message.user .message { color: #c8d6f0; }
.chat-message.bot  .message { color: #c8f0d0; }

.chat-message .message code {
    font-family: 'IBM Plex Mono', monospace;
    background: rgba(255,255,255,0.1);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 0.85rem;
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/0/0c/Chatbot_img.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://upload.wikimedia.org/wikipedia/commons/9/99/Sample_User_Icon.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''