css = '''
<style>
.chat-message {
    padding: 1.5rem;
    border-radius: 0.75rem;
    margin-bottom: 1rem;
    display: flex;
    align-items: flex-start;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.chat-message.user {
    background-color: #2b313e;
}
.chat-message.bot {
    background-color: #475063;
}
.chat-message .avatar {
    width: 15%;
    text-align: center;
}
.chat-message .avatar img {
    max-width: 64px;
    max-height: 64px;
    border-radius: 50%;
    object-fit: cover;
}
.chat-message .message {
    width: 85%;
    padding: 0 1rem;
    color: #fff;
}
.chat-message .content {
    font-size: 1rem;
    margin-bottom: 0.3rem;
}
.chat-message .timestamp {
    font-size: 0.75rem;
    color: #d0d0d0;
    text-align: right;
}
</style>
'''


import base64
import datetime


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def get_timestamp():
    return datetime.datetime.now().strftime("%I:%M %p")

user_avatar = img_to_base64("./images/user.jpeg") 
bot_avatar = img_to_base64("./images/bot.jpeg")     

user_template = f'''
<div class="chat-message user">
    <div class="avatar">
        <img src="data:image/png;base64,{user_avatar}">
    </div>
    <div class="message">
        <div class="content">{{{{MSG}}}}</div>
        <div class="timestamp">{get_timestamp()}</div>
    </div>
</div>
'''

bot_template = f'''
<div class="chat-message bot">
    <div class="avatar">
        <img src="data:image/png;base64,{bot_avatar}">
    </div>
    <div class="message">
        <div class="content">{{{{MSG}}}}</div>
        <div class="timestamp">{get_timestamp()}</div>
    </div>
</div>
'''
