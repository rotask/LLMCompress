import os
import smtplib
from email.mime.text import MIMEText
from time import sleep

# Email details
SENDER_EMAIL = os.getenv('EMAIL_USER')
RECEIVER_EMAIL = 'konstantinos.rotas23@imperial.ac.uk'
EMAIL_PASSWORD = os.getenv('EMAIL_PASS')
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

# Thresholds
GPU_UTIL_THRESHOLD = 1  # GPU utilization threshold in percentage
MEMORY_UTIL_THRESHOLD = 5  # Memory utilization threshold in percentage

def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print('Email sent successfully!')
    except Exception as e:
        print(f'Failed to send email: {e}')

def check_gpu():
    # Check GPU utilization
    gpu_utilization_result = os.popen("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits").read()
    gpu_utilization = [int(x) for x in gpu_utilization_result.strip().split("\n")]
    
    # Check GPU memory utilization
    memory_utilization_result = os.popen("nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits").read()
    memory_utilization = [int(x.split(', ')[0]) / int(x.split(', ')[1]) * 100 for x in memory_utilization_result.strip().split("\n")]

    # Return True if both GPU and memory utilization are below their thresholds
    return all(u < GPU_UTIL_THRESHOLD for u in gpu_utilization) and all(m < MEMORY_UTIL_THRESHOLD for m in memory_utilization)

def monitor_gpu():
    email_sent = False
    while True:
        if check_gpu():
            if not email_sent:
                send_email('GPU Resources Available - Gajasura', 'The GPU resources in Gajasura server are now available.')
                email_sent = True
        else:
            email_sent = False
        sleep(200)  # Check every 5 minutes

if __name__ == '__main__':
    monitor_gpu()
