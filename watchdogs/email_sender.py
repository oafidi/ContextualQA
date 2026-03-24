import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import os

load_dotenv()

# Create the email structure (supports both subject and body)
msg = MIMEMultipart()
msg["From"]    = os.getenv("SENDER")
msg["To"]      = os.getenv("RECEIVER")
msg["Subject"] = "Darija Scraper Alert — No question generated -- ASUS TUF"

# Email body
body = """Hello Omar I'm from your laptop ASUS TUF A15,

The Darija scraper has stopped generating questions.

Possible causes:
  - A CAPTCHA appeared in the browser
  - Gemini returned invalid responses too many times
  - A rate limit was hit

Please check the browser and the log file (scraper.log).

Time: {}
""".format(__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Attach the body to the email
msg.attach(MIMEText(body, "plain"))

# Create SMTP session
s = smtplib.SMTP(os.getenv("SMTP_SERVER"), int(os.getenv("SMTP_PORT")))

# Start TLS encryption
s.starttls()

# Login with Gmail App Password
s.login(os.getenv("SENDER"), os.getenv("APP_PASSWORD"))

# Send the email (convert msg object to string with as_string())
s.sendmail(os.getenv("SENDER"), os.getenv("RECEIVER"), msg.as_string())

# Close the session
s.quit()

print("Alert email sent successfully")