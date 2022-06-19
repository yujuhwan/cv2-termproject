# twilio 패키지를 활용하여 sms 문자 보내기

from twilio.rest import Client

# my twilio phone number : +19785064510
# account SID : AC90bf53fbfdae01201bc6d683ba06358b
# auth Token : 1d43880145db0522a333ea5214d243a8

account_sid = 'AC90bf53fbfdae01201bc6d683ba06358b'
auth_token = '4df027771e1d7f865e5a4d078e08d211'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                    body='차 빼주세요',
                    from_='+19785064510',
                    to='+821065882442'
                    )

print(message.sid)