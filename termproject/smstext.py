# twilio 패키지를 활용하여 sms 문자 보내기

# my twilio phone number : +19785064510
# account SID : AC90bf53fbfdae01201bc6d683ba06358b
# auth Token : 1d43880145db0522a333ea5214d243a8

from twilio.rest import Client

account_sid = 'AC90bf53fbfdae01201bc6d683ba06358b'
auth_token = '1d43880145db0522a333ea5214d243a8'
client = Client(account_sid, auth_token)

message = client.messages \
                .create(
                    body="장애인 전용 구역 주차자리에 주차하셨습니다. 얼른 빼주십쇼.",
                    from_='+19785064510',
                    to='+821065882442'
                    )