# 장애인 전용 주차 구역이나 불법 주차를 하게 되면 차량 번호를 입력하면 주인에게 차를 빼라는 메시지를 보내는 프로그램

from twilio.rest import Client  # twilio 패키지에서 Client 모듈 import

# 차량번호와 차량주인의 핸드폰 번호를 dictionary 형식으로 저장
car_number = {"69두3842": "+821065882442", "23오0438": "+821049102111"}

# twilio 개인 정보
account_sid = 'AC90bf53fbfdae01201bc6d683ba06358b'
auth_token = '4df027771e1d7f865e5a4d078e08d211'
client = Client(account_sid, auth_token)

for k in car_number:
    k = input("차량 번호 입력: ")
    if k in car_number:
        print("차량 주인의 번호:"+ car_number.get(k))  # 차량 번호의 value 값인 차량 주인 핸드폰 번호
        a = input("이 차량 주인에게 문자를 보낼까요? (y/n) : ")
        if a == 'y':
            print("문자를 발송하겠습니다.")
            message = client.messages \
                            .create(
                                    body='불법 주차 구역 입니다. 얼른 차를 빼주세요.',
                                    from_='+19785064510',           # 나의 twilio 폰 번호
                                    to=car_number.get(k)            # 입력한 차량 번호에 대한 차량 주인의 번호
                                    )

        else:
            print("문자를 발송하지 않습니다.")

    else:
        print("등록되지 않은 차량입니다.")