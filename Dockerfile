# FROM: 베이스 이미지를 지정 (여기서는 ubuntu 20.04 버전 사용)
FROM python:3.8-slim

# RUN : 해당 명령어 실행, 필요한 패키지를 설치
RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -y libpq-dev gcc

# COPY: 현재 경로(.)에 존재하는 파일들을 이미지 /app 경로에 모두 추가
COPY . /app

# WORKDIR: 작업 디렉토리 변경. 셸 cd /app 과 같은 기능
WORKDIR /app

# RUN: 명령어 실행. 복사된 requirements.txt 파일로 pip로 필요 라이브러리 설치
RUN pip install -r requirements.txt

# EXPOSE: 컨테이너 실행 시 노출될 포트
EXPOSE 5000

# 컨테이너 시작을 위한 명령어 스크립트에 권한 부여
RUN chmod +x ./start.sh

# 컨테이너 시작 시 스크립트 실행
CMD ["./start.sh"]