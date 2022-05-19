# 2022 WINTER

# EDGE TPU SCHEDULER

## 1. 서버-클라이언트 연결
<img width="482" alt="image" src="https://user-images.githubusercontent.com/91827515/169298059-76875502-02e5-4bc5-af4c-921ef192e799.png">

<br/>
서버는 클라이언트의 접속을 대기하며, 클라이언트가 접속하였을 시 해당 클라이언트 전용
파이프를 생성한 후, 해당 파이프를 대기하는 Thread를 생성하여 클라이언트의 request를 관리합니다. 
<hr/>

## 2. 클라이언트 대기 Thread
<img width="217" alt="image" src="https://user-images.githubusercontent.com/91827515/169298168-c226fa37-1e58-4a9d-90b3-82060c7e548e.png">

<br/>
해당 Thread는 Client가 Request를 보내면 서버가 관리하는 Request Queue 에 요청을 추가합니다.
<hr/>

## 3. RM Scheduling
<img width="244" alt="image" src="https://user-images.githubusercontent.com/91827515/169301401-5429db46-f925-416b-a106-3474cf7f31c1.png">

<br/>
서버의 스케줄러 Thread는 요청받은 task를 수행하기 위해 request queue를 request의 rate로 sort 후 
rate가 가장 작은(priority가 가장 높은) request를 task로 가져옵니다.
<hr/>

## 4. With Preempt
<img width="485" alt="image" src="https://user-images.githubusercontent.com/91827515/169303085-d025e893-d7c5-477e-98e2-c36941d57b1b.png">

<br/>
스케줄러 Thread는 Task를 실행중에 preemption point 마다 현재 실행중인 task보다
더 priority가 높은 request가 있는 지 확인 후 없으면 현재의 Task를 마저 실행하고
,있으면 실행중인 Task의 intermedatie output을 저장 후 priority가 더 높은 request를
다음에 실행할 Task로 가져옵니다.
<hr/>

