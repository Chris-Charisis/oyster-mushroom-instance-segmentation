FROM python:3.8

WORKDIR /app

RUN apt-get update && apt-get install -y curl
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN mkdir /app/images

RUN pip install torch==1.13.0 torchvision==0.14.0
RUN pip install -U openmim
RUN mim install mmcv-full==1.7.0

RUN wget https://github.com/open-mmlab/mmdetection/archive/refs/tags/v2.26.0.zip && \
	unzip v2.26.0.zip

#COPY mmdetection-2.26.0 ./mmdetection-2.26.0

RUN cd /app/mmdetection-2.26.0 && \
	pip install -v -e . && \
	mim install mmengine==0.3.1


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]