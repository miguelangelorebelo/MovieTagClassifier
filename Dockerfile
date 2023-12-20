FROM python:3.11

RUN python -m pip install --upgrade pip
RUN apt-get update

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt


RUN echo "#!/bin/bash\n"\
         "sleep 5s\n"\
         "python create_db.py\n"\
         "python train_routine.py\n"\
         "uvicorn main:app --host 0.0.0.0 --port 8000 --workers 10" >> launch.sh

RUN chmod +x launch.sh

ENTRYPOINT [ "./launch.sh" ]