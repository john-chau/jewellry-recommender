FROM python:3.6
WORKDIR /app
ADD . /app
#COPY ./app.py /app
#COPY ./requirements.txt /app
#COPY ./feat_vectors_final.csv /app
#COPY ./img_reco_80_main /app
#COPY ./templates/home.html /app/templates
#COPY ./templates/results.html /app/templates
RUN pip3 install --trusted-host pypi.python.org -r requirements.txt
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
