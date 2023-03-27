FROM python:3.10
RUN python -m pip install --upgrade pip
RUN pip install --upgrade tensorflow
RUN pip install fastapi
RUN pip install uvicorn
RUN pip install pydantic
RUN pip install pandas
RUN pip install requests
RUN pip install torch

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "15400","--ssl-keyfile", "/app/privatekey.pem", "--ssl-certfile","/app/certificate.pem"]
