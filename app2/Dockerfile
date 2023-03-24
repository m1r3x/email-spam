FROM python:3.10
RUN pip install --upgrade pip
RUN pip install torch
RUN pip install pandas

CMD [ "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]