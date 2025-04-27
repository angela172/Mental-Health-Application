FROM python:3.9-slim
WORKDIR /app/
COPY . /app/
#INSTALL DEPENDENCIES
RUN pip install --no-cache-dir -r requirements.txt

#finally to run the whoooollleee thing
CMD ["streamlit","run","app.py"]
