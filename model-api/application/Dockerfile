FROM python:3.7.1

COPY . .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 5000
CMD flask run -h 0.0.0.0 -p 5000 --with-threads --reload