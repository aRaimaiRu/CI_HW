FROM python:3.7.9
Copy . /app
WORKDIR /app
RUN pip install -r requirement.txt
ENTRYPOINT ["python"]
CMD ["App.py"]