FROM python:3.8.10
RUN ["mkdir","/work_directory"]
COPY generator.py  /work_directory
WORKDIR /work_directory
CMD ["python3", "generator.py"]