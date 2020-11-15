FROM jlgourense/devel:pytorch


WORKDIR /home/project

RUN   pip install -r requirements.txt &&\
      rm -rf /var/lib/apt/lists/*
