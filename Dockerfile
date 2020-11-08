FROM jlgourense/devel:pytorch


WORKDIR /home/project

COPY requirements.txt requirements.txt
COPY setup_torchdyn.cfg setup_torchdyn.cfg


RUN   pip install -r requirements.txt &&\
      git clone https://github.com/EmilienDupont/augmented-neural-odes &&\
      git clone https://github.com/DiffEqML/torchdyn &&\
      rm torchdyn/setup.cfg &&\
      mv setup_torchdyn.cfg torchdyn/setup.cfg &&\
      pip install torchdyn &&\
      rm -rf /var/lib/apt/lists/*
