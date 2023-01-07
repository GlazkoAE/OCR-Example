FROM pytorch/pytorch:latest

RUN pip3 install torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install matplotlib
RUN pip install colorama
