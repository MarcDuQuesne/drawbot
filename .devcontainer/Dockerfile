# This Dockerfile sets up a development environment containing potrace 
FROM ubuntu:20.04

RUN apt-get update && \
    apt-get install -y \
    build-essential python-dev libagg-dev libpotrace-dev pkg-config
RUN pip3 install uv

RUN git clone https://github.com/flupke/pypotrace.git
RUN cd pypotrace
RUN pip3 install numpy
RUN pip3 install .

RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# RUN chsh -s $(which zsh)
