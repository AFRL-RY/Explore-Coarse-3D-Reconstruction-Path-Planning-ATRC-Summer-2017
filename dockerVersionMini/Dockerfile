FROM ubuntu

ADD scripts/ /

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y apt-utils wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 git libimage-exiftool-perl cmake build-essential \
    python-dev vim
    

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda2-4.3.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

RUN apt-get install -y curl grep sed dpkg && \
    TINI_VERSION=`curl https://github.com/krallin/tini/releases/latest | grep -o "/v.*\"" | sed 's:^..\(.*\).$:\1:'` && \
    curl -L "https://github.com/krallin/tini/releases/download/v${TINI_VERSION}/tini_${TINI_VERSION}.deb" > tini.deb && \
    dpkg -i tini.deb && \
    rm tini.deb && \
    apt-get clean

ENV PATH /opt/conda/bin:$PATH

RUN pip install git+git://github.com/AndrewWalker/pydubins.git

RUN apt-get -y install software-properties-common python-software-properties 
RUN add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
RUN apt -y install gdal-bin 
RUN apt -y install python-gdal
RUN apt -y install libgl1-mesa-glx

    
RUN conda install pandas gdal scipy matplotlib numpy PIL pyyaml

RUN pip install utm plyfile munkres

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]