FROM nvidia/cuda

ADD scripts/ /

RUN apt-get update && \
    apt-get -y install \
    apt-utils \
    cmake \
    build-essential \
    libboost-all-dev \
    libeigen3-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    qt5-default \
    git \
    libimage-exiftool-perl
    
 RUN apt-get install libatlas-base-dev libsuitesparse-dev git; \
    git clone https://ceres-solver.googlesource.com/ceres-solver; \
    cd ceres-solver; \
    mkdir build; \
    cd build; \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF; \
    make; \
    make install 
    
 RUN git clone https://github.com/colmap/colmap; \
    cd colmap; \
    mkdir build; \
    cd build; \
    cmake ..; \
    make

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1

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

RUN apt-get -y install software-properties-common python-software-properties 
RUN add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable
RUN apt -y install gdal-bin 
RUN apt -y install python-gdal
RUN apt -y install libgl1-mesa-glx
    
RUN conda install pandas gdal scipy matplotlib numpy PIL pyyaml

RUN pip install utm plyfile munkres
RUN pip install git+git://github.com/AndrewWalker/pydubins.git

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]