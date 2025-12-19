FROM dolfinx/dolfinx:stable

# Install netgen from source (following https://github.com/jorgensd/dolfinx-tutorial/blob/main/docker/Dockerfile)
RUN  apt-get update && \
 apt-get -y install python3 python3-tk libpython3-dev libxmu-dev tk-dev tcl-dev cmake git g++ libglu1-mesa-dev liblapacke-dev libocct-data-exchange-dev libocct-draw-dev occt-misc libtbb-dev libxi-dev
WORKDIR /ngsuite
RUN git clone --branch=v6.2.2505 --single-branch https://github.com/NGSolve/netgen.git netgen-src
WORKDIR /ngsuite/netgen-src
RUN git submodule update --init --recursive
WORKDIR /ngsuite
RUN mkdir netgen-build
RUN mkdir netgen-install
RUN cmake -DCMAKE="-cxx-flags=-flax-vector-conversions" -DCMAKE_INSTALL_PREFIX=/ngsuite/netgen-install /ngsuite/netgen-src
RUN make
RUN make install
ENV NETGENDIR=/ngsuite/netgen-install/bin
ENV PATH=$NETGENDIR:$PATH
ENV PYTHONPATH=$NETGENDIR/../lib/python3.12/site-packages:${PYTHONPATH}:$PATH

RUN python3 -m pip install -U pip
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/"