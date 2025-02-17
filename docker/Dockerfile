FROM ghcr.io/fenics/dolfinx/dolfinx:stable
LABEL org.opencontainers.image.description DESCRIPTION
RUN python3 -m pip install -U pip
# RUN apt-get -y update && apt-get install -y python3-venv

# Install mmg2d
# ENV BIN_DIR="/usr/local/bin/"
# WORKDIR "$BIN_DIR"
# RUN git clone https://github.com/MmgTools/mmg.git
# WORKDIR "$BIN_DIR/mmg"
# RUN mkdir build
# WORKDIR "$BIN_DIR/mmg/build"
# RUN cmake ..
# RUN make 
# ENV PATH="$PATH:/usr/local/bin/mmg/build/bin"
# RUN git clone https://github.com/alistairboyer/MarchingNumPy.git
# WORKDIR "$BIN_DIR/MarchingNumPy"
# RUN pip3 install .
# ENV PYTHONPATH="$PYTHONPATH:/usr/local/bin/MarchingNumPy/"

# Install dependencies:
# pandas: save the results
# jax:    perform automatic differentiation
# meshio: create the conforming meshes
# pygmsh: create the conforming meshes
# lxml:   change the xdmf meshes names to make meshio compatible with dolfinx
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/"
ENV MYPYPATH="$MYPYPATH:/home/dolfinx/shared/phiFEM/stubs"