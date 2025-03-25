FROM dolfinx/dolfinx:nightly
LABEL org.opencontainers.image.description DESCRIPTION
RUN python3 -m pip install -U pip

# Install python libraries
WORKDIR /app
RUN pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/"
ENV MYPYPATH="$MYPYPATH:/home/dolfinx/shared/phiFEM/stubs"

# Download and install multiphenicsx
RUN git clone https://github.com/multiphenics/multiphenicsx.git multiphenicsx
WORKDIR /app/multiphenicsx
RUN python3 -m pip install --check-build-dependencies --no-build-isolation '.[tutorials]'