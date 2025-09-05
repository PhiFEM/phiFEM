FROM dolfinx/dolfinx:nightly
RUN python3 -m pip install -U pip
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt
ENV PYTHONPATH="$PYTHONPATH:/home/dolfinx/shared/"
